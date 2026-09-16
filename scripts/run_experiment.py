"""运行六种单变量基线与固定单变量 LSTM 的统一测试集实验。"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from uuid import uuid4

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.baselines import default_baseline_models
from src.config import ConfigError, load_config
from src.data import (
    UNIVARIATE_FEATURES,
    MarketDataError,
    ScalerManager,
    build_windows,
    clean_market_data,
    create_dataloaders,
    load_market_csv,
    split_by_time,
)
from src.evaluation import EvaluationError, ExperimentRecord, ExperimentStore
from src.models import (
    LSTMError,
    LSTMPredictor,
    LSTMRegressor,
    LSTMTrainer,
    TrainerConfig,
    save_model_artifact,
    set_random_seed,
)
from src.services import UnifiedTestBacktestService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "一次运行 Naive、MA5、MA10、MA20、SES、ARIMA 与固定单变量 LSTM；"
            "验证集只用于 Early Stopping，测试集只做一次最终评价。"
        )
    )
    parser.add_argument("--file", required=True, type=Path, help="本地标准行情 CSV。")
    parser.add_argument("--ts-code", help="可选；校验 CSV 内股票代码。")
    parser.add_argument("--config", type=Path, help="配置文件，默认 configs/default.yaml。")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--experiment-id", help="可选；默认按时间和随机后缀生成。")
    parser.add_argument(
        "--result-kind",
        choices=("test", "formal"),
        default="test",
        help="默认标记为 test，避免把阶段 6 小样例误当正式实验。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="实验根目录；默认使用配置 paths.experiments_dir。",
    )
    return parser


def _write_json_atomic(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _default_experiment_id() -> str:
    now = datetime.now().astimezone()
    return f"exp_{now.strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        loaded = load_market_csv(args.file, expected_ts_code=args.ts_code)
        cleaned = clean_market_data(loaded.data, expected_ts_code=args.ts_code)
        split = split_by_time(cleaned.data, ratios=config.split_ratios)
        scaler = ScalerManager.fit(split, UNIVARIATE_FEATURES, target_column="close")
        windows = build_windows(
            split,
            scaler,
            UNIVARIATE_FEATURES,
            window_size=config.window_size,
            target_column="close",
        )
        loaders = create_dataloaders(
            windows,
            batch_size=config.lstm.batch_size,
            train_shuffle=False,
            seed=config.random_seed,
        )

        experiment_id = args.experiment_id or _default_experiment_id()
        store = ExperimentStore(args.output_dir or config.paths.experiments_dir)
        experiment_dir = store.experiment_directory(experiment_id)
        if experiment_dir.exists() and any(experiment_dir.iterdir()):
            parser.error(f"实验目录已存在且非空，请更换 --experiment-id：{experiment_dir}")
        experiment_dir.mkdir(parents=True, exist_ok=True)

        set_random_seed(config.random_seed)
        lstm = LSTMRegressor(
            input_size=len(UNIVARIATE_FEATURES),
            hidden_size=config.lstm.hidden_size,
            num_layers=config.lstm.num_layers,
            dropout=config.lstm.dropout,
        )
        trainer = LSTMTrainer(
            TrainerConfig(
                learning_rate=config.lstm.learning_rate,
                max_epochs=config.lstm.max_epochs,
                early_stopping_patience=config.lstm.early_stopping_patience,
            ),
            device=args.device,
        )
        training_started = time.perf_counter()
        training = trainer.fit(lstm, loaders["train"], loaders["validation"])
        training_elapsed = time.perf_counter() - training_started

        model_path = experiment_dir / "selected_lstm.pt"
        scaler_path = experiment_dir / "selected_lstm.scaler.json"
        history_path = experiment_dir / "training_history.json"
        save_model_artifact(
            model_path,
            lstm,
            feature_columns=UNIVARIATE_FEATURES,
            target_column="close",
            window_size=config.window_size,
            random_seed=config.random_seed,
            best_epoch=training.best_epoch,
            best_validation_loss=training.best_validation_loss,
        )
        scaler.save(scaler_path)
        _write_json_atomic(
            history_path,
            {
                "best_epoch": training.best_epoch,
                "best_validation_loss": training.best_validation_loss,
                "stopped_early": training.stopped_early,
                "total_elapsed_seconds": training.total_elapsed_seconds,
                "device": training.device,
                "history": [asdict(item) for item in training.history],
                "validation_only_for_selection_and_early_stopping": True,
                "test_samples_used_for_training_or_early_stopping": 0,
            },
        )

        target_dates = windows.test.target_dates.tolist()
        predictions: dict[str, object] = {}
        elapsed_seconds: dict[str, float] = {"LSTM_training": training_elapsed}
        model_parameters: dict[str, dict[str, object]] = {}
        for model in default_baseline_models():
            started = time.perf_counter()
            frame = model.fit(split.train).predict(split.data, target_dates)
            elapsed_seconds[model.model_name] = time.perf_counter() - started
            predictions[model.model_name] = frame
            params = model.get_params()
            if model.model_name == "SES":
                params = {**params, "fitted_smoothing_level": model.fitted_smoothing_level}
            model_parameters[model.model_name] = params

        predictor = LSTMPredictor(
            lstm,
            scaler,
            feature_columns=UNIVARIATE_FEATURES,
            target_column="close",
            window_size=config.window_size,
            device=args.device,
            model_name="LSTM",
        )
        prediction_started = time.perf_counter()
        predictions["LSTM"] = predictor.predict_samples(windows.test)
        elapsed_seconds["LSTM_prediction"] = time.perf_counter() - prediction_started
        model_parameters["LSTM"] = {
            "input_size": len(UNIVARIATE_FEATURES),
            "hidden_size": config.lstm.hidden_size,
            "num_layers": config.lstm.num_layers,
            "dropout": config.lstm.dropout,
            "learning_rate": config.lstm.learning_rate,
            "batch_size": config.lstm.batch_size,
            "max_epochs": config.lstm.max_epochs,
            "early_stopping_patience": config.lstm.early_stopping_patience,
            "loss_function": config.lstm.loss_function,
            "optimizer": config.lstm.optimizer,
            "best_epoch": training.best_epoch,
            "best_validation_loss": training.best_validation_loss,
        }

        backtest_started = time.perf_counter()
        result = UnifiedTestBacktestService(target_dates).run(predictions)  # type: ignore[arg-type]
        elapsed_seconds["backtest_and_metrics"] = time.perf_counter() - backtest_started
        elapsed_seconds["total"] = sum(
            value for key, value in elapsed_seconds.items() if key != "total"
        )

        ts_code = str(cleaned.data.iloc[0]["ts_code"])
        created_at = datetime.now().astimezone().isoformat(timespec="seconds")
        record = ExperimentRecord(
            experiment_id=experiment_id,
            created_at=created_at,
            result_kind=args.result_kind,
            status="completed",
            ts_code=ts_code,
            data_range={
                "start_date": cleaned.data.iloc[0]["trade_date"].date().isoformat(),
                "end_date": cleaned.data.iloc[-1]["trade_date"].date().isoformat(),
            },
            split_boundaries=split.boundaries_dict(),
            feature_columns=UNIVARIATE_FEATURES,
            window_size=config.window_size,
            model_parameters=model_parameters,
            random_seed=config.random_seed,
            metrics={},
            elapsed_seconds=elapsed_seconds,
            artifact_paths={
                "lstm_model": str(model_path.relative_to(store.root)),
                "scaler": str(scaler_path.relative_to(store.root)),
                "training_history": str(history_path.relative_to(store.root)),
            },
            validation_usage=("early_stopping",),
            notes=(
                "主对比仅包含 close 单变量基线与 close 单变量 LSTM。",
                "验证集只用于 Early Stopping；测试集仅在模型固定后评价一次。",
                "R² 不用于模型排名。",
            ),
        )
        saved = store.save(record, result)
        print(
            json.dumps(
                {
                    "experiment_id": saved.experiment_id,
                    "result_kind": saved.result_kind,
                    "status": saved.status,
                    "experiment_directory": str(experiment_dir),
                    "models": list(predictions),
                    "common_test_samples": len(result.alignment.common_target_dates),
                    "best_epoch": training.best_epoch,
                    "best_validation_loss": training.best_validation_loss,
                    "test_evaluation_count": saved.test_evaluation_count,
                    "formal_result_warning": (
                        "这是测试结果，不是正式实验结论。"
                        if saved.result_kind == "test"
                        else "已标记 formal；请在阶段 8 完成正式实验审查。"
                    ),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    except (MarketDataError, LSTMError, EvaluationError, ConfigError, OSError, ValueError) as exc:
        parser.exit(2, f"错误：{exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())

