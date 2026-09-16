"""从本地 CSV 训练单变量或多变量 LSTM，并保存最佳模型与独立 Scaler。"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ConfigError, load_config
from src.data import (
    MULTIVARIATE_FEATURES,
    UNIVARIATE_FEATURES,
    MarketDataError,
    ScalerManager,
    build_windows,
    clean_market_data,
    create_dataloaders,
    load_market_csv,
    split_by_time,
)
from src.models import (
    LSTMError,
    LSTMRegressor,
    LSTMTrainer,
    TrainerConfig,
    save_model_artifact,
    set_random_seed,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="从本地标准行情 CSV 训练 LSTM；Early Stopping 仅使用验证集。"
    )
    parser.add_argument("--file", required=True, type=Path, help="本地行情 CSV 文件。")
    parser.add_argument("--ts-code", help="可选；校验 CSV 内股票代码。")
    parser.add_argument(
        "--mode", choices=("univariate", "multivariate"), default="univariate"
    )
    parser.add_argument("--config", type=Path, help="配置文件，默认 configs/default.yaml。")
    parser.add_argument("--output-dir", type=Path, help="输出目录，默认配置中的 models_dir。")
    parser.add_argument("--name", help="输出文件前缀；默认由股票代码和模式生成。")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖同名模型产物。")
    return parser


def _write_json_atomic(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        features = (
            UNIVARIATE_FEATURES if args.mode == "univariate" else MULTIVARIATE_FEATURES
        )
        loaded = load_market_csv(args.file, expected_ts_code=args.ts_code)
        cleaned = clean_market_data(loaded.data, expected_ts_code=args.ts_code)
        split = split_by_time(cleaned.data, ratios=config.split_ratios)
        scaler = ScalerManager.fit(split, features, target_column="close")
        windows = build_windows(
            split,
            scaler,
            features,
            window_size=config.window_size,
            target_column="close",
        )
        set_random_seed(config.random_seed)
        loaders = create_dataloaders(
            windows,
            batch_size=config.lstm.batch_size,
            train_shuffle=False,
            seed=config.random_seed,
        )
        model = LSTMRegressor(
            input_size=len(features),
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
        result = trainer.fit(model, loaders["train"], loaders["validation"])

        output_dir = (args.output_dir or config.paths.models_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        ts_code = str(cleaned.data.iloc[0]["ts_code"])
        safe_code = "".join(char if char.isalnum() else "_" for char in ts_code)
        prefix = args.name or f"{safe_code}_{args.mode}_lstm"
        model_path = output_dir / f"{prefix}.pt"
        scaler_path = output_dir / f"{prefix}.scaler.json"
        history_path = output_dir / f"{prefix}.history.json"
        existing = [path for path in (model_path, scaler_path, history_path) if path.exists()]
        if existing and not args.overwrite:
            names = "、".join(str(path) for path in existing)
            parser.error(f"输出文件已存在，请更换 --name 或显式使用 --overwrite：{names}")

        save_model_artifact(
            model_path,
            model,
            feature_columns=features,
            target_column="close",
            window_size=config.window_size,
            random_seed=config.random_seed,
            best_epoch=result.best_epoch,
            best_validation_loss=result.best_validation_loss,
        )
        scaler.save(scaler_path)
        _write_json_atomic(
            history_path,
            {
                "best_epoch": result.best_epoch,
                "best_validation_loss": result.best_validation_loss,
                "stopped_early": result.stopped_early,
                "total_elapsed_seconds": result.total_elapsed_seconds,
                "device": result.device,
                "history": [asdict(record) for record in result.history],
            },
        )
        print(
            json.dumps(
                {
                    "model_path": str(model_path),
                    "scaler_path": str(scaler_path),
                    "history_path": str(history_path),
                    "mode": args.mode,
                    "feature_columns": list(features),
                    "window_size": config.window_size,
                    "best_epoch": result.best_epoch,
                    "best_validation_loss": result.best_validation_loss,
                    "epochs_ran": len(result.history),
                    "device": result.device,
                    "test_samples_used_for_training_or_early_stopping": 0,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    except (MarketDataError, LSTMError, ConfigError, OSError, ValueError) as exc:
        parser.exit(2, f"错误：{exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
