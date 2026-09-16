"""阶段 8 正式实验编排、导出与一致性检查。"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
import yaml

from src.baselines.suite import default_baseline_models
from src.data.cleaning import clean_market_data
from src.data.csv_loader import load_market_csv
from src.data.dataset import build_windows, create_dataloaders
from src.data.scaling import ScalerManager
from src.data.splitting import TemporalSplit, split_by_time
from src.evaluation.metrics import direction_accuracy, mae, r2_score, rmse
from src.models.lstm import LSTMRegressor
from src.models.persistence import save_model_artifact
from src.models.prediction import LSTMPredictor
from src.models.reproducibility import set_random_seed
from src.models.training import LSTMTrainer, TrainerConfig
from src.services.backtesting import UnifiedTestBacktestService

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REQUIRED_PARAMETER_SPACE = {
    "window_size": [20, 30, 60],
    "hidden_size": [32, 64, 128],
    "num_layers": [1, 2],
    "learning_rate": [0.01, 0.001, 0.0001],
}


class Stage8Error(RuntimeError):
    """阶段 8 配置、数据或结果不满足正式实验约束。"""


@dataclass(frozen=True)
class StockSpec:
    ts_code: str
    name: str
    industry: str
    file: Path


@dataclass(frozen=True)
class LSTMParams:
    window_size: int = 60
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    batch_size: int = 32
    learning_rate: float = 0.001
    max_epochs: int = 100
    early_stopping_patience: int = 10
    min_delta: float = 0.0

    def varied(self, name: str, value: int | float) -> "LSTMParams":
        if name not in REQUIRED_PARAMETER_SPACE:
            raise Stage8Error(f"不允许的参数实验维度：{name}")
        return replace(self, **{name: value})

    def slug(self) -> str:
        lr = format(self.learning_rate, ".4g").replace(".", "p")
        return f"w{self.window_size}_h{self.hidden_size}_l{self.num_layers}_lr{lr}"


@dataclass(frozen=True)
class Stage8Config:
    path: Path
    protocol_version: str
    minimum_stocks: int
    preferred_stocks: int
    minimum_rows: int
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    stocks: tuple[StockSpec, ...]
    split_ratios: tuple[float, float, float]
    target_column: str
    univariate_features: tuple[str, ...]
    multivariate_features: tuple[str, ...]
    baseline_models: tuple[str, ...]
    arima_order: tuple[int, int, int]
    random_seeds: tuple[int, ...]
    tuning_seed: int
    parameter_order: tuple[str, ...]
    parameter_space: dict[str, tuple[int | float, ...]]
    defaults: LSTMParams
    output_root: Path
    tables_dir_name: str
    figures_dir_name: str
    experiments_dir_name: str


@dataclass(frozen=True)
class PreparedStock:
    spec: StockSpec
    data: pd.DataFrame
    split: TemporalSplit
    source_sha256: str
    cleaning_report: dict[str, Any]


def _project_path(value: str | Path) -> Path:
    raw = Path(value)
    resolved = raw.resolve() if raw.is_absolute() else (PROJECT_ROOT / raw).resolve()
    if not resolved.is_relative_to(PROJECT_ROOT):
        raise Stage8Error(f"阶段 8 路径不得超出项目目录：{value}")
    return resolved


def _read_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Stage8Error(f"配置项 {name} 必须是映射。")
    return value


def load_stage8_config(path: str | Path = "configs/stage8_formal.yaml") -> Stage8Config:
    config_path = _project_path(path)
    if not config_path.is_file():
        raise Stage8Error(f"阶段 8 配置不存在：{config_path}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8-sig"))
    root = _read_mapping(raw, "root")
    project = _read_mapping(root.get("project"), "project")
    data = _read_mapping(root.get("data"), "data")
    protocol = _read_mapping(root.get("protocol"), "protocol")
    defaults_raw = _read_mapping(root.get("lstm_defaults"), "lstm_defaults")
    controlled = _read_mapping(
        root.get("controlled_parameter_experiments"), "controlled_parameter_experiments"
    )
    outputs = _read_mapping(root.get("outputs"), "outputs")

    stocks: list[StockSpec] = []
    for item in data.get("stocks", []):
        entry = _read_mapping(item, "data.stocks[]")
        stocks.append(
            StockSpec(
                ts_code=str(entry["ts_code"]),
                name=str(entry["name"]),
                industry=str(entry["industry"]),
                file=_project_path(str(entry["file"])),
            )
        )
    if len(stocks) < 3:
        raise Stage8Error("候选股票配置至少需要 3 只。")

    seeds = tuple(int(value) for value in protocol.get("random_seeds", []))
    if len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise Stage8Error("正式 LSTM 随机种子必须至少 3 个且互不相同。")
    ratios = tuple(float(value) for value in protocol["split_ratios"])
    if len(ratios) != 3 or not math.isclose(sum(ratios), 1.0, abs_tol=1e-9):
        raise Stage8Error("时间划分比例必须是和为 1 的三个数。")

    order = tuple(str(value) for value in controlled.get("order", []))
    if order != tuple(REQUIRED_PARAMETER_SPACE):
        raise Stage8Error("参数实验顺序必须固定为窗口、隐藏单元、层数、学习率。")
    parameter_space: dict[str, tuple[int | float, ...]] = {}
    for name, required in REQUIRED_PARAMETER_SPACE.items():
        values = tuple(controlled.get(name, []))
        if list(values) != required:
            raise Stage8Error(f"{name} 参数范围必须严格为 {required}。")
        parameter_space[name] = values

    params = LSTMParams(
        window_size=int(defaults_raw["window_size"]),
        hidden_size=int(defaults_raw["hidden_size"]),
        num_layers=int(defaults_raw["num_layers"]),
        dropout=float(defaults_raw["dropout"]),
        batch_size=int(defaults_raw["batch_size"]),
        learning_rate=float(defaults_raw["learning_rate"]),
        max_epochs=int(defaults_raw["max_epochs"]),
        early_stopping_patience=int(defaults_raw["early_stopping_patience"]),
        min_delta=float(defaults_raw.get("min_delta", 0.0)),
    )
    return Stage8Config(
        path=config_path,
        protocol_version=str(project["protocol_version"]),
        minimum_stocks=int(data["minimum_stocks"]),
        preferred_stocks=int(data["preferred_stocks"]),
        minimum_rows=int(data["minimum_rows_per_stock"]),
        start_date=pd.Timestamp(str(data["start_date"])),
        end_date=pd.Timestamp(str(data["end_date"])),
        stocks=tuple(stocks),
        split_ratios=ratios,  # type: ignore[arg-type]
        target_column=str(protocol["target_column"]),
        univariate_features=tuple(protocol["univariate_features"]),
        multivariate_features=tuple(protocol["multivariate_features"]),
        baseline_models=tuple(protocol["baseline_models"]),
        arima_order=tuple(int(value) for value in protocol["arima_order"]),  # type: ignore[arg-type]
        random_seeds=seeds,
        tuning_seed=int(protocol["tuning_seed"]),
        parameter_order=order,
        parameter_space=parameter_space,
        defaults=params,
        output_root=_project_path(str(outputs["root"])),
        tables_dir_name=str(outputs["tables_dir"]),
        figures_dir_name=str(outputs["figures_dir"]),
        experiments_dir_name=str(outputs["experiments_dir"]),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_available_stocks(config: Stage8Config) -> tuple[list[PreparedStock], pd.DataFrame]:
    prepared: list[PreparedStock] = []
    rows: list[dict[str, Any]] = []
    for spec in config.stocks:
        provenance_path = spec.file.with_suffix(".source.json")
        provenance: dict[str, Any] = {}
        if provenance_path.is_file():
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        base = {
            "ts_code": spec.ts_code,
            "stock_name": spec.name,
            "industry": spec.industry,
            "source_file": str(spec.file.relative_to(PROJECT_ROOT)),
            "data_source": provenance.get("source", "local CSV"),
            "adjustment": provenance.get("adjustment", "unknown"),
            "fetched_at": provenance.get("fetched_at", ""),
        }
        if not spec.file.is_file():
            rows.append({**base, "status": "missing", "reason": "CSV 文件不存在"})
            continue
        try:
            loaded = load_market_csv(spec.file, expected_ts_code=spec.ts_code)
            cleaned = clean_market_data(loaded.data, expected_ts_code=spec.ts_code)
            frame = cleaned.data.loc[
                cleaned.data["trade_date"].between(config.start_date, config.end_date)
            ].reset_index(drop=True)
            if len(frame) < config.minimum_rows:
                raise Stage8Error(
                    f"有效样本 {len(frame)} 小于最低要求 {config.minimum_rows}"
                )
            split = split_by_time(frame, config.split_ratios)
            if len(split.validation.frame) < 2 or len(split.test.frame) < 2:
                raise Stage8Error("验证集或测试集样本不足。")
            if len(split.train.frame) <= max(REQUIRED_PARAMETER_SPACE["window_size"]):
                raise Stage8Error("训练集长度不足以构建最大窗口。")
            item = PreparedStock(
                spec=spec,
                data=frame,
                split=split,
                source_sha256=_sha256(spec.file),
                cleaning_report=cleaned.report.to_dict(),
            )
            prepared.append(item)
            rows.append(
                {
                    **base,
                    "status": "ready",
                    "reason": "",
                    "start_date": frame.iloc[0]["trade_date"].date().isoformat(),
                    "end_date": frame.iloc[-1]["trade_date"].date().isoformat(),
                    "sample_count": len(frame),
                    "train_count": len(split.train.frame),
                    "validation_count": len(split.validation.frame),
                    "test_count": len(split.test.frame),
                    "duplicate_rows_removed": cleaned.report.duplicate_rows_removed,
                    "observed_gap_count": len(cleaned.report.observed_date_gaps),
                    "source_sha256": item.source_sha256,
                }
            )
        except Exception as exc:
            rows.append({**base, "status": "invalid", "reason": str(exc)})
    inventory = pd.DataFrame(rows)
    return prepared, inventory


def preflight(config: Stage8Config) -> tuple[list[PreparedStock], pd.DataFrame]:
    prepared, inventory = prepare_available_stocks(config)
    if len(prepared) < config.minimum_stocks:
        states = "; ".join(
            f"{row.ts_code}:{row.status}({row.reason})" for row in inventory.itertuples()
        )
        raise Stage8Error(
            f"正式数据不足：至少需要 {config.minimum_stocks} 只有效股票，"
            f"当前仅 {len(prepared)} 只。{states}"
        )
    industries = {item.spec.industry for item in prepared}
    if len(industries) < config.minimum_stocks:
        raise Stage8Error("有效股票行业数量不足，必须覆盖至少 3 个不同行业。")
    return prepared, inventory


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    os.replace(temporary, path)


def _csv_write(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    os.replace(temporary, path)


def _relative(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")


def _metric_dict(frame: pd.DataFrame, *, naive_rmse: float | None = None) -> dict[str, Any]:
    dates = frame["target_date"].tolist()
    actual = frame["actual"].to_numpy(dtype=float)
    predicted = frame["predicted"].to_numpy(dtype=float)
    previous = frame["previous_close"].to_numpy(dtype=float)
    value_rmse = rmse(actual, predicted, target_dates=dates)
    result: dict[str, Any] = {
        "sample_count": len(frame),
        "rmse": value_rmse,
        "mae": mae(actual, predicted, target_dates=dates),
        "r2": r2_score(actual, predicted, target_dates=dates),
        "direction_accuracy": direction_accuracy(
            actual, predicted, previous, target_dates=dates
        ),
    }
    result["relative_naive_rmse_improvement_pct"] = (
        None if naive_rmse is None else (naive_rmse - value_rmse) / naive_rmse * 100.0
    )
    return result


def _history_frame(training: Any) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "epoch": item.epoch,
                "train_loss": item.train_loss,
                "validation_loss": item.validation_loss,
                "elapsed_seconds": item.elapsed_seconds,
            }
            for item in training.history
        ]
    )


def _run_lstm_experiment(
    *,
    stock: PreparedStock,
    config: Stage8Config,
    params: LSTMParams,
    features: tuple[str, ...],
    feature_mode: str,
    seed: int,
    partition: str,
    experiment_id: str,
    run_root: Path,
    device: str,
    naive_rmse: float | None = None,
    selection_axis: str | None = None,
    selection_value: int | float | None = None,
) -> dict[str, Any]:
    experiment_dir = run_root / config.experiments_dir_name / experiment_id
    experiment_dir.mkdir(parents=True, exist_ok=False)
    print(f"[LSTM] 开始 {experiment_id}", flush=True)
    started = time.perf_counter()
    scaler = ScalerManager.fit(stock.split, features, target_column=config.target_column)
    windows = build_windows(
        stock.split,
        scaler,
        features,
        window_size=params.window_size,
        target_column=config.target_column,
    )
    loaders = create_dataloaders(
        windows,
        batch_size=params.batch_size,
        train_shuffle=False,
        seed=seed,
    )
    set_random_seed(seed)
    model = LSTMRegressor(
        input_size=len(features),
        hidden_size=params.hidden_size,
        num_layers=params.num_layers,
        dropout=params.dropout,
    )
    trainer = LSTMTrainer(
        TrainerConfig(
            learning_rate=params.learning_rate,
            max_epochs=params.max_epochs,
            early_stopping_patience=params.early_stopping_patience,
            min_delta=params.min_delta,
        ),
        device=device,
    )
    training = trainer.fit(model, loaders["train"], loaders["validation"])

    model_path = save_model_artifact(
        experiment_dir / "model.pt",
        model,
        feature_columns=features,
        target_column=config.target_column,
        window_size=params.window_size,
        random_seed=seed,
        best_epoch=training.best_epoch,
        best_validation_loss=training.best_validation_loss,
    )
    scaler_path = scaler.save(experiment_dir / "scaler.json")
    history_path = experiment_dir / "training_history.csv"
    _csv_write(history_path, _history_frame(training))

    samples = windows.validation if partition == "validation" else windows.test
    predictor = LSTMPredictor(
        model,
        scaler,
        feature_columns=features,
        target_column=config.target_column,
        window_size=params.window_size,
        device=device,
        model_name=f"LSTM-{feature_mode}",
    )
    predictions = predictor.predict_samples(samples)
    prediction_path = experiment_dir / "daily_predictions.csv"
    _csv_write(prediction_path, predictions)
    metrics = _metric_dict(predictions, naive_rmse=naive_rmse)
    metrics_path = experiment_dir / "metrics.json"
    _json_dump(metrics_path, metrics)

    elapsed = time.perf_counter() - started
    record = {
        "experiment_id": experiment_id,
        "status": "completed",
        "result_kind": "formal",
        "protocol_version": config.protocol_version,
        "partition": partition,
        "test_evaluation_count": 1 if partition == "test" else 0,
        "ts_code": stock.spec.ts_code,
        "stock_name": stock.spec.name,
        "industry": stock.spec.industry,
        "feature_mode": feature_mode,
        "feature_columns": list(features),
        "random_seed": seed,
        "parameters": asdict(params),
        "selection_axis": selection_axis,
        "selection_value": selection_value,
        "split_boundaries": stock.split.boundaries_dict(),
        "data_start_date": stock.data.iloc[0]["trade_date"].date().isoformat(),
        "data_end_date": stock.data.iloc[-1]["trade_date"].date().isoformat(),
        "data_sample_count": len(stock.data),
        "source_file": _relative(stock.spec.file, PROJECT_ROOT),
        "source_sha256": stock.source_sha256,
        "best_epoch": training.best_epoch,
        "best_validation_loss": training.best_validation_loss,
        "stopped_early": training.stopped_early,
        "elapsed_seconds": elapsed,
        "device": training.device,
        "metrics": metrics,
        "artifacts": {
            "config": _relative(experiment_dir / "experiment_config.json", run_root),
            "model": _relative(model_path, run_root),
            "scaler": _relative(scaler_path, run_root),
            "training_history": _relative(history_path, run_root),
            "predictions": _relative(prediction_path, run_root),
            "metrics": _relative(metrics_path, run_root),
        },
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "notes": [
            "Scaler 仅在训练集 fit。",
            "验证集用于参数选择和早停；测试集不参与选择。",
            "保存并恢复验证损失最低 epoch 的模型。",
        ],
    }
    _json_dump(experiment_dir / "experiment_config.json", record)
    print(f"[LSTM] 完成 {experiment_id}，{elapsed:.2f}s", flush=True)
    return record


def _run_baseline_experiment(
    *, stock: PreparedStock, config: Stage8Config, run_root: Path
) -> tuple[dict[str, Any], dict[str, pd.DataFrame], dict[str, dict[str, Any]]]:
    code = stock.spec.ts_code.replace(".", "_")
    experiment_id = f"stage8_test_{code}_baselines"
    experiment_dir = run_root / config.experiments_dir_name / experiment_id
    experiment_dir.mkdir(parents=True, exist_ok=False)
    print(f"[BASELINE] 开始 {experiment_id}", flush=True)
    started = time.perf_counter()
    target_dates = pd.DatetimeIndex(stock.split.test.frame["trade_date"])
    models = default_baseline_models(arima_order=config.arima_order)
    predictions: dict[str, pd.DataFrame] = {}
    model_params: dict[str, Any] = {}
    for model in models:
        frame = model.fit(stock.split.train).predict(stock.split.data, target_dates)
        predictions[model.model_name] = frame
        model_params[model.model_name] = model.get_params()
    actual_names = tuple(predictions)
    if actual_names != config.baseline_models:
        raise Stage8Error(
            f"基线模型集合与正式协议不一致：{actual_names} != {config.baseline_models}"
        )
    result = UnifiedTestBacktestService(target_dates).run(predictions)
    combined = result.combined_predictions()
    prediction_path = experiment_dir / "daily_predictions.csv"
    _csv_write(prediction_path, combined)
    metrics = result.metrics_dict()
    metrics_path = experiment_dir / "metrics.json"
    _json_dump(metrics_path, metrics)
    model_path = experiment_dir / "baseline_models.joblib"
    joblib.dump(models, model_path)
    elapsed = time.perf_counter() - started
    record = {
        "experiment_id": experiment_id,
        "status": "completed",
        "result_kind": "formal",
        "protocol_version": config.protocol_version,
        "partition": "test",
        "test_evaluation_count": 1,
        "ts_code": stock.spec.ts_code,
        "stock_name": stock.spec.name,
        "industry": stock.spec.industry,
        "feature_mode": "univariate_baselines",
        "feature_columns": ["close"],
        "random_seed": None,
        "parameters": model_params,
        "split_boundaries": stock.split.boundaries_dict(),
        "source_file": _relative(stock.spec.file, PROJECT_ROOT),
        "source_sha256": stock.source_sha256,
        "elapsed_seconds": elapsed,
        "metrics": metrics,
        "artifacts": {
            "config": _relative(experiment_dir / "experiment_config.json", run_root),
            "model": _relative(model_path, run_root),
            "predictions": _relative(prediction_path, run_root),
            "metrics": _relative(metrics_path, run_root),
        },
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "notes": [
            "全部基线为 close 单变量滚动一步预测。",
            "测试集仅评价一次，R² 不用于排名。",
        ],
    }
    _json_dump(experiment_dir / "experiment_config.json", record)
    print(f"[BASELINE] 完成 {experiment_id}，{elapsed:.2f}s", flush=True)
    return record, predictions, metrics


def _record_row(record: Mapping[str, Any]) -> dict[str, Any]:
    metrics = record.get("metrics", {})
    artifacts = record.get("artifacts", {})
    params = record.get("parameters", {})
    return {
        "experiment_id": record["experiment_id"],
        "status": record["status"],
        "partition": record["partition"],
        "test_evaluation_count": record["test_evaluation_count"],
        "ts_code": record["ts_code"],
        "industry": record["industry"],
        "feature_mode": record["feature_mode"],
        "random_seed": record.get("random_seed"),
        "parameters_json": json.dumps(params, ensure_ascii=False, sort_keys=True),
        "rmse": metrics.get("rmse"),
        "mae": metrics.get("mae"),
        "r2": metrics.get("r2"),
        "direction_accuracy": metrics.get("direction_accuracy"),
        "relative_naive_rmse_improvement_pct": metrics.get(
            "relative_naive_rmse_improvement_pct"
        ),
        "elapsed_seconds": record.get("elapsed_seconds"),
        "prediction_file": artifacts.get("predictions"),
        "config_file": artifacts.get("config"),
        "model_file": artifacts.get("model"),
        "scaler_file": artifacts.get("scaler"),
        "training_history_file": artifacts.get("training_history"),
    }


def _aggregate_sources(group: pd.DataFrame, column: str) -> str:
    return ";".join(sorted({str(value) for value in group[column] if pd.notna(value)}))


def _build_model_metrics(
    baseline_records: Sequence[Mapping[str, Any]],
    lstm_records: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in baseline_records:
        prediction_file = record["artifacts"]["predictions"]
        for model_name, metrics in record["metrics"].items():
            rows.append(
                {
                    "row_type": "run",
                    "scope": "stock",
                    "ts_code": record["ts_code"],
                    "industry": record["industry"],
                    "model_name": model_name,
                    "feature_mode": "univariate",
                    "random_seed": None,
                    "experiment_id": record["experiment_id"],
                    "prediction_file": prediction_file,
                    **metrics,
                }
            )
    for record in lstm_records:
        rows.append(
            {
                "row_type": "run",
                "scope": "stock",
                "ts_code": record["ts_code"],
                "industry": record["industry"],
                "model_name": f"LSTM-{record['feature_mode']}",
                "feature_mode": record["feature_mode"],
                "random_seed": record["random_seed"],
                "experiment_id": record["experiment_id"],
                "prediction_file": record["artifacts"]["predictions"],
                **record["metrics"],
            }
        )
    run_frame = pd.DataFrame(rows)
    aggregate_rows: list[dict[str, Any]] = []
    metrics = [
        "rmse",
        "mae",
        "r2",
        "direction_accuracy",
        "relative_naive_rmse_improvement_pct",
    ]
    for (ts_code, industry, model_name, feature_mode), group in run_frame.groupby(
        ["ts_code", "industry", "model_name", "feature_mode"], dropna=False, sort=True
    ):
        if not model_name.startswith("LSTM-"):
            continue
        base = {
            "scope": "stock",
            "ts_code": ts_code,
            "industry": industry,
            "model_name": model_name,
            "feature_mode": feature_mode,
            "random_seed": None,
            "experiment_id": _aggregate_sources(group, "experiment_id"),
            "prediction_file": _aggregate_sources(group, "prediction_file"),
            "sample_count": int(group["sample_count"].min()),
        }
        aggregate_rows.append(
            {
                "row_type": "mean",
                **base,
                **{name: float(group[name].mean()) for name in metrics},
            }
        )
        aggregate_rows.append(
            {
                "row_type": "std",
                **base,
                **{name: float(group[name].std(ddof=1)) for name in metrics},
            }
        )
    for (model_name, feature_mode), group in run_frame.groupby(
        ["model_name", "feature_mode"], dropna=False, sort=True
    ):
        base = {
            "scope": "all_stocks",
            "ts_code": "ALL",
            "industry": "ALL",
            "model_name": model_name,
            "feature_mode": feature_mode,
            "random_seed": None,
            "experiment_id": _aggregate_sources(group, "experiment_id"),
            "prediction_file": _aggregate_sources(group, "prediction_file"),
            "sample_count": int(group["sample_count"].min()),
        }
        aggregate_rows.append(
            {
                "row_type": "mean",
                **base,
                **{name: float(group[name].mean()) for name in metrics},
            }
        )
        aggregate_rows.append(
            {
                "row_type": "std",
                **base,
                **{name: float(group[name].std(ddof=1)) for name in metrics},
            }
        )
    return pd.concat([run_frame, pd.DataFrame(aggregate_rows)], ignore_index=True)


def _build_ablation_summary(model_metrics: pd.DataFrame) -> pd.DataFrame:
    selected = model_metrics.loc[
        model_metrics["model_name"].isin(["LSTM-univariate", "LSTM-multivariate"])
        & model_metrics["row_type"].isin(["mean", "std"])
    ].copy()
    selected.insert(0, "ablation_comparison", "close vs OHLCVA")
    return selected


def run_formal_experiments(
    config: Stage8Config,
    *,
    run_id: str | None = None,
    device: str = "auto",
) -> Path:
    stocks, inventory = preflight(config)
    selected_run_id = run_id or datetime.now().strftime("formal_%Y%m%d_%H%M%S")
    if not selected_run_id.replace("_", "").replace("-", "").isalnum():
        raise Stage8Error("run_id 只能包含字母、数字、下划线和连字符。")
    run_root = (config.output_root / selected_run_id).resolve()
    if run_root.exists():
        raise Stage8Error(f"运行目录已存在，拒绝覆盖：{run_root}")
    tables_dir = run_root / config.tables_dir_name
    figures_dir = run_root / config.figures_dir_name
    experiments_dir = run_root / config.experiments_dir_name
    for directory in (tables_dir, figures_dir, experiments_dir):
        directory.mkdir(parents=True, exist_ok=False)
    _csv_write(tables_dir / "正式数据清单.csv", inventory.loc[inventory["status"] == "ready"])
    _json_dump(
        run_root / "run_manifest.json",
        {
            "run_id": selected_run_id,
            "status": "running",
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "config_file": _relative(config.path, PROJECT_ROOT),
            "config_sha256": _sha256(config.path),
            "protocol_version": config.protocol_version,
            "selected_stocks": [stock.spec.ts_code for stock in stocks],
            "selection_uses": "validation_only",
            "test_evaluation_count_per_final_experiment": 1,
        },
    )

    all_records: list[dict[str, Any]] = []
    parameter_rows: list[dict[str, Any]] = []
    locked = config.defaults
    chosen_values: dict[str, int | float] = {}
    for axis in config.parameter_order:
        axis_rows: list[dict[str, Any]] = []
        for value in config.parameter_space[axis]:
            candidate = locked.varied(axis, value)
            for stock in stocks:
                code = stock.spec.ts_code.replace(".", "_")
                value_slug = str(value).replace(".", "p")
                experiment_id = (
                    f"stage8_validation_{code}_{axis}_{value_slug}_"
                    f"{candidate.slug()}_seed{config.tuning_seed}"
                )
                record = _run_lstm_experiment(
                    stock=stock,
                    config=config,
                    params=candidate,
                    features=config.univariate_features,
                    feature_mode="univariate",
                    seed=config.tuning_seed,
                    partition="validation",
                    experiment_id=experiment_id,
                    run_root=run_root,
                    device=device,
                    selection_axis=axis,
                    selection_value=value,
                )
                all_records.append(record)
                row = {
                    "row_type": "run",
                    "parameter_name": axis,
                    "parameter_value": value,
                    "selected": False,
                    "ts_code": stock.spec.ts_code,
                    "industry": stock.spec.industry,
                    "validation_rmse": record["metrics"]["rmse"],
                    "validation_mae": record["metrics"]["mae"],
                    "validation_da": record["metrics"]["direction_accuracy"],
                    "random_seed": config.tuning_seed,
                    "experiment_id": experiment_id,
                    "prediction_file": record["artifacts"]["predictions"],
                    "parameters_json": json.dumps(asdict(candidate), sort_keys=True),
                }
                axis_rows.append(row)
                parameter_rows.append(row)
        axis_frame = pd.DataFrame(axis_rows)
        means = axis_frame.groupby("parameter_value", sort=False)["validation_rmse"].mean()
        ordered_values = list(config.parameter_space[axis])
        best_value = min(ordered_values, key=lambda item: (float(means.loc[item]), ordered_values.index(item)))
        chosen_values[axis] = best_value
        locked = locked.varied(axis, best_value)
        print(f"[SELECT] {axis}={best_value}，仅依据验证集平均 RMSE", flush=True)
        for value in ordered_values:
            group = axis_frame.loc[axis_frame["parameter_value"] == value]
            parameter_rows.append(
                {
                    "row_type": "aggregate",
                    "parameter_name": axis,
                    "parameter_value": value,
                    "selected": value == best_value,
                    "ts_code": "ALL",
                    "industry": "ALL",
                    "validation_rmse": float(group["validation_rmse"].mean()),
                    "validation_rmse_std": float(group["validation_rmse"].std(ddof=1)),
                    "validation_mae": float(group["validation_mae"].mean()),
                    "validation_da": float(group["validation_da"].mean()),
                    "random_seed": config.tuning_seed,
                    "experiment_id": _aggregate_sources(group, "experiment_id"),
                    "prediction_file": _aggregate_sources(group, "prediction_file"),
                    "parameters_json": json.dumps(
                        asdict(locked.varied(axis, value)), sort_keys=True
                    ),
                }
            )

    parameter_summary = pd.DataFrame(parameter_rows)
    _csv_write(tables_dir / "参数实验汇总.csv", parameter_summary)
    _json_dump(
        run_root / "locked_lstm_parameters.json",
        {
            "selection_partition": "validation",
            "tuning_seed": config.tuning_seed,
            "controlled_order": list(config.parameter_order),
            "selected_values": chosen_values,
            "locked_parameters": asdict(locked),
            "test_data_used_for_selection": False,
        },
    )

    baseline_records: list[dict[str, Any]] = []
    final_lstm_records: list[dict[str, Any]] = []
    for stock in stocks:
        baseline_record, _baseline_predictions, baseline_metrics = _run_baseline_experiment(
            stock=stock, config=config, run_root=run_root
        )
        baseline_records.append(baseline_record)
        all_records.append(baseline_record)
        naive_value = float(baseline_metrics["Naive"]["rmse"])
        for feature_mode, features in (
            ("univariate", config.univariate_features),
            ("multivariate", config.multivariate_features),
        ):
            for seed in config.random_seeds:
                code = stock.spec.ts_code.replace(".", "_")
                experiment_id = (
                    f"stage8_test_{code}_{feature_mode}_{locked.slug()}_seed{seed}"
                )
                record = _run_lstm_experiment(
                    stock=stock,
                    config=config,
                    params=locked,
                    features=features,
                    feature_mode=feature_mode,
                    seed=seed,
                    partition="test",
                    experiment_id=experiment_id,
                    run_root=run_root,
                    device=device,
                    naive_rmse=naive_value,
                )
                final_lstm_records.append(record)
                all_records.append(record)

    formal_summary = pd.DataFrame([_record_row(record) for record in all_records])
    _csv_write(tables_dir / "正式实验汇总.csv", formal_summary)
    model_metrics = _build_model_metrics(baseline_records, final_lstm_records)
    main_metrics = model_metrics.loc[model_metrics["feature_mode"] != "multivariate"].copy()
    _csv_write(tables_dir / "模型指标汇总.csv", main_metrics)
    ablation = _build_ablation_summary(model_metrics)
    _csv_write(tables_dir / "特征消融实验.csv", ablation)

    export_stage8_figures(
        run_root=run_root,
        config=config,
        stocks=stocks,
        parameter_summary=parameter_summary,
        model_metrics=model_metrics,
        final_records=final_lstm_records,
        baseline_records=baseline_records,
    )
    _write_analysis_document(
        run_root=run_root,
        stocks=stocks,
        model_metrics=model_metrics,
        selected_params=locked,
    )
    report = check_stage8_consistency(run_root, raise_on_error=True)
    _json_dump(
        run_root / "run_manifest.json",
        {
            "run_id": selected_run_id,
            "status": "completed",
            "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "config_file": _relative(config.path, PROJECT_ROOT),
            "config_sha256": _sha256(config.path),
            "protocol_version": config.protocol_version,
            "selected_stocks": [stock.spec.ts_code for stock in stocks],
            "locked_parameters": asdict(locked),
            "consistency_check": report,
        },
    )
    return run_root


def _xml(value: object) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _svg_line_chart(
    path: Path,
    *,
    title: str,
    x_labels: Sequence[str],
    series: Mapping[str, Sequence[float]],
    y_label: str,
) -> None:
    width, height = 1200, 680
    left, right, top, bottom = 95, 35, 70, 100
    plot_w, plot_h = width - left - right, height - top - bottom
    values = [float(v) for line in series.values() for v in line if np.isfinite(v)]
    if not values:
        raise Stage8Error(f"图表没有可绘制数值：{title}")
    low, high = min(values), max(values)
    if math.isclose(low, high):
        low -= 1.0
        high += 1.0
    padding = (high - low) * 0.08
    low, high = low - padding, high + padding
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]
    count = max(len(x_labels), 2)

    def point(index: int, value: float) -> tuple[float, float]:
        x = left + index * plot_w / (count - 1)
        y = top + (high - value) * plot_h / (high - low)
        return x, y

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="36" text-anchor="middle" font-size="24" font-family="sans-serif">{_xml(title)}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="#333"/>',
    ]
    for tick in range(6):
        value = low + (high - low) * tick / 5
        y = top + plot_h - plot_h * tick / 5
        parts.extend(
            [
                f'<line x1="{left}" y1="{y:.2f}" x2="{left+plot_w}" y2="{y:.2f}" stroke="#e5e7eb"/>',
                f'<text x="{left-10}" y="{y+5:.2f}" text-anchor="end" font-size="13" font-family="sans-serif">{value:.4g}</text>',
            ]
        )
    label_step = max(1, len(x_labels) // 8)
    for index, label in enumerate(x_labels):
        if index % label_step == 0 or index == len(x_labels) - 1:
            x, _ = point(index, low)
            parts.append(
                f'<text x="{x:.2f}" y="{top+plot_h+25}" text-anchor="middle" font-size="12" font-family="sans-serif">{_xml(label)}</text>'
            )
    for series_index, (name, line) in enumerate(series.items()):
        color = colors[series_index % len(colors)]
        coordinates = " ".join(
            f"{x:.2f},{y:.2f}" for x, y in (point(i, float(v)) for i, v in enumerate(line))
        )
        parts.append(
            f'<polyline points="{coordinates}" fill="none" stroke="{color}" stroke-width="2"/>'
        )
        legend_x = left + (series_index % 3) * 260
        legend_y = height - 48 + (series_index // 3) * 22
        parts.extend(
            [
                f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x+28}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>',
                f'<text x="{legend_x+35}" y="{legend_y+5}" font-size="14" font-family="sans-serif">{_xml(name)}</text>',
            ]
        )
    parts.append(
        f'<text x="24" y="{top+plot_h/2}" transform="rotate(-90 24 {top+plot_h/2})" text-anchor="middle" font-size="15" font-family="sans-serif">{_xml(y_label)}</text>'
    )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def _svg_bar_chart(
    path: Path,
    *,
    title: str,
    labels: Sequence[str],
    values: Sequence[float],
    errors: Sequence[float] | None,
    y_label: str,
) -> None:
    width, height = 1100, 680
    left, right, top, bottom = 100, 30, 70, 145
    plot_w, plot_h = width - left - right, height - top - bottom
    safe_errors = list(errors or [0.0] * len(values))
    maximum = max(float(v) + abs(float(e)) for v, e in zip(values, safe_errors, strict=True))
    minimum = min(0.0, min(float(v) - abs(float(e)) for v, e in zip(values, safe_errors, strict=True)))
    if math.isclose(maximum, minimum):
        maximum += 1.0
    span = maximum - minimum
    maximum += span * 0.1
    minimum -= span * 0.05
    zero_y = top + (maximum - 0.0) * plot_h / (maximum - minimum)
    slot = plot_w / max(len(labels), 1)
    bar_w = slot * 0.62
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="36" text-anchor="middle" font-size="24" font-family="sans-serif">{_xml(title)}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#333"/>',
        f'<line x1="{left}" y1="{zero_y:.2f}" x2="{left+plot_w}" y2="{zero_y:.2f}" stroke="#333"/>',
    ]
    for tick in range(6):
        value = minimum + (maximum - minimum) * tick / 5
        y = top + plot_h - plot_h * tick / 5
        parts.extend(
            [
                f'<line x1="{left}" y1="{y:.2f}" x2="{left+plot_w}" y2="{y:.2f}" stroke="#e5e7eb"/>',
                f'<text x="{left-10}" y="{y+5:.2f}" text-anchor="end" font-size="13" font-family="sans-serif">{value:.4g}</text>',
            ]
        )
    for index, (label, value, error) in enumerate(zip(labels, values, safe_errors, strict=True)):
        x = left + index * slot + (slot - bar_w) / 2
        y_value = top + (maximum - float(value)) * plot_h / (maximum - minimum)
        y = min(y_value, zero_y)
        h = abs(zero_y - y_value)
        color = "#2563eb" if float(value) >= 0 else "#dc2626"
        center = x + bar_w / 2
        error_px = abs(float(error)) * plot_h / (maximum - minimum)
        parts.extend(
            [
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{max(h,1):.2f}" fill="{color}" opacity="0.85"/>',
                f'<line x1="{center:.2f}" y1="{y_value-error_px:.2f}" x2="{center:.2f}" y2="{y_value+error_px:.2f}" stroke="#111"/>',
                f'<line x1="{center-6:.2f}" y1="{y_value-error_px:.2f}" x2="{center+6:.2f}" y2="{y_value-error_px:.2f}" stroke="#111"/>',
                f'<line x1="{center-6:.2f}" y1="{y_value+error_px:.2f}" x2="{center+6:.2f}" y2="{y_value+error_px:.2f}" stroke="#111"/>',
                f'<text x="{center:.2f}" y="{top+plot_h+24}" transform="rotate(25 {center:.2f} {top+plot_h+24})" text-anchor="start" font-size="13" font-family="sans-serif">{_xml(label)}</text>',
            ]
        )
    parts.append(
        f'<text x="25" y="{top+plot_h/2}" transform="rotate(-90 25 {top+plot_h/2})" text-anchor="middle" font-size="15" font-family="sans-serif">{_xml(y_label)}</text>'
    )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def export_stage8_figures(
    *,
    run_root: Path,
    config: Stage8Config,
    stocks: Sequence[PreparedStock],
    parameter_summary: pd.DataFrame,
    model_metrics: pd.DataFrame,
    final_records: Sequence[Mapping[str, Any]],
    baseline_records: Sequence[Mapping[str, Any]],
) -> None:
    figures = run_root / config.figures_dir_name
    for stock in stocks:
        baseline_record = next(r for r in baseline_records if r["ts_code"] == stock.spec.ts_code)
        baseline_frame = pd.read_csv(run_root / baseline_record["artifacts"]["predictions"])
        baseline_frame["target_date"] = pd.to_datetime(baseline_frame["target_date"])
        actual = baseline_frame.loc[baseline_frame["model_name"] == "Naive"].sort_values("target_date")
        series: dict[str, Sequence[float]] = {"真实值": actual["actual"].tolist()}
        for name in ("Naive", "ARIMA"):
            line = baseline_frame.loc[baseline_frame["model_name"] == name].sort_values("target_date")
            series[name] = line["predicted"].tolist()
        for mode in ("univariate", "multivariate"):
            records = [
                r for r in final_records
                if r["ts_code"] == stock.spec.ts_code and r["feature_mode"] == mode
            ]
            merged = []
            for record in records:
                frame = pd.read_csv(run_root / record["artifacts"]["predictions"])
                frame["target_date"] = pd.to_datetime(frame["target_date"])
                merged.append(frame[["target_date", "predicted"]].rename(columns={"predicted": record["experiment_id"]}))
            combined = merged[0]
            for frame in merged[1:]:
                combined = combined.merge(frame, on="target_date", how="inner", validate="one_to_one")
            series[f"LSTM-{mode}-种子均值"] = combined.drop(columns="target_date").mean(axis=1).tolist()
        _svg_line_chart(
            figures / f"真实值与预测值_{stock.spec.ts_code.replace('.', '_')}.svg",
            title=f"{stock.spec.name}（{stock.spec.ts_code}）真实值与预测值",
            x_labels=[date.date().isoformat() for date in actual["target_date"]],
            series=series,
            y_label="收盘价",
        )

    overall_mean = model_metrics.loc[
        (model_metrics["scope"] == "all_stocks") & (model_metrics["row_type"] == "mean")
    ].copy()
    overall_std = model_metrics.loc[
        (model_metrics["scope"] == "all_stocks") & (model_metrics["row_type"] == "std")
    ].set_index("model_name")
    labels = overall_mean["model_name"].tolist()
    _svg_bar_chart(
        figures / "模型RMSE.svg",
        title="模型测试集 RMSE（跨股票与随机种子）",
        labels=labels,
        values=overall_mean["rmse"].tolist(),
        errors=[float(overall_std.loc[name, "rmse"]) for name in labels],
        y_label="RMSE",
    )
    _svg_bar_chart(
        figures / "模型DA.svg",
        title="模型测试集方向准确率 DA（跨股票与随机种子）",
        labels=labels,
        values=overall_mean["direction_accuracy"].tolist(),
        errors=[float(overall_std.loc[name, "direction_accuracy"]) for name in labels],
        y_label="DA",
    )
    ablation = overall_mean.loc[overall_mean["model_name"].str.startswith("LSTM-")]
    _svg_bar_chart(
        figures / "特征消融.svg",
        title="单变量与多变量 LSTM 特征消融（测试集 RMSE）",
        labels=ablation["model_name"].tolist(),
        values=ablation["rmse"].tolist(),
        errors=[float(overall_std.loc[name, "rmse"]) for name in ablation["model_name"]],
        y_label="RMSE",
    )
    windows = parameter_summary.loc[
        (parameter_summary["row_type"] == "aggregate")
        & (parameter_summary["parameter_name"] == "window_size")
    ].copy()
    _svg_bar_chart(
        figures / "窗口对比.svg",
        title="滑动窗口控制变量实验（验证集 RMSE）",
        labels=[str(value) for value in windows["parameter_value"]],
        values=windows["validation_rmse"].tolist(),
        errors=windows["validation_rmse_std"].fillna(0.0).tolist(),
        y_label="验证集 RMSE",
    )
    first_stock = stocks[0].spec.ts_code
    for mode in ("univariate", "multivariate"):
        record = next(
            r for r in final_records
            if r["ts_code"] == first_stock
            and r["feature_mode"] == mode
            and r["random_seed"] == config.random_seeds[0]
        )
        history = pd.read_csv(run_root / record["artifacts"]["training_history"])
        _svg_line_chart(
            figures / f"训练损失_{first_stock.replace('.', '_')}_{mode}.svg",
            title=f"{first_stock} {mode} LSTM 训练损失",
            x_labels=[str(value) for value in history["epoch"]],
            series={
                "训练损失": history["train_loss"].tolist(),
                "验证损失": history["validation_loss"].tolist(),
            },
            y_label="MSE Loss",
        )



def _write_analysis_document(
    *,
    run_root: Path,
    stocks: Sequence[PreparedStock],
    model_metrics: pd.DataFrame,
    selected_params: LSTMParams,
) -> None:
    overall = model_metrics.loc[
        (model_metrics["scope"] == "all_stocks") & (model_metrics["row_type"] == "mean")
    ].set_index("model_name")
    uni = overall.loc["LSTM-univariate"]
    multi = overall.loc["LSTM-multivariate"]
    naive = overall.loc["Naive"]
    lstm_better = float(uni["rmse"]) < float(naive["rmse"])
    multi_better = float(multi["rmse"]) < float(uni["rmse"])
    stock_means = model_metrics.loc[
        (model_metrics["scope"] == "stock")
        & (model_metrics["row_type"] == "mean")
        & (model_metrics["model_name"].isin(["LSTM-univariate", "LSTM-multivariate"]))
    ]
    spread_lines = []
    for stock in stocks:
        current = stock_means.loc[stock_means["ts_code"] == stock.spec.ts_code].set_index(
            "model_name"
        )
        spread_lines.append(
            f"- {stock.spec.ts_code}（{stock.spec.industry}）：单变量 LSTM RMSE "
            f"{current.loc['LSTM-univariate', 'rmse']:.6g}，DA "
            f"{current.loc['LSTM-univariate', 'direction_accuracy']:.2%}；多变量 LSTM RMSE "
            f"{current.loc['LSTM-multivariate', 'rmse']:.6g}，DA "
            f"{current.loc['LSTM-multivariate', 'direction_accuracy']:.2%}。"
        )
    conclusion_lstm = (
        "跨股票与固定随机种子均值下，单变量 LSTM 的 RMSE 低于 Naive。"
        if lstm_better
        else "跨股票与固定随机种子均值下，单变量 LSTM 的 RMSE 未低于 Naive，不能声称 LSTM 稳定优于 Naive。"
    )
    stock_pivot = stock_means.pivot(index="ts_code", columns="model_name")
    rmse_improved_count = int(
        (stock_pivot["rmse"]["LSTM-multivariate"] < stock_pivot["rmse"]["LSTM-univariate"]).sum()
    )
    da_improved_count = int(
        (
            stock_pivot["direction_accuracy"]["LSTM-multivariate"]
            > stock_pivot["direction_accuracy"]["LSTM-univariate"]
        ).sum()
    )
    if rmse_improved_count == len(stocks) and da_improved_count == len(stocks):
        conclusion_multi = "多变量 LSTM 在全部股票上同时改善 RMSE 与 DA，可认为在本样本内表现稳定。"
    else:
        conclusion_multi = (
            f"多变量 LSTM 在 {rmse_improved_count}/{len(stocks)} 只股票上降低 RMSE，"
            f"但仅在 {da_improved_count}/{len(stocks)} 只股票上提高 DA；"
            "因此不能认为新增 OHLCVA 特征稳定有效。"
        )
    doc = f"""# 实验结果分析

> 本文档由阶段 8 正式实验脚本从可追溯 CSV 自动生成。运行目录：`{_relative(run_root, PROJECT_ROOT)}`。

## 1. 正式数据与协议

- 股票数量：{len(stocks)}；行业数量：{len({item.spec.industry for item in stocks})}。
- 所有股票统一采用 70% / 15% / 15% 时间顺序划分，不 shuffle。
- Scaler 仅在训练集拟合；参数选择和 Early Stopping 仅使用验证集；锁定后测试集每个最终实验只评价一次。
- 锁定参数：`{json.dumps(asdict(selected_params), ensure_ascii=False)}`。
- 主比较只包含 Naive、MA5、MA10、MA20、SES、ARIMA 与单变量 LSTM；多变量 LSTM 仅进入特征消融。

## 2. LSTM 与 Naive

{conclusion_lstm}

单变量 LSTM 跨股票/种子平均 RMSE 为 {float(uni['rmse']):.6g}，Naive 为 {float(naive['rmse']):.6g}；单变量 LSTM 平均 DA 为 {float(uni['direction_accuracy']):.2%}，相对 Naive RMSE 提升率均值为 {float(uni['relative_naive_rmse_improvement_pct']):.2f}%。模型判断以 DA 与相对 Naive 提升率为主，R² 只作为价格水平拟合度参考，不参与排名。

## 3. 特征消融

{conclusion_multi}

多变量 LSTM 平均 RMSE 为 {float(multi['rmse']):.6g}，平均 DA 为 {float(multi['direction_accuracy']):.2%}；单变量对应值为 {float(uni['rmse']):.6g} 与 {float(uni['direction_accuracy']):.2%}。是否“稳定有效”还必须查看 `特征消融实验.csv` 中逐股票均值与标准差，不能只依据总体平均值。

## 4. 不同股票差异

{chr(10).join(spread_lines)}

股票之间价格尺度、波动率、行业属性与有效样本区间不同，RMSE 的绝对值不宜跨股票直接解释为经济意义上的优劣；应同时观察相对 Naive 提升率与 DA。

## 5. 研究局限

1. 研究只覆盖配置中的股票与时间区间，不能外推到全部 A 股或未来市场状态。
2. 参数实验是预先限定范围的控制变量实验，不是全局最优搜索，变量间交互可能未被充分覆盖。
3. 多变量输入只含 OHLCVA，未包含公司基本面、宏观变量、新闻情绪和复权因子敏感性分析。
4. 固定随机种子能量化部分训练波动，但不能消除市场制度变化、样本选择与模型设定偏差。
5. 预测结果用于教学和研究，不构成投资建议；高 R² 不代表方向预测或交易收益优秀。

## 6. 追溯入口

- `tables/正式实验汇总.csv`：每次实验的配置、实验 ID 与工件路径。
- `tables/模型指标汇总.csv`：公平主比较及随机种子均值/标准差。
- `tables/参数实验汇总.csv`：验证集控制变量选择过程。
- `tables/特征消融实验.csv`：单变量/多变量结果。
- 每个表中的 `experiment_id` 与 `prediction_file` 可追溯到逐日预测原文件。
"""
    (PROJECT_ROOT / "docs" / "实验结果分析.md").write_text(doc, encoding="utf-8")


def check_stage8_consistency(
    run_root: str | Path, *, raise_on_error: bool = False
) -> dict[str, Any]:
    root = Path(run_root).resolve()
    errors: list[str] = []
    checks: list[str] = []
    required = {
        "formal": root / "tables" / "正式实验汇总.csv",
        "metrics": root / "tables" / "模型指标汇总.csv",
        "parameters": root / "tables" / "参数实验汇总.csv",
        "ablation": root / "tables" / "特征消融实验.csv",
        "inventory": root / "tables" / "正式数据清单.csv",
        "locked": root / "locked_lstm_parameters.json",
    }
    for name, path in required.items():
        if not path.is_file():
            errors.append(f"缺少 {name} 文件：{path}")
    if errors:
        report = {"status": "failed", "errors": errors, "checks": checks}
        if raise_on_error:
            raise Stage8Error("；".join(errors))
        return report

    formal = pd.read_csv(required["formal"])
    metrics = pd.read_csv(required["metrics"])
    parameters = pd.read_csv(required["parameters"])
    ablation = pd.read_csv(required["ablation"])
    inventory = pd.read_csv(required["inventory"])
    locked = json.loads(required["locked"].read_text(encoding="utf-8"))

    if len(inventory) < 3 or inventory["industry"].nunique() < 3:
        errors.append("正式数据清单不足 3 只股票或不足 3 个行业。")
    else:
        checks.append("正式数据覆盖至少 3 只、3 个行业。")
    if inventory["source_sha256"].isna().any():
        errors.append("正式数据清单缺少源文件 SHA-256。")

    experiment_ids = set(formal["experiment_id"].astype(str))
    for row in formal.itertuples(index=False):
        for column in ("config_file", "model_file", "prediction_file"):
            value = getattr(row, column)
            if pd.isna(value) or not (root / str(value)).is_file():
                errors.append(f"{row.experiment_id} 缺少可追溯工件 {column}={value}")
        if str(row.feature_mode) in {"univariate", "multivariate"}:
            for column in ("scaler_file", "training_history_file"):
                value = getattr(row, column)
                if pd.isna(value) or not (root / str(value)).is_file():
                    errors.append(f"{row.experiment_id} 缺少 LSTM 工件 {column}={value}")
        if row.partition == "test" and int(row.test_evaluation_count) != 1:
            errors.append(f"{row.experiment_id} 测试集评价次数不是 1。")
        if row.partition == "validation" and int(row.test_evaluation_count) != 0:
            errors.append(f"{row.experiment_id} 验证实验错误记录了测试评价。")
    checks.append("逐实验配置、模型、Scaler、历史、预测与测试评价次数已检查。")

    parameter_run_ids = set(
        parameters.loc[parameters["row_type"] == "run", "experiment_id"].astype(str)
    )
    validation_ids = set(formal.loc[formal["partition"] == "validation", "experiment_id"].astype(str))
    if not parameter_run_ids.issubset(validation_ids):
        errors.append("参数实验包含非验证集实验 ID。")
    if locked.get("test_data_used_for_selection") is not False:
        errors.append("锁定参数文件未明确声明测试集不参与选择。")
    selected = parameters.loc[
        (parameters["row_type"] == "aggregate") & (parameters["selected"] == True)  # noqa: E712
    ]
    if set(selected["parameter_name"]) != set(REQUIRED_PARAMETER_SPACE):
        errors.append("每个控制变量维度没有且仅有一个验证集选择值。")
    checks.append("参数选择仅引用验证集，且范围为预先限定的控制变量集合。")

    if (metrics["feature_mode"] == "multivariate").any():
        errors.append("模型指标主比较表混入了多变量 LSTM。")
    if not (ablation["feature_mode"] == "multivariate").any():
        errors.append("特征消融表缺少多变量 LSTM。")
    checks.append("主比较公平性与特征消融分表已检查。")

    final = formal.loc[
        (formal["partition"] == "test")
        & (formal["feature_mode"].isin(["univariate", "multivariate"]))
    ]
    for (ts_code, mode), group in final.groupby(["ts_code", "feature_mode"]):
        seeds = {int(value) for value in group["random_seed"]}
        if len(seeds) < 3:
            errors.append(f"{ts_code}/{mode} 的固定随机种子少于 3 个。")
    expected_seeds = set(load_stage8_config().random_seeds)
    for (ts_code, mode), group in final.groupby(["ts_code", "feature_mode"]):
        seeds = {int(value) for value in group["random_seed"]}
        if seeds != expected_seeds:
            errors.append(f"{ts_code}/{mode} 的随机种子集合与正式配置不一致：{sorted(seeds)}")
    if final["parameters_json"].nunique() != 1:
        errors.append("最终 LSTM 重复运行没有使用完全一致的锁定超参数。")
    validation_runs = parameters.loc[parameters["row_type"] == "run"]
    for (name, value), group in validation_runs.groupby(["parameter_name", "parameter_value"]):
        if group["parameters_json"].nunique() != 1:
            errors.append(f"参数实验 {name}={value} 在不同股票间配置不一致。")
    checks.append("最终随机种子集合与重复运行配置一致性已检查。")

    for ts_code, group in formal.loc[formal["partition"] == "test"].groupby("ts_code"):
        reference_dates: tuple[str, ...] | None = None
        reference_actual: tuple[float, ...] | None = None
        reference_previous: tuple[float, ...] | None = None
        for row in group.itertuples(index=False):
            frame = pd.read_csv(root / str(row.prediction_file))
            for _, model_frame in frame.groupby("model_name", sort=False):
                ordered = model_frame.sort_values("target_date")
                dates = tuple(pd.to_datetime(ordered["target_date"]).dt.date.astype(str))
                actual = tuple(ordered["actual"].astype(float))
                previous = tuple(ordered["previous_close"].astype(float))
                if reference_dates is None:
                    reference_dates, reference_actual, reference_previous = dates, actual, previous
                elif dates != reference_dates or actual != reference_actual or previous != reference_previous:
                    errors.append(f"{ts_code} 的模型目标日期或实际值集合不一致：{row.experiment_id}")
    checks.append("同一股票所有模型的测试目标日期、actual 与 previous_close 已检查。")

    run_metrics = metrics.loc[metrics["row_type"] == "run"].copy()
    naive_by_stock = {
        row.ts_code: float(row.rmse)
        for row in run_metrics.loc[run_metrics["model_name"] == "Naive"].itertuples()
    }
    for row in run_metrics.itertuples(index=False):
        path = root / str(row.prediction_file)
        if not path.is_file():
            errors.append(f"指标行缺少预测文件：{row.experiment_id}")
            continue
        frame = pd.read_csv(path)
        if "model_name" in frame and frame["model_name"].nunique() > 1:
            frame = frame.loc[frame["model_name"] == row.model_name]
        calculated = _metric_dict(frame, naive_rmse=naive_by_stock.get(row.ts_code))
        formal_row = formal.loc[formal["experiment_id"] == row.experiment_id].iloc[0]
        config_payload = json.loads((root / str(formal_row["config_file"])).read_text(encoding="utf-8"))
        metrics_path = root / config_payload["artifacts"]["metrics"]
        json_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if row.model_name in json_metrics:
            json_metrics = json_metrics[row.model_name]
        for key in ("rmse", "mae", "r2", "direction_accuracy"):
            csv_value = float(getattr(row, key))
            if not math.isclose(csv_value, float(calculated[key]), rel_tol=1e-9, abs_tol=1e-9):
                errors.append(f"{row.experiment_id}/{row.model_name} 的 {key} 与原始预测不一致。")
            if not math.isclose(csv_value, float(json_metrics[key]), rel_tol=1e-9, abs_tol=1e-9):
                errors.append(f"{row.experiment_id}/{row.model_name} 的 CSV 与 JSON {key} 不一致。")
    checks.append("模型指标已从逐日预测重算，并核对 CSV 与 JSON 一致性。")

    for table_name, frame in (("参数实验", parameters), ("模型指标", metrics), ("特征消融", ablation)):
        for row in frame.itertuples(index=False):
            ids = str(row.experiment_id).split(";")
            files = str(row.prediction_file).split(";")
            if any(item not in experiment_ids for item in ids):
                errors.append(f"{table_name}表含未知 experiment_id：{row.experiment_id}")
            if any(not (root / item).is_file() for item in files):
                errors.append(f"{table_name}表含不存在 prediction_file：{row.prediction_file}")
    checks.append("论文表格 experiment_id 与原始预测路径追溯已检查。")

    report = {
        "status": "passed" if not errors else "failed",
        "checked_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "run_root": str(root),
        "checks": checks,
        "errors": errors,
    }
    _json_dump(root / "consistency_report.json", report)
    if errors and raise_on_error:
        raise Stage8Error("结果一致性检查失败：" + "；".join(errors))
    return report



