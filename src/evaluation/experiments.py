"""可追溯实验记录与 JSON/CSV 产物持久化。"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence
from uuid import uuid4

import pandas as pd

from .errors import ExperimentStoreError

if TYPE_CHECKING:
    from src.services.backtesting import BacktestResult

_EXPERIMENT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_VALID_STATUS = {"started", "completed", "failed"}
_VALID_RESULT_KIND = {"test", "formal"}


@dataclass(frozen=True)
class ExperimentRecord:
    """一次实验的完整可追溯元数据。"""

    experiment_id: str
    created_at: str
    result_kind: str
    status: str
    ts_code: str
    data_range: dict[str, str]
    split_boundaries: dict[str, dict[str, object]]
    feature_columns: tuple[str, ...]
    window_size: int
    model_parameters: dict[str, dict[str, object]]
    random_seed: int
    metrics: dict[str, dict[str, int | float]]
    elapsed_seconds: dict[str, float]
    artifact_paths: dict[str, str]
    evaluation_partition: str = "test"
    validation_usage: tuple[str, ...] = ("parameter_selection", "early_stopping")
    test_evaluation_count: int = 1
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not _EXPERIMENT_ID.fullmatch(self.experiment_id):
            raise ExperimentStoreError("experiment_id 只能包含字母、数字、点、下划线和连字符。")
        if self.status not in _VALID_STATUS:
            raise ExperimentStoreError(f"未知实验状态：{self.status}。")
        if self.result_kind not in _VALID_RESULT_KIND:
            raise ExperimentStoreError("result_kind 只能是 test 或 formal。")
        if self.evaluation_partition != "test":
            raise ExperimentStoreError("最终指标只允许来自 test 分区。")
        if self.test_evaluation_count != 1:
            raise ExperimentStoreError("固定模型的测试集最终评价必须且只能记录一次。")
        if not self.ts_code.strip():
            raise ExperimentStoreError("股票代码不能为空。")
        if self.window_size <= 0 or self.random_seed <= 0:
            raise ExperimentStoreError("window_size 与 random_seed 必须为正整数。")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExperimentRecord":
        values = dict(payload)
        values["feature_columns"] = tuple(values.get("feature_columns", ()))
        values["validation_usage"] = tuple(values.get("validation_usage", ()))
        values["notes"] = tuple(values.get("notes", ()))
        return cls(**values)


class ExperimentStore:
    """在 experiments_dir/<experiment_id> 下原子保存实验产物。"""

    STANDARD_ARTIFACTS = {
        "experiment_record": "experiment_record.json",
        "experiment_config": "experiment_config.json",
        "metrics_json": "metrics.json",
        "metrics_csv": "metrics.csv",
        "daily_predictions": "daily_predictions.csv",
        "model_comparison": "model_comparison.csv",
        "alignment_report": "alignment.json",
    }

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def experiment_directory(self, experiment_id: str, *, create: bool = False) -> Path:
        if not _EXPERIMENT_ID.fullmatch(experiment_id):
            raise ExperimentStoreError("experiment_id 格式非法。")
        path = (self.root / experiment_id).resolve()
        if not path.is_relative_to(self.root):
            raise ExperimentStoreError("实验目录不能超出 experiments_dir。")
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
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

    @staticmethod
    def _write_csv_atomic(path: Path, frame: pd.DataFrame) -> None:
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            frame.to_csv(temporary, index=False, encoding="utf-8-sig")
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()

    def save(self, record: ExperimentRecord, result: "BacktestResult") -> ExperimentRecord:
        """保存配置、指标、逐日预测、比较表、对齐报告和总记录。"""

        if record.status != "completed":
            raise ExperimentStoreError("只有 completed 实验才能保存最终测试指标。")
        directory = self.experiment_directory(record.experiment_id, create=True)
        standard_paths = {
            key: str((directory / filename).relative_to(self.root))
            for key, filename in self.STANDARD_ARTIFACTS.items()
        }
        saved_record = replace(
            record,
            metrics=result.metrics_dict(),
            artifact_paths={**standard_paths, **record.artifact_paths},
        )
        config_payload = {
            "experiment_id": saved_record.experiment_id,
            "created_at": saved_record.created_at,
            "result_kind": saved_record.result_kind,
            "status": saved_record.status,
            "ts_code": saved_record.ts_code,
            "data_range": saved_record.data_range,
            "split_boundaries": saved_record.split_boundaries,
            "feature_columns": list(saved_record.feature_columns),
            "window_size": saved_record.window_size,
            "model_parameters": saved_record.model_parameters,
            "random_seed": saved_record.random_seed,
            "evaluation_partition": saved_record.evaluation_partition,
            "validation_usage": list(saved_record.validation_usage),
            "test_evaluation_count": saved_record.test_evaluation_count,
            "result_note": (
                "固定小型样例/自动测试结果，不是正式实验结论。"
                if saved_record.result_kind == "test"
                else "正式实验标记；仍须由阶段 8 的正式实验流程审查。"
            ),
        }
        metrics_payload = {
            "experiment_id": saved_record.experiment_id,
            "comparison_partition": "test",
            "ranking_basis": [
                "direction_accuracy descending",
                "relative_naive_rmse_improvement_pct descending",
                "rmse ascending",
            ],
            "r2_usage": "R² 仅作价格水平拟合度参考，不用于模型排名。",
            "metrics": result.metrics_dict(),
        }
        metrics_csv = pd.DataFrame(
            [{"model_name": name, **metrics.to_dict()} for name, metrics in result.metrics.items()]
        )
        combined = result.combined_predictions()

        self._write_json_atomic(directory / "experiment_config.json", config_payload)
        self._write_json_atomic(directory / "metrics.json", metrics_payload)
        self._write_csv_atomic(directory / "metrics.csv", metrics_csv)
        self._write_csv_atomic(directory / "daily_predictions.csv", combined)
        self._write_csv_atomic(directory / "model_comparison.csv", result.comparison)
        self._write_json_atomic(directory / "alignment.json", result.alignment.to_dict())
        self._write_json_atomic(directory / "experiment_record.json", saved_record.to_dict())
        return saved_record

    def load(self, experiment_id: str) -> ExperimentRecord:
        path = self.experiment_directory(experiment_id) / "experiment_record.json"
        if not path.is_file():
            raise ExperimentStoreError(f"实验记录不存在：{experiment_id}。")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise TypeError("实验记录根节点不是对象")
            return ExperimentRecord.from_dict(payload)
        except (OSError, ValueError, TypeError) as exc:
            raise ExperimentStoreError(f"实验记录读取失败：{path}。") from exc

    def list_artifact_files(self, experiment_id: str) -> Sequence[Path]:
        directory = self.experiment_directory(experiment_id)
        if not directory.is_dir():
            return ()
        return tuple(sorted(path for path in directory.iterdir() if path.is_file()))
