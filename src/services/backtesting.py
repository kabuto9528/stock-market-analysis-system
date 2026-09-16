"""统一测试集回测：公共目标日期对齐、指标计算与公平排序。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from src.evaluation.errors import BacktestError, EvaluationInputError
from src.evaluation.metrics import (
    MIN_METRIC_SAMPLES,
    PredictionMetrics,
    evaluate_prediction_frame,
    normalize_target_dates,
    rmse,
)

PREDICTION_COLUMNS: tuple[str, ...] = (
    "target_date",
    "previous_close",
    "actual",
    "predicted",
    "model_name",
)


@dataclass(frozen=True)
class AlignmentReport:
    """模型日期取交集的完整审计信息。"""

    comparison_partition: str
    expected_test_target_dates: tuple[str, ...]
    original_target_dates: dict[str, tuple[str, ...]]
    excluded_target_dates: dict[str, tuple[str, ...]]
    missing_expected_target_dates: dict[str, tuple[str, ...]]
    common_target_dates: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class BacktestResult:
    """公共测试日期上的逐日预测、指标、比较表和对齐报告。"""

    predictions: dict[str, pd.DataFrame]
    metrics: dict[str, PredictionMetrics]
    comparison: pd.DataFrame
    alignment: AlignmentReport

    def combined_predictions(self) -> pd.DataFrame:
        return pd.concat(self.predictions.values(), ignore_index=True).loc[:, PREDICTION_COLUMNS]

    def metrics_dict(self) -> dict[str, dict[str, int | float]]:
        return {name: metric.to_dict() for name, metric in self.metrics.items()}


class UnifiedTestBacktestService:
    """只接受测试集目标日期，并让所有模型在日期交集上评价。"""

    def __init__(
        self,
        test_target_dates: Sequence[object],
        *,
        minimum_samples: int = MIN_METRIC_SAMPLES,
    ) -> None:
        self.minimum_samples = minimum_samples
        try:
            self.test_target_dates = normalize_target_dates(
                test_target_dates,
                label="test_target_dates",
                minimum_samples=minimum_samples,
            )
        except EvaluationInputError as exc:
            raise BacktestError(str(exc)) from exc
        self._test_date_set = set(self.test_target_dates)

    def _validate_frame(self, key: str, frame: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(frame, pd.DataFrame):
            raise BacktestError(f"{key} 预测结果必须是 pandas DataFrame。")
        missing = [column for column in PREDICTION_COLUMNS if column not in frame.columns]
        if missing:
            raise BacktestError(f"{key} 预测结果缺少字段：{'、'.join(missing)}。")
        selected = frame.loc[:, PREDICTION_COLUMNS].copy()
        try:
            dates = normalize_target_dates(
                selected["target_date"].tolist(),
                label=f"{key}.target_date",
                minimum_samples=self.minimum_samples,
            )
        except EvaluationInputError as exc:
            raise BacktestError(str(exc)) from exc
        outside = [date.date().isoformat() for date in dates if date not in self._test_date_set]
        if outside:
            raise BacktestError(
                f"{key} 包含非测试集目标日期：{'、'.join(outside[:8])}；"
                "验证集只允许调参和早停，不得混入最终评价。"
            )
        names = selected["model_name"].astype(str).str.strip().unique().tolist()
        if names != [key]:
            raise BacktestError(
                f"预测映射键 {key!r} 必须与唯一 model_name 一致，实际为 {names}。"
            )
        for column in ("previous_close", "actual", "predicted"):
            try:
                values = selected[column].to_numpy(dtype=np.float64, copy=True)
            except (TypeError, ValueError) as exc:
                raise BacktestError(f"{key}.{column} 必须为数值。") from exc
            if not np.isfinite(values).all():
                raise BacktestError(f"{key}.{column} 包含 NaN 或 Infinity。")
            selected[column] = values
        selected["target_date"] = dates.to_numpy(dtype="datetime64[ns]")
        return selected.reset_index(drop=True)

    def run(self, predictions: Mapping[str, pd.DataFrame]) -> BacktestResult:
        """在公共 test target_date 上评价；日期不一致时显式记录被排除日期。"""

        if not predictions:
            raise BacktestError("至少需要一个模型预测结果。")
        if "Naive" not in predictions:
            raise BacktestError("相对提升率要求预测集合包含名为 Naive 的基线。")
        validated = {name: self._validate_frame(name, frame) for name, frame in predictions.items()}
        date_sets = {
            name: set(pd.DatetimeIndex(frame["target_date"]))
            for name, frame in validated.items()
        }
        common = set.intersection(*date_sets.values())
        common_dates = pd.DatetimeIndex(sorted(common))
        if len(common_dates) < self.minimum_samples:
            raise BacktestError(
                f"模型公共目标日期至少需要 {self.minimum_samples} 个，当前仅 {len(common_dates)} 个。"
            )

        aligned: dict[str, pd.DataFrame] = {}
        original_target_dates: dict[str, tuple[str, ...]] = {}
        excluded_target_dates: dict[str, tuple[str, ...]] = {}
        missing_expected_target_dates: dict[str, tuple[str, ...]] = {}
        reference_actual: np.ndarray | None = None
        reference_previous: np.ndarray | None = None
        for name, frame in validated.items():
            original_dates = pd.DatetimeIndex(frame["target_date"])
            original_target_dates[name] = tuple(
                date.date().isoformat() for date in original_dates
            )
            excluded_target_dates[name] = tuple(
                date.date().isoformat() for date in original_dates if date not in common
            )
            original_set = set(original_dates)
            missing_expected_target_dates[name] = tuple(
                date.date().isoformat()
                for date in self.test_target_dates
                if date not in original_set
            )
            indexed = frame.set_index("target_date", drop=False)
            current = indexed.loc[common_dates].reset_index(drop=True).loc[:, PREDICTION_COLUMNS]
            actual = current["actual"].to_numpy(dtype=np.float64)
            previous = current["previous_close"].to_numpy(dtype=np.float64)
            if reference_actual is None:
                reference_actual = actual
                reference_previous = previous
            else:
                if not np.array_equal(actual, reference_actual):
                    raise BacktestError(f"{name} 在公共日期上的 actual 与其他模型不一致。")
                if not np.array_equal(previous, reference_previous):
                    raise BacktestError(
                        f"{name} 在公共日期上的 previous_close 与其他模型不一致。"
                    )
            aligned[name] = current

        naive_frame = aligned["Naive"]
        naive_rmse = rmse(
            naive_frame["actual"].to_numpy(),
            naive_frame["predicted"].to_numpy(),
            target_dates=naive_frame["target_date"].tolist(),
            minimum_samples=self.minimum_samples,
        )
        if naive_rmse <= 0:
            raise BacktestError("Naive RMSE 为 0，无法定义相对 Naive RMSE 提升率。")

        metrics = {
            name: evaluate_prediction_frame(
                frame,
                naive_rmse=naive_rmse,
                minimum_samples=self.minimum_samples,
            )
            for name, frame in aligned.items()
        }
        rows = [{"model_name": name, **metric.to_dict()} for name, metric in metrics.items()]
        comparison = pd.DataFrame(rows)
        comparison = comparison.sort_values(
            by=["direction_accuracy", "relative_naive_rmse_improvement_pct", "rmse", "model_name"],
            ascending=[False, False, True, True],
            kind="mergesort",
        ).reset_index(drop=True)
        comparison.insert(0, "rank", np.arange(1, len(comparison) + 1, dtype=int))

        alignment = AlignmentReport(
            comparison_partition="test",
            expected_test_target_dates=tuple(
                date.date().isoformat() for date in self.test_target_dates
            ),
            original_target_dates=original_target_dates,
            excluded_target_dates=excluded_target_dates,
            missing_expected_target_dates=missing_expected_target_dates,
            common_target_dates=tuple(date.date().isoformat() for date in common_dates),
        )
        return BacktestResult(
            predictions=aligned,
            metrics=metrics,
            comparison=comparison,
            alignment=alignment,
        )


