"""严格校验日期与数值输入的回归和方向评价指标。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from .errors import EvaluationInputError

MIN_METRIC_SAMPLES = 2


@dataclass(frozen=True)
class ValidatedMetricInputs:
    """已完成长度、日期、有限值与最少样本校验的指标输入。"""

    actual: np.ndarray
    predicted: np.ndarray
    target_dates: pd.DatetimeIndex
    previous_close: np.ndarray | None = None


@dataclass(frozen=True)
class PredictionMetrics:
    """一组模型在同一目标日期集合上的最终测试指标。"""

    sample_count: int
    rmse: float
    mae: float
    r2: float
    direction_accuracy: float
    relative_naive_rmse_improvement_pct: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def normalize_target_dates(
    values: Sequence[object],
    *,
    label: str = "target_dates",
    minimum_samples: int = MIN_METRIC_SAMPLES,
) -> pd.DatetimeIndex:
    """解析并检查严格升序、唯一、无缺失的目标日期。"""

    if isinstance(minimum_samples, bool) or not isinstance(minimum_samples, int) or minimum_samples <= 0:
        raise EvaluationInputError("minimum_samples 必须为正整数。")
    try:
        dates = pd.DatetimeIndex(
            pd.to_datetime(list(values), errors="raise", format="mixed")
        ).normalize()
    except (TypeError, ValueError) as exc:
        raise EvaluationInputError(f"{label} 包含无法解析的日期。") from exc
    if len(dates) < minimum_samples:
        raise EvaluationInputError(
            f"{label} 至少需要 {minimum_samples} 个样本，当前仅 {len(dates)} 个。"
        )
    if dates.hasnans:
        raise EvaluationInputError(f"{label} 包含缺失日期。")
    if dates.has_duplicates:
        raise EvaluationInputError(f"{label} 包含重复日期。")
    if not dates.is_monotonic_increasing:
        raise EvaluationInputError(f"{label} 必须严格升序，禁止日期错位或 shuffle。")
    return dates


def _numeric_vector(values: Sequence[float] | np.ndarray, *, label: str) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise EvaluationInputError(f"{label} 必须是一维数值序列。") from exc
    if array.ndim != 1:
        raise EvaluationInputError(f"{label} 必须是一维数值序列。")
    if not np.isfinite(array).all():
        raise EvaluationInputError(f"{label} 包含 NaN 或 Infinity。")
    return array


def validate_metric_inputs(
    actual: Sequence[float] | np.ndarray,
    predicted: Sequence[float] | np.ndarray,
    *,
    target_dates: Sequence[object],
    previous_close: Sequence[float] | np.ndarray | None = None,
    minimum_samples: int = MIN_METRIC_SAMPLES,
) -> ValidatedMetricInputs:
    """统一检查长度、日期、NaN、Infinity 和最少样本数。"""

    actual_array = _numeric_vector(actual, label="actual")
    predicted_array = _numeric_vector(predicted, label="predicted")
    dates = normalize_target_dates(
        target_dates, label="target_dates", minimum_samples=minimum_samples
    )
    lengths = {len(actual_array), len(predicted_array), len(dates)}
    previous_array: np.ndarray | None = None
    if previous_close is not None:
        previous_array = _numeric_vector(previous_close, label="previous_close")
        lengths.add(len(previous_array))
    if len(lengths) != 1:
        raise EvaluationInputError(
            "actual、predicted、target_dates 与 previous_close（如提供）长度必须一致。"
        )
    return ValidatedMetricInputs(
        actual=actual_array,
        predicted=predicted_array,
        target_dates=dates,
        previous_close=previous_array,
    )


def rmse(
    actual: Sequence[float] | np.ndarray,
    predicted: Sequence[float] | np.ndarray,
    *,
    target_dates: Sequence[object],
    minimum_samples: int = MIN_METRIC_SAMPLES,
) -> float:
    inputs = validate_metric_inputs(
        actual, predicted, target_dates=target_dates, minimum_samples=minimum_samples
    )
    return float(np.sqrt(np.mean(np.square(inputs.predicted - inputs.actual))))


def mae(
    actual: Sequence[float] | np.ndarray,
    predicted: Sequence[float] | np.ndarray,
    *,
    target_dates: Sequence[object],
    minimum_samples: int = MIN_METRIC_SAMPLES,
) -> float:
    inputs = validate_metric_inputs(
        actual, predicted, target_dates=target_dates, minimum_samples=minimum_samples
    )
    return float(np.mean(np.abs(inputs.predicted - inputs.actual)))


def r2_score(
    actual: Sequence[float] | np.ndarray,
    predicted: Sequence[float] | np.ndarray,
    *,
    target_dates: Sequence[object],
    minimum_samples: int = MIN_METRIC_SAMPLES,
) -> float:
    """价格水平拟合度参考；常量真实序列完美预测为 1，否则为 0。"""

    inputs = validate_metric_inputs(
        actual, predicted, target_dates=target_dates, minimum_samples=minimum_samples
    )
    residual_sum = float(np.sum(np.square(inputs.actual - inputs.predicted)))
    total_sum = float(np.sum(np.square(inputs.actual - np.mean(inputs.actual))))
    if total_sum == 0.0:
        return 1.0 if residual_sum == 0.0 else 0.0
    return float(1.0 - residual_sum / total_sum)


def direction_accuracy(
    actual: Sequence[float] | np.ndarray,
    predicted: Sequence[float] | np.ndarray,
    previous_close: Sequence[float] | np.ndarray,
    *,
    target_dates: Sequence[object],
    minimum_samples: int = MIN_METRIC_SAMPLES,
) -> float:
    """计算三分类方向准确率。

    预测方向为 sign(predicted - previous_close)，实际方向为
    sign(actual - previous_close)。符号 -1、0、1 分别表示下跌、平盘、上涨；
    只有两个符号完全一致才记为正确，因此平盘不会被并入上涨或下跌。
    """

    inputs = validate_metric_inputs(
        actual,
        predicted,
        target_dates=target_dates,
        previous_close=previous_close,
        minimum_samples=minimum_samples,
    )
    assert inputs.previous_close is not None
    predicted_direction = np.sign(inputs.predicted - inputs.previous_close)
    actual_direction = np.sign(inputs.actual - inputs.previous_close)
    return float(np.mean(predicted_direction == actual_direction))


def relative_naive_rmse_improvement(model_rmse: float, naive_rmse: float) -> float:
    """返回 ``(naive_rmse - model_rmse) / naive_rmse * 100%``。"""

    try:
        model_value = float(model_rmse)
        naive_value = float(naive_rmse)
    except (TypeError, ValueError) as exc:
        raise EvaluationInputError("model_rmse 与 naive_rmse 必须为有限数值。") from exc
    if not np.isfinite([model_value, naive_value]).all():
        raise EvaluationInputError("model_rmse 与 naive_rmse 必须为有限数值。")
    if model_value < 0 or naive_value <= 0:
        raise EvaluationInputError("model_rmse 必须非负且 naive_rmse 必须大于 0。")
    return float((naive_value - model_value) / naive_value * 100.0)


def evaluate_prediction_frame(
    frame: pd.DataFrame,
    *,
    naive_rmse: float,
    minimum_samples: int = MIN_METRIC_SAMPLES,
) -> PredictionMetrics:
    """对统一预测表计算全部指标；调用方必须先完成公共日期对齐。"""

    required = ("target_date", "previous_close", "actual", "predicted")
    if not isinstance(frame, pd.DataFrame):
        raise EvaluationInputError("预测结果必须是 pandas DataFrame。")
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise EvaluationInputError("预测结果缺少字段：" + "、".join(missing) + "。")
    inputs = validate_metric_inputs(
        frame["actual"].to_numpy(),
        frame["predicted"].to_numpy(),
        target_dates=frame["target_date"].tolist(),
        previous_close=frame["previous_close"].to_numpy(),
        minimum_samples=minimum_samples,
    )
    model_rmse = float(np.sqrt(np.mean(np.square(inputs.predicted - inputs.actual))))
    model_mae = float(np.mean(np.abs(inputs.predicted - inputs.actual)))
    residual_sum = float(np.sum(np.square(inputs.actual - inputs.predicted)))
    total_sum = float(np.sum(np.square(inputs.actual - np.mean(inputs.actual))))
    model_r2 = (1.0 if residual_sum == 0.0 else 0.0) if total_sum == 0.0 else (
        1.0 - residual_sum / total_sum
    )
    assert inputs.previous_close is not None
    model_da = float(
        np.mean(
            np.sign(inputs.predicted - inputs.previous_close)
            == np.sign(inputs.actual - inputs.previous_close)
        )
    )
    return PredictionMetrics(
        sample_count=len(inputs.actual),
        rmse=model_rmse,
        mae=model_mae,
        r2=float(model_r2),
        direction_accuracy=model_da,
        relative_naive_rmse_improvement_pct=relative_naive_rmse_improvement(
            model_rmse, naive_rmse
        ),
    )
