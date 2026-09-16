"""评价指标与实验记录公共接口。"""

from .errors import (
    BacktestError,
    EvaluationError,
    EvaluationInputError,
    ExperimentStoreError,
)
from .experiments import ExperimentRecord, ExperimentStore
from .metrics import (
    MIN_METRIC_SAMPLES,
    PredictionMetrics,
    direction_accuracy,
    evaluate_prediction_frame,
    mae,
    normalize_target_dates,
    r2_score,
    relative_naive_rmse_improvement,
    rmse,
    validate_metric_inputs,
)

__all__ = [
    "BacktestError",
    "EvaluationError",
    "EvaluationInputError",
    "ExperimentRecord",
    "ExperimentStore",
    "ExperimentStoreError",
    "MIN_METRIC_SAMPLES",
    "PredictionMetrics",
    "direction_accuracy",
    "evaluate_prediction_frame",
    "mae",
    "normalize_target_dates",
    "r2_score",
    "relative_naive_rmse_improvement",
    "rmse",
    "validate_metric_inputs",
]
