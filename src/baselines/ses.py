"""简单指数平滑（SES）基线。"""

from __future__ import annotations

from typing import Any

import numpy as np
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

from .base import ForecastModel
from .errors import BaselineDataError, BaselineFitError


class SimpleExponentialSmoothingModel(ForecastModel):
    """只在训练集估计平滑系数，滚动时参数固定、状态按已实现值更新。"""

    def __init__(self, smoothing_level: float | None = None) -> None:
        super().__init__()
        if smoothing_level is not None and (
            not np.isfinite(smoothing_level) or not 0 < smoothing_level <= 1
        ):
            raise BaselineDataError("SES smoothing_level 必须位于 (0, 1]，或设为 None 仅在训练集估计。")
        self.smoothing_level = smoothing_level
        self._fitted_alpha = 0.0
        self._fitted_level = 0.0
        self._current_level = 0.0

    @property
    def model_name(self) -> str:
        return "SES"

    def get_params(self) -> dict[str, Any]:
        return {"smoothing_level": self.smoothing_level}

    @property
    def minimum_observations(self) -> int:
        return 3

    @property
    def fitted_smoothing_level(self) -> float:
        if not self._is_fitted:
            raise BaselineFitError("SES 尚未拟合，无法读取平滑系数。")
        return self._fitted_alpha

    def _fit_values(self, values: np.ndarray) -> None:
        model = SimpleExpSmoothing(values, initialization_method="estimated")
        if self.smoothing_level is None:
            result = model.fit(optimized=True, remove_bias=False)
        else:
            result = model.fit(
                smoothing_level=self.smoothing_level,
                optimized=False,
                remove_bias=False,
            )
        alpha = float(result.params["smoothing_level"])
        level = float(np.asarray(result.level, dtype=np.float64)[-1])
        if not np.isfinite(alpha) or not np.isfinite(level):
            raise BaselineFitError("SES 拟合得到非有限平滑参数或状态。")
        self._fitted_alpha = alpha
        self._fitted_level = level

    def _reset_prediction_state(self) -> None:
        self._current_level = self._fitted_level

    def _forecast_next(self) -> float:
        return self._current_level

    def _update(self, observed: float) -> None:
        self._current_level = (
            self._fitted_alpha * observed + (1.0 - self._fitted_alpha) * self._current_level
        )

    def _forecast_static(self, steps: int) -> np.ndarray:
        return np.full(steps, self._fitted_level, dtype=np.float64)
