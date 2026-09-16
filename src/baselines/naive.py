"""前一交易记录收盘价基线。"""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import ForecastModel


class NaiveForecastModel(ForecastModel):
    """预测值恒等于目标日前最近一个已实现 close。"""

    def __init__(self) -> None:
        super().__init__()
        self._fitted_last = 0.0
        self._current_last = 0.0

    @property
    def model_name(self) -> str:
        return "Naive"

    def get_params(self) -> dict[str, Any]:
        return {}

    @property
    def minimum_observations(self) -> int:
        return 1

    def _fit_values(self, values: np.ndarray) -> None:
        self._fitted_last = float(values[-1])

    def _reset_prediction_state(self) -> None:
        self._current_last = self._fitted_last

    def _forecast_next(self) -> float:
        return self._current_last

    def _update(self, observed: float) -> None:
        self._current_last = observed

    def _forecast_static(self, steps: int) -> np.ndarray:
        return np.full(steps, self._fitted_last, dtype=np.float64)
