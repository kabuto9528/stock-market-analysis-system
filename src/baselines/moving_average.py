"""固定窗口移动平均基线。"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from .base import ForecastModel
from .errors import BaselineDataError


class MovingAverageForecastModel(ForecastModel):
    """使用目标日前最近 window 个已实现 close 的算术平均。"""

    def __init__(self, window: int) -> None:
        super().__init__()
        if isinstance(window, bool) or not isinstance(window, int) or window <= 0:
            raise BaselineDataError("移动平均 window 必须为正整数。")
        self.window = window
        self._fitted_tail: tuple[float, ...] = ()
        self._current: deque[float] = deque(maxlen=window)

    @property
    def model_name(self) -> str:
        return f"MA{self.window}"

    def get_params(self) -> dict[str, Any]:
        return {"window": self.window}

    @property
    def minimum_observations(self) -> int:
        return self.window

    def _fit_values(self, values: np.ndarray) -> None:
        self._fitted_tail = tuple(float(value) for value in values[-self.window :])

    def _reset_prediction_state(self) -> None:
        self._current = deque(self._fitted_tail, maxlen=self.window)

    def _forecast_next(self) -> float:
        return float(np.mean(self._current))

    def _update(self, observed: float) -> None:
        self._current.append(observed)

    def _forecast_static(self, steps: int) -> np.ndarray:
        history = deque(self._fitted_tail, maxlen=self.window)
        forecasts: list[float] = []
        for _ in range(steps):
            predicted = float(np.mean(history))
            forecasts.append(predicted)
            history.append(predicted)
        return np.asarray(forecasts, dtype=np.float64)
