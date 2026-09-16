"""有限固定阶数的 ARIMA 基线。"""

from __future__ import annotations

from typing import Any

import numpy as np
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults

from .base import ForecastModel
from .errors import BaselineDataError, BaselineFitError, BaselinePredictionError


class ARIMAForecastModel(ForecastModel):
    """ARIMA 参数只在训练集拟合；滚动预测 append 真实历史但不重新估计参数。"""

    def __init__(self, order: tuple[int, int, int] = (1, 1, 0)) -> None:
        super().__init__()
        if (
            not isinstance(order, tuple)
            or len(order) != 3
            or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in order)
        ):
            raise BaselineDataError("ARIMA order 必须是三个非负整数构成的元组 (p, d, q)。")
        self.order = order
        self._fit_result: ARIMAResults | None = None
        self._prediction_result: ARIMAResults | None = None

    @property
    def model_name(self) -> str:
        return "ARIMA"

    def get_params(self) -> dict[str, Any]:
        return {"order": self.order}

    @property
    def minimum_observations(self) -> int:
        return max(8, sum(self.order) + 3)

    def _fit_values(self, values: np.ndarray) -> None:
        try:
            self._fit_result = ARIMA(values, order=self.order, trend=None).fit()
        except Exception as exc:
            raise BaselineFitError(f"ARIMA{self.order} 拟合失败：{exc}") from exc
        if self._fit_result is None:
            raise BaselineFitError("ARIMA 拟合未返回结果。")

    def _reset_prediction_state(self) -> None:
        if self._fit_result is None:
            raise BaselineFitError("ARIMA 拟合结果不存在。")
        self._prediction_result = self._fit_result

    def _forecast_next(self) -> float:
        if self._prediction_result is None:
            raise BaselinePredictionError("ARIMA 预测状态尚未初始化。")
        value = float(np.asarray(self._prediction_result.forecast(steps=1), dtype=np.float64)[0])
        return value

    def _update(self, observed: float) -> None:
        if self._prediction_result is None:
            raise BaselinePredictionError("ARIMA 预测状态尚未初始化。")
        try:
            self._prediction_result = self._prediction_result.append([observed], refit=False)
        except Exception as exc:
            raise BaselinePredictionError(f"ARIMA 滚动状态更新失败：{exc}") from exc

    def _forecast_static(self, steps: int) -> np.ndarray:
        if self._fit_result is None:
            raise BaselineFitError("ARIMA 拟合结果不存在。")
        return np.asarray(self._fit_result.forecast(steps=steps), dtype=np.float64)
