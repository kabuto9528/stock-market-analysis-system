"""统一 ForecastModel 接口与无未来数据的回测执行骨架。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.data import DataPartition

from .errors import (
    BaselineDataError,
    BaselineError,
    BaselineFitError,
    BaselineNotFittedError,
    BaselinePredictionError,
)

PREDICTION_COLUMNS: tuple[str, ...] = (
    "target_date",
    "previous_close",
    "actual",
    "predicted",
    "model_name",
)


class ForecastStrategy(str, Enum):
    """统一回测协议。"""

    ROLLING_ONE_STEP = "rolling_one_step"
    STATIC_MULTI_STEP = "static_multi_step"


def _normalize_strategy(strategy: ForecastStrategy | str) -> ForecastStrategy:
    try:
        return ForecastStrategy(strategy)
    except ValueError as exc:
        choices = "、".join(item.value for item in ForecastStrategy)
        raise BaselineDataError(f"未知预测策略 {strategy!r}；可选值为：{choices}。") from exc


def _normalize_dates(values: Sequence[object], *, label: str) -> pd.DatetimeIndex:
    try:
        dates = pd.DatetimeIndex(pd.to_datetime(list(values), errors="raise", format="mixed")).normalize()
    except (TypeError, ValueError) as exc:
        raise BaselineDataError(f"{label}包含无法解析的日期。") from exc
    if len(dates) == 0:
        raise BaselineDataError(f"{label}不能为空。")
    if dates.hasnans:
        raise BaselineDataError(f"{label}包含缺失日期。")
    if dates.has_duplicates:
        raise BaselineDataError(f"{label}包含重复日期。")
    if not dates.is_monotonic_increasing:
        raise BaselineDataError(f"{label}必须严格升序，禁止 shuffle。")
    return dates


def _validate_history_frame(data: pd.DataFrame, *, label: str) -> tuple[pd.DatetimeIndex, np.ndarray]:
    if not isinstance(data, pd.DataFrame):
        raise BaselineDataError(f"{label}必须是 pandas DataFrame。")
    missing = [column for column in ("trade_date", "close") if column not in data.columns]
    if missing:
        raise BaselineDataError(f"{label}缺少字段：{'、'.join(missing)}。")
    if data.empty:
        raise BaselineDataError(f"{label}为空。")
    dates = _normalize_dates(data["trade_date"].tolist(), label=f"{label} trade_date")
    try:
        close = data["close"].to_numpy(dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise BaselineDataError(f"{label} close 必须是数值。") from exc
    if close.ndim != 1 or not np.isfinite(close).all():
        raise BaselineDataError(f"{label} close 包含缺失值或正负无穷。")
    return dates, close


class ForecastModel(ABC):
    """所有基线共享的拟合、预测、字段和防泄漏控制。"""

    def __init__(self) -> None:
        self._is_fitted = False
        self._training_dates: pd.DatetimeIndex | None = None
        self._training_values: np.ndarray | None = None

    @property
    @abstractmethod
    def model_name(self) -> str:
        """稳定的模型名称。"""

    @abstractmethod
    def get_params(self) -> dict[str, Any]:
        """返回构造参数，供实验配置追溯。"""

    @property
    @abstractmethod
    def minimum_observations(self) -> int:
        """模型拟合所需的最少训练记录数。"""

    def fit(self, train_data: DataPartition) -> "ForecastModel":
        """只接受名为 train 的阶段 3 分区，禁止把验证/测试数据用于拟合。"""

        self._is_fitted = False
        if not isinstance(train_data, DataPartition) or train_data.name != "train":
            raise BaselineDataError(
                "fit 只接受 split_by_time 返回的 train DataPartition，禁止传入验证、测试或全量数据。"
            )
        dates, values = _validate_history_frame(train_data.frame, label="训练分区")
        if len(values) < self.minimum_observations:
            raise BaselineDataError(
                f"{self.model_name} 至少需要 {self.minimum_observations} 条训练记录，当前仅 {len(values)} 条。"
            )
        try:
            self._fit_values(values.copy())
        except BaselineError:
            raise
        except Exception as exc:  # 统计库异常统一转换为可读错误
            raise BaselineFitError(f"{self.model_name} 拟合失败：{exc}") from exc
        self._training_dates = dates
        self._training_values = values.copy()
        self._is_fitted = True
        return self

    def predict(
        self,
        data: pd.DataFrame,
        target_dates: Sequence[object],
        *,
        strategy: ForecastStrategy | str = ForecastStrategy.ROLLING_ONE_STEP,
        expected_trading_dates: Sequence[object] | None = None,
    ) -> pd.DataFrame:
        """按目标日期预测；滚动协议始终先预测，再接收该日真实值更新状态。"""

        if not self._is_fitted or self._training_dates is None or self._training_values is None:
            raise BaselineNotFittedError(f"{self.model_name} 尚未拟合。")
        protocol = _normalize_strategy(strategy)
        dates, close = _validate_history_frame(data, label="预测行情")
        targets = _normalize_dates(target_dates, label="target_dates")
        train_count = len(self._training_dates)
        if len(dates) <= train_count:
            raise BaselineDataError("预测行情没有训练区间之后的记录。")
        if not dates[:train_count].equals(self._training_dates):
            raise BaselineDataError("预测行情的训练日期前缀与模型拟合区间不一致。")
        if not np.array_equal(close[:train_count], self._training_values):
            raise BaselineDataError("预测行情的训练 close 与模型拟合数据不一致。")
        training_end = self._training_dates[-1]
        if targets[0] <= training_end:
            raise BaselineDataError("所有 target_date 必须严格晚于训练区间结束日期。")

        position_by_date = {date: index for index, date in enumerate(dates)}
        missing_targets = [date.date().isoformat() for date in targets if date not in position_by_date]
        if missing_targets:
            raise BaselineDataError("预测行情缺少目标日期：" + "、".join(missing_targets[:8]) + "。")
        target_positions = [position_by_date[date] for date in targets]
        if expected_trading_dates is not None:
            self._validate_expected_calendar(
                dates=dates,
                targets=targets,
                expected_trading_dates=expected_trading_dates,
            )

        try:
            if protocol is ForecastStrategy.ROLLING_ONE_STEP:
                predicted = self._rolling_predictions(close, target_positions)
            else:
                predicted = self._static_predictions(target_positions, train_count)
        except BaselineError:
            raise
        except Exception as exc:
            raise BaselinePredictionError(f"{self.model_name} 预测失败：{exc}") from exc

        previous = np.asarray([close[position - 1] for position in target_positions], dtype=np.float64)
        actual = np.asarray([close[position] for position in target_positions], dtype=np.float64)
        predicted_array = np.asarray(predicted, dtype=np.float64)
        if predicted_array.shape != (len(targets),) or not np.isfinite(predicted_array).all():
            raise BaselinePredictionError(f"{self.model_name} 返回了维度错误或非有限预测值。")
        result = pd.DataFrame(
            {
                "target_date": targets.to_numpy(dtype="datetime64[ns]"),
                "previous_close": previous,
                "actual": actual,
                "predicted": predicted_array,
                "model_name": [self.model_name] * len(targets),
            }
        )
        return result.loc[:, PREDICTION_COLUMNS]

    def _rolling_predictions(self, close: np.ndarray, target_positions: list[int]) -> list[float]:
        self._reset_prediction_state()
        target_set = set(target_positions)
        values_by_position: dict[int, float] = {}
        first_future = len(self._training_values)  # type: ignore[arg-type]
        for position in range(first_future, target_positions[-1] + 1):
            if position in target_set:
                values_by_position[position] = float(self._forecast_next())
            self._update(float(close[position]))
        return [values_by_position[position] for position in target_positions]

    def _static_predictions(self, target_positions: list[int], train_count: int) -> list[float]:
        steps = target_positions[-1] - train_count + 1
        all_forecasts = np.asarray(self._forecast_static(steps), dtype=np.float64).reshape(-1)
        if len(all_forecasts) != steps:
            raise BaselinePredictionError(
                f"{self.model_name} 静态预测应返回 {steps} 步，实际返回 {len(all_forecasts)} 步。"
            )
        return [float(all_forecasts[position - train_count]) for position in target_positions]

    def _validate_expected_calendar(
        self,
        *,
        dates: pd.DatetimeIndex,
        targets: pd.DatetimeIndex,
        expected_trading_dates: Sequence[object],
    ) -> None:
        calendar = _normalize_dates(expected_trading_dates, label="expected_trading_dates")
        training_start = self._training_dates[0]  # type: ignore[index]
        training_end = self._training_dates[-1]  # type: ignore[index]
        calendar_set = set(calendar)
        required_anchors = (training_start, training_end, *targets)
        absent_anchors = [
            date.date().isoformat() for date in required_anchors if date not in calendar_set
        ]
        if absent_anchors:
            raise BaselineDataError(
                "expected_trading_dates 必须覆盖训练起止日期和全部目标日期，缺少："
                + "、".join(absent_anchors[:8])
                + "。"
            )

        required = calendar[(calendar >= training_start) & (calendar <= targets[-1])]
        available = set(dates)
        missing = [date.date().isoformat() for date in required if date not in available]
        if missing:
            raise BaselineDataError(
                "严格交易日连续性检查失败，行情缺少预期开市日期："
                + "、".join(missing[:8])
                + "。应丢弃受影响样本或补充真实行情，禁止伪造价格。"
            )

        observed = dates[(dates >= training_start) & (dates <= targets[-1])]
        unexpected = [date.date().isoformat() for date in observed if date not in calendar_set]
        if unexpected:
            raise BaselineDataError(
                "严格交易日连续性检查失败，行情包含非预期开市日期："
                + "、".join(unexpected[:8])
                + "。"
            )

    @abstractmethod
    def _fit_values(self, values: np.ndarray) -> None:
        """仅接收训练 close。"""

    @abstractmethod
    def _reset_prediction_state(self) -> None:
        """每次预测前恢复到训练结束时状态。"""

    @abstractmethod
    def _forecast_next(self) -> float:
        """在接收当前目标真实值之前产生一步预测。"""

    @abstractmethod
    def _update(self, observed: float) -> None:
        """一步预测完成后，使用已实现的真实值更新状态。"""

    @abstractmethod
    def _forecast_static(self, steps: int) -> np.ndarray:
        """不读取训练结束后真实值的多步预测。"""
