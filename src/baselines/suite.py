"""六种默认基线的统一执行与日期对齐检查。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import pandas as pd

from src.data import DataPartition

from .arima import ARIMAForecastModel
from .base import PREDICTION_COLUMNS, ForecastModel, ForecastStrategy
from .errors import BaselineDataError
from .moving_average import MovingAverageForecastModel
from .naive import NaiveForecastModel
from .ses import SimpleExponentialSmoothingModel


@dataclass(frozen=True)
class BaselineSuiteResult:
    """按模型名保存字段一致、目标日期一致的预测表。"""

    predictions: dict[str, pd.DataFrame]

    @property
    def model_names(self) -> tuple[str, ...]:
        return tuple(self.predictions)

    @property
    def target_dates(self) -> pd.DatetimeIndex:
        first = next(iter(self.predictions.values()))
        return pd.DatetimeIndex(first["target_date"])

    def combined(self) -> pd.DataFrame:
        return pd.concat(self.predictions.values(), ignore_index=True).loc[:, PREDICTION_COLUMNS]


def default_baseline_models(
    *,
    ses_smoothing_level: float | None = None,
    arima_order: tuple[int, int, int] = (1, 1, 0),
) -> tuple[ForecastModel, ...]:
    """返回 Naive、MA5、MA10、MA20、SES、ARIMA 六种有限配置基线。"""

    return (
        NaiveForecastModel(),
        MovingAverageForecastModel(5),
        MovingAverageForecastModel(10),
        MovingAverageForecastModel(20),
        SimpleExponentialSmoothingModel(ses_smoothing_level),
        ARIMAForecastModel(arima_order),
    )


def run_baseline_suite(
    models: Iterable[ForecastModel],
    train_data: DataPartition,
    data: pd.DataFrame,
    target_dates: Sequence[object],
    *,
    strategy: ForecastStrategy | str = ForecastStrategy.ROLLING_ONE_STEP,
    expected_trading_dates: Sequence[object] | None = None,
) -> BaselineSuiteResult:
    """使用同一训练分区、协议和目标日期运行全部模型并强制对齐。"""

    model_list = tuple(models)
    if not model_list:
        raise BaselineDataError("基线模型列表不能为空。")
    names = [model.model_name for model in model_list]
    if len(names) != len(set(names)):
        raise BaselineDataError("基线模型名称必须唯一。")

    predictions: dict[str, pd.DataFrame] = {}
    reference_dates: pd.DatetimeIndex | None = None
    reference_actual: pd.Series | None = None
    reference_previous: pd.Series | None = None
    for model in model_list:
        frame = model.fit(train_data).predict(
            data,
            target_dates,
            strategy=strategy,
            expected_trading_dates=expected_trading_dates,
        )
        if tuple(frame.columns) != PREDICTION_COLUMNS:
            raise BaselineDataError(f"{model.model_name} 预测字段不符合统一结构。")
        dates = pd.DatetimeIndex(frame["target_date"])
        if reference_dates is None:
            reference_dates = dates
            reference_actual = frame["actual"].reset_index(drop=True)
            reference_previous = frame["previous_close"].reset_index(drop=True)
        else:
            if not dates.equals(reference_dates):
                raise BaselineDataError(f"{model.model_name} 的目标日期与其他基线不一致。")
            if not frame["actual"].reset_index(drop=True).equals(reference_actual):
                raise BaselineDataError(f"{model.model_name} 的 actual 与其他基线不一致。")
            if not frame["previous_close"].reset_index(drop=True).equals(reference_previous):
                raise BaselineDataError(f"{model.model_name} 的 previous_close 与其他基线不一致。")
        predictions[model.model_name] = frame
    return BaselineSuiteResult(predictions=predictions)
