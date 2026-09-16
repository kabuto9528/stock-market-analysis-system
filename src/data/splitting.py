"""严格按时间顺序划分训练、验证和测试区间。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from .errors import MarketDataValidationError
from .schema import NUMERIC_COLUMNS, STANDARD_MARKET_COLUMNS

_SPLIT_NAMES = ("train", "validation", "test")


@dataclass(frozen=True)
class DateBoundary:
    """一个时间分区的闭区间边界。"""

    name: str
    start_date: str
    end_date: str
    row_count: int
    start_index: int
    end_index: int


@dataclass(frozen=True)
class DataPartition:
    """带受控名称和边界的行情分区。"""

    name: str
    frame: pd.DataFrame
    boundary: DateBoundary


@dataclass(frozen=True)
class TemporalSplit:
    """保留完整升序数据及三个互斥时间分区。"""

    data: pd.DataFrame
    train: DataPartition
    validation: DataPartition
    test: DataPartition
    ratios: tuple[float, float, float]

    @property
    def boundaries(self) -> dict[str, DateBoundary]:
        return {
            "train": self.train.boundary,
            "validation": self.validation.boundary,
            "test": self.test.boundary,
        }

    def boundaries_dict(self) -> dict[str, dict[str, object]]:
        return {name: asdict(boundary) for name, boundary in self.boundaries.items()}


def _normalize_ratios(ratios: object) -> tuple[float, float, float]:
    try:
        if all(hasattr(ratios, name) for name in _SPLIT_NAMES):
            values = tuple(float(getattr(ratios, name)) for name in _SPLIT_NAMES)
        elif isinstance(ratios, Sequence) and not isinstance(ratios, (str, bytes)):
            if len(ratios) != 3:
                raise MarketDataValidationError("划分比例必须依次包含 train、validation、test 三项。")
            values = tuple(float(value) for value in ratios)
        else:
            raise MarketDataValidationError("划分比例必须是三个数值或 SplitRatios 对象。")
    except (TypeError, ValueError) as exc:
        raise MarketDataValidationError("划分比例必须是可转换为浮点数的三个数值。") from exc

    if not all(np.isfinite(values)) or any(value <= 0 or value >= 1 for value in values):
        raise MarketDataValidationError("训练、验证、测试比例都必须是 0 与 1 之间的有限数。")
    if not np.isclose(sum(values), 1.0, rtol=0.0, atol=1e-9):
        raise MarketDataValidationError("训练、验证、测试比例之和必须等于 1。")
    return values  # type: ignore[return-value]


def _validate_prepared_data(data: pd.DataFrame) -> None:
    if not isinstance(data, pd.DataFrame):
        raise MarketDataValidationError("待划分数据必须是 pandas DataFrame。")
    missing = [column for column in STANDARD_MARKET_COLUMNS if column not in data.columns]
    if missing:
        raise MarketDataValidationError("待划分数据缺少字段：" + "、".join(missing) + "。")
    if data.empty:
        raise MarketDataValidationError("待划分数据为空。")
    if data["trade_date"].isna().any():
        raise MarketDataValidationError("trade_date 包含缺失值。")
    if not pd.api.types.is_datetime64_any_dtype(data["trade_date"]):
        raise MarketDataValidationError("trade_date 必须先转换为 datetime64 类型。")
    if not data["trade_date"].is_monotonic_increasing:
        raise MarketDataValidationError("待划分数据必须按 trade_date 严格升序。")
    if data["trade_date"].duplicated().any():
        raise MarketDataValidationError("待划分数据包含重复 trade_date。")
    values = data.loc[:, NUMERIC_COLUMNS].to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(values).all():
        raise MarketDataValidationError("待划分数据包含缺失值或正负无穷。")


def _partition(name: str, data: pd.DataFrame, start: int, end: int) -> DataPartition:
    frame = data.iloc[start:end].copy().reset_index(drop=True)
    boundary = DateBoundary(
        name=name,
        start_date=pd.Timestamp(frame.iloc[0]["trade_date"]).date().isoformat(),
        end_date=pd.Timestamp(frame.iloc[-1]["trade_date"]).date().isoformat(),
        row_count=len(frame),
        start_index=start,
        end_index=end - 1,
    )
    return DataPartition(name=name, frame=frame, boundary=boundary)


def assert_partition_dates_disjoint(split: TemporalSplit) -> None:
    """确认三个原始日期区间没有任何重叠。"""

    date_sets = {
        name: set(getattr(split, name).frame["trade_date"].tolist())
        for name in _SPLIT_NAMES
    }
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = date_sets[left] & date_sets[right]
        if overlap:
            raise MarketDataValidationError(f"{left} 与 {right} 的日期发生重叠。")


def split_by_time(
    data: pd.DataFrame,
    ratios: object = (0.70, 0.15, 0.15),
) -> TemporalSplit:
    """使用稳定的前段/中段/后段切片进行 70/15/15 顺序划分，绝不 shuffle。"""

    _validate_prepared_data(data)
    normalized = _normalize_ratios(ratios)
    total = len(data)
    train_count = int(np.floor(total * normalized[0]))
    validation_count = int(np.floor(total * normalized[1]))
    test_count = total - train_count - validation_count
    counts = (train_count, validation_count, test_count)
    if any(count <= 0 for count in counts):
        raise MarketDataValidationError(
            f"样本不足以按比例划分为三个非空集合：总行数={total}，划分行数={counts}。"
        )

    prepared = data.copy().reset_index(drop=True)
    validation_start = train_count
    test_start = train_count + validation_count
    result = TemporalSplit(
        data=prepared,
        train=_partition("train", prepared, 0, validation_start),
        validation=_partition("validation", prepared, validation_start, test_start),
        test=_partition("test", prepared, test_start, total),
        ratios=normalized,
    )
    assert_partition_dates_disjoint(result)
    return result
