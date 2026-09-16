"""行情清洗：类型统一、稳定排序、去重与仅前向填充。"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .errors import MarketDataValidationError
from .schema import NUMERIC_COLUMNS, canonicalize_market_data


@dataclass(frozen=True)
class ObservedDateGap:
    """相邻两条真实行情记录之间的自然日间隔，仅用于质量提示。"""

    previous_date: str
    current_date: str
    calendar_days: int


@dataclass(frozen=True)
class CleaningReport:
    """可序列化的数据清洗审计记录。"""

    input_rows: int
    output_rows: int
    duplicate_rows_removed: int
    forward_filled_by_column: dict[str, int]
    leading_rows_removed: int
    leading_dates_removed: tuple[str, ...]
    observed_date_gaps: tuple[ObservedDateGap, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CleanMarketDataResult:
    """清洗后的标准行情及审计报告。"""

    data: pd.DataFrame
    report: CleaningReport


def _observed_date_gaps(dates: pd.Series) -> tuple[ObservedDateGap, ...]:
    normalized = pd.DatetimeIndex(dates)
    gaps: list[ObservedDateGap] = []
    for previous, current in zip(normalized[:-1], normalized[1:], strict=False):
        days = int((current - previous).days)
        if days > 1:
            gaps.append(
                ObservedDateGap(
                    previous_date=previous.date().isoformat(),
                    current_date=current.date().isoformat(),
                    calendar_days=days,
                )
            )
    return tuple(gaps)


def _non_finite_locations(frame: pd.DataFrame) -> list[str]:
    values = frame.loc[:, NUMERIC_COLUMNS].to_numpy(dtype=np.float64, copy=False)
    rows, columns = np.where(~np.isfinite(values))
    examples: list[str] = []
    for row, column in zip(rows[:8], columns[:8], strict=False):
        date = pd.Timestamp(frame.iloc[int(row)]["trade_date"]).date().isoformat()
        examples.append(f"{date}:{NUMERIC_COLUMNS[int(column)]}")
    return examples


def clean_market_data(
    frame: pd.DataFrame,
    *,
    expected_ts_code: str | None = None,
) -> CleanMarketDataResult:
    """清洗标准日线行情，不使用任何未来记录填补过去缺失值。

    处理顺序固定为：类型转换与升序排序、同日保留最后一条、数值列
    前向填充、删除前向填充后仍缺失的开头记录、有限值校验。禁止后向填充。
    """

    input_rows = len(frame) if isinstance(frame, pd.DataFrame) else 0
    canonical, duplicate_count = canonicalize_market_data(
        frame, expected_ts_code=expected_ts_code
    )
    if canonical.empty:
        raise MarketDataValidationError("行情数据为空，无法执行清洗。")

    missing_before = canonical.loc[:, NUMERIC_COLUMNS].isna()
    filled = canonical.copy()
    filled.loc[:, NUMERIC_COLUMNS] = filled.loc[:, NUMERIC_COLUMNS].ffill()
    missing_after = filled.loc[:, NUMERIC_COLUMNS].isna()
    forward_counts = {
        column: int(missing_before[column].sum() - missing_after[column].sum())
        for column in NUMERIC_COLUMNS
    }

    leading_mask = missing_after.any(axis=1)
    removed_dates = tuple(
        pd.Timestamp(value).date().isoformat()
        for value in filled.loc[leading_mask, "trade_date"].tolist()
    )
    cleaned = filled.loc[~leading_mask].reset_index(drop=True)
    if cleaned.empty:
        raise MarketDataValidationError(
            "前向填充后所有记录仍含缺失值；没有可用于划分和构造窗口的数据。"
        )

    non_finite = _non_finite_locations(cleaned)
    if non_finite:
        raise MarketDataValidationError(
            "数值字段包含 NaN 或正负无穷，示例位置：" + "、".join(non_finite) + "。"
        )

    if not cleaned["trade_date"].is_monotonic_increasing:
        raise MarketDataValidationError("清洗结果未按 trade_date 升序排列。")
    if cleaned["trade_date"].duplicated().any():
        raise MarketDataValidationError("清洗结果仍包含重复 trade_date。")

    report = CleaningReport(
        input_rows=input_rows,
        output_rows=len(cleaned),
        duplicate_rows_removed=duplicate_count,
        forward_filled_by_column=forward_counts,
        leading_rows_removed=int(leading_mask.sum()),
        leading_dates_removed=removed_dates,
        observed_date_gaps=_observed_date_gaps(cleaned["trade_date"]),
    )
    return CleanMarketDataResult(data=cleaned, report=report)
