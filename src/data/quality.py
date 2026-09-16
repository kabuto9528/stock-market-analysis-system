"""标准行情数据质量检查。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from .schema import NUMERIC_COLUMNS, PRICE_COLUMNS, STANDARD_MARKET_COLUMNS, validate_required_columns


@dataclass(frozen=True)
class DataQualityReport:
    """可 JSON 序列化的阶段 2 行情质量报告。"""

    total_rows: int
    start_date: str | None
    end_date: str | None
    duplicate_dates: int
    duplicate_date_values: tuple[str, ...]
    missing_values: dict[str, int]
    non_positive_prices: dict[str, int]
    negative_volume: int
    negative_amount: int
    ohlc_logic_errors: dict[str, int]
    passed: bool
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """返回适合写入 JSON 的普通字典。"""

        return asdict(self)


def generate_quality_report(frame: pd.DataFrame) -> DataQualityReport:
    """统计重复、缺失、价格/成交量异常及完整 OHLC 逻辑关系。"""

    validate_required_columns(frame.columns)
    data = frame.loc[:, STANDARD_MARKET_COLUMNS].copy()
    dates = pd.to_datetime(data["trade_date"], errors="coerce")
    valid_dates = dates.dropna()

    duplicate_mask = data.assign(trade_date=dates).duplicated(
        subset=["ts_code", "trade_date"], keep="first"
    )
    duplicate_values = tuple(
        sorted(dates[duplicate_mask & dates.notna()].dt.strftime("%Y-%m-%d").unique().tolist())
    )
    missing_values = {column: int(data[column].isna().sum()) for column in STANDARD_MARKET_COLUMNS}

    numeric = {column: pd.to_numeric(data[column], errors="coerce") for column in NUMERIC_COLUMNS}
    non_positive_prices = {
        column: int(numeric[column].le(0).fillna(False).sum()) for column in PRICE_COLUMNS
    }
    negative_volume = int(numeric["vol"].lt(0).fillna(False).sum())
    negative_amount = int(numeric["amount"].lt(0).fillna(False).sum())

    relation_masks = {
        "high_below_open": numeric["high"] < numeric["open"],
        "high_below_close": numeric["high"] < numeric["close"],
        "high_below_low": numeric["high"] < numeric["low"],
        "low_above_open": numeric["low"] > numeric["open"],
        "low_above_close": numeric["low"] > numeric["close"],
        "low_above_high": numeric["low"] > numeric["high"],
    }
    any_ohlc_error = pd.Series(False, index=data.index)
    ohlc_logic_errors: dict[str, int] = {}
    for name, mask in relation_masks.items():
        clean_mask = mask.fillna(False)
        ohlc_logic_errors[name] = int(clean_mask.sum())
        any_ohlc_error |= clean_mask
    ohlc_logic_errors["total_rows"] = int(any_ohlc_error.sum())

    warnings: list[str] = []
    if duplicate_mask.any():
        warnings.append(f"发现 {int(duplicate_mask.sum())} 条重复日期记录。")
    missing_total = sum(missing_values.values())
    if missing_total:
        warnings.append(f"发现 {missing_total} 个缺失值。")
    non_positive_total = sum(non_positive_prices.values())
    if non_positive_total:
        warnings.append(f"发现 {non_positive_total} 个非正价格。")
    if negative_volume:
        warnings.append(f"发现 {negative_volume} 条负成交量记录。")
    if negative_amount:
        warnings.append(f"发现 {negative_amount} 条负成交额记录。")
    if ohlc_logic_errors["total_rows"]:
        warnings.append(f"发现 {ohlc_logic_errors['total_rows']} 条 OHLC 逻辑异常记录。")

    passed = not any(
        (
            int(duplicate_mask.sum()),
            missing_total,
            non_positive_total,
            negative_volume,
            negative_amount,
            ohlc_logic_errors["total_rows"],
        )
    )
    return DataQualityReport(
        total_rows=len(data),
        start_date=valid_dates.min().strftime("%Y-%m-%d") if not valid_dates.empty else None,
        end_date=valid_dates.max().strftime("%Y-%m-%d") if not valid_dates.empty else None,
        duplicate_dates=int(duplicate_mask.sum()),
        duplicate_date_values=duplicate_values,
        missing_values=missing_values,
        non_positive_prices=non_positive_prices,
        negative_volume=negative_volume,
        negative_amount=negative_amount,
        ohlc_logic_errors=ohlc_logic_errors,
        passed=passed,
        warnings=tuple(warnings),
    )
