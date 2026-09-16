"""标准行情字段与 DataFrame 规范化。"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from .errors import MarketDataValidationError

STANDARD_MARKET_COLUMNS: tuple[str, ...] = (
    "ts_code",
    "trade_date",
    "open",
    "high",
    "low",
    "close",
    "vol",
    "amount",
)
PRICE_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close")
NUMERIC_COLUMNS: tuple[str, ...] = (*PRICE_COLUMNS, "vol", "amount")


def _row_numbers(mask: pd.Series, *, limit: int = 8) -> str:
    rows = [str(int(index) + 2) for index in mask[mask].index[:limit]]
    suffix = " 等" if int(mask.sum()) > limit else ""
    return f"{', '.join(rows)}{suffix}"


def validate_required_columns(columns: Iterable[object]) -> None:
    """确认标准字段全部存在；允许数据源携带未使用的附加列。"""

    normalized = {str(column).strip() for column in columns}
    missing = [column for column in STANDARD_MARKET_COLUMNS if column not in normalized]
    if missing:
        raise MarketDataValidationError(
            f"缺少标准行情字段：{', '.join(missing)}；必需字段为："
            f"{', '.join(STANDARD_MARKET_COLUMNS)}。"
        )


def coerce_market_data(
    frame: pd.DataFrame,
    *,
    expected_ts_code: str | None = None,
) -> pd.DataFrame:
    """转换为标准列、日期和数值类型，但保留重复日期供质量报告统计。"""

    if not isinstance(frame, pd.DataFrame):
        raise MarketDataValidationError("行情数据必须是 pandas DataFrame。")

    renamed = frame.copy()
    renamed.columns = [str(column).strip() for column in renamed.columns]
    validate_required_columns(renamed.columns)
    result = renamed.loc[:, STANDARD_MARKET_COLUMNS].copy()
    result.reset_index(drop=True, inplace=True)

    result["ts_code"] = result["ts_code"].astype("string").str.strip().str.upper()
    missing_code = result["ts_code"].isna() | result["ts_code"].eq("")
    if missing_code.any():
        raise MarketDataValidationError(
            f"ts_code 不能为空，CSV 行号：{_row_numbers(missing_code)}。"
        )

    codes = result["ts_code"].dropna().unique().tolist()
    if len(codes) != 1:
        raise MarketDataValidationError(
            f"单个行情数据集只能包含一个 ts_code，当前包含：{', '.join(map(str, codes))}。"
        )
    if expected_ts_code and codes[0] != expected_ts_code.strip().upper():
        raise MarketDataValidationError(
            f"行情代码 {codes[0]} 与期望代码 {expected_ts_code.strip().upper()} 不一致。"
        )

    raw_dates = result["trade_date"]
    missing_date = raw_dates.isna() | raw_dates.astype("string").str.strip().eq("")
    parsed_dates = pd.to_datetime(raw_dates, errors="coerce", format="mixed")
    invalid_date = parsed_dates.isna() & ~missing_date
    if missing_date.any() or invalid_date.any():
        bad = missing_date | invalid_date
        examples = raw_dates[bad].astype("string").head(5).tolist()
        raise MarketDataValidationError(
            f"trade_date 存在缺失或无法解析的值，CSV 行号：{_row_numbers(bad)}；"
            f"示例：{examples}。支持 YYYY-MM-DD 或 YYYYMMDD。"
        )
    result["trade_date"] = parsed_dates.dt.normalize()

    for column in NUMERIC_COLUMNS:
        raw = result[column]
        raw_text = raw.astype("string").str.strip()
        missing = raw.isna() | raw_text.eq("")
        converted = pd.to_numeric(raw, errors="coerce")
        invalid = converted.isna() & ~missing
        if invalid.any():
            examples = raw[invalid].astype("string").head(5).tolist()
            raise MarketDataValidationError(
                f"字段 {column} 存在无法转换为数值的内容，CSV 行号："
                f"{_row_numbers(invalid)}；示例：{examples}。"
            )
        result[column] = converted.astype("float64")

    return result.sort_values("trade_date", kind="mergesort").reset_index(drop=True)


def canonicalize_market_data(
    frame: pd.DataFrame,
    *,
    expected_ts_code: str | None = None,
) -> tuple[pd.DataFrame, int]:
    """规范化并按日期升序去重；同日冲突时保留输入中最后一条记录。"""

    result = coerce_market_data(frame, expected_ts_code=expected_ts_code)
    duplicate_count = int(result.duplicated(subset=["ts_code", "trade_date"], keep="last").sum())
    if duplicate_count:
        result = result.drop_duplicates(
            subset=["ts_code", "trade_date"], keep="last"
        ).reset_index(drop=True)
    return result, duplicate_count
