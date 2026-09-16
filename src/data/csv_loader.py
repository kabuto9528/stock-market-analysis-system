"""本地 CSV 行情导入。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .errors import CSVImportError, MarketDataValidationError
from .quality import DataQualityReport, generate_quality_report
from .schema import canonicalize_market_data, coerce_market_data


@dataclass(frozen=True)
class CSVLoadResult:
    """CSV 导入后的标准数据和去重前质量报告。"""

    data: pd.DataFrame
    quality_report: DataQualityReport
    duplicate_rows_removed: int


def load_market_csv(
    path: str | Path,
    *,
    expected_ts_code: str | None = None,
    encoding: str = "utf-8-sig",
) -> CSVLoadResult:
    """读取本地 CSV，校验字段并转换、排序、去除重复日期。"""

    csv_path = Path(path).expanduser()
    if not csv_path.is_file():
        raise CSVImportError(f"CSV 文件不存在：{csv_path}")
    try:
        raw = pd.read_csv(csv_path, encoding=encoding, dtype=object)
    except UnicodeDecodeError as exc:
        raise CSVImportError(f"CSV 编码解析失败：{csv_path}；默认要求 UTF-8。") from exc
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise CSVImportError(f"CSV 读取失败：{csv_path}；{exc}") from exc

    try:
        coerced = coerce_market_data(raw, expected_ts_code=expected_ts_code)
        report = generate_quality_report(coerced)
        data, duplicate_count = canonicalize_market_data(
            coerced, expected_ts_code=expected_ts_code
        )
    except MarketDataValidationError as exc:
        raise CSVImportError(f"CSV 校验失败：{exc}") from exc

    return CSVLoadResult(
        data=data,
        quality_report=report,
        duplicate_rows_removed=duplicate_count,
    )
