from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data import CSVImportError, load_market_csv


COLUMNS = ["ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount"]


def _rows() -> list[list[object]]:
    return [
        ["600000.SH", "2024-01-03", 10.2, 10.8, 10.0, 10.5, 1200, 12600],
        ["600000.SH", "20240102", 10.0, 10.5, 9.8, 10.2, 1000, 10100],
    ]


def test_load_normal_csv_converts_types_and_sorts_dates(tmp_path: Path) -> None:
    path = tmp_path / "market.csv"
    pd.DataFrame(_rows(), columns=COLUMNS).to_csv(path, index=False)

    result = load_market_csv(path, expected_ts_code="600000.SH")

    assert result.data.columns.tolist() == COLUMNS
    assert result.data["trade_date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2024-01-02",
        "2024-01-03",
    ]
    assert result.data["close"].dtype == "float64"
    assert result.quality_report.passed is True


def test_load_csv_reports_missing_field(tmp_path: Path) -> None:
    path = tmp_path / "missing.csv"
    pd.DataFrame(_rows(), columns=COLUMNS).drop(columns="amount").to_csv(path, index=False)

    with pytest.raises(CSVImportError, match="缺少标准行情字段：amount"):
        load_market_csv(path)


def test_load_csv_reports_and_removes_duplicate_date(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.csv"
    rows = _rows() + [["600000.SH", "2024-01-03", 10.3, 10.9, 10.1, 10.6, 1300, 13800]]
    pd.DataFrame(rows, columns=COLUMNS).to_csv(path, index=False)

    result = load_market_csv(path)

    assert result.quality_report.duplicate_dates == 1
    assert result.duplicate_rows_removed == 1
    assert len(result.data) == 2
    assert result.data.loc[result.data["trade_date"] == pd.Timestamp("2024-01-03"), "close"].item() == 10.6


def test_load_csv_reports_invalid_numeric_value(tmp_path: Path) -> None:
    path = tmp_path / "invalid.csv"
    rows = _rows()
    rows[0][5] = "not-a-number"
    pd.DataFrame(rows, columns=COLUMNS).to_csv(path, index=False)

    with pytest.raises(CSVImportError, match="字段 close 存在无法转换"):
        load_market_csv(path)
