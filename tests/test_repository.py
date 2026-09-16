from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data import CacheError, MarketDataRepository, safe_stock_filename


def _frame(rows: list[tuple[str, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ts_code": "600000.SH",
                "trade_date": day,
                "open": close - 0.2,
                "high": close + 0.3,
                "low": close - 0.4,
                "close": close,
                "vol": 1000.0,
                "amount": 10000.0,
            }
            for day, close in rows
        ]
    )


def test_repository_save_load_and_explicit_overwrite(tmp_path: Path) -> None:
    repository = MarketDataRepository(tmp_path)
    original = _frame([("2024-01-02", 10.0), ("2024-01-03", 10.2)])

    first = repository.save(original)
    loaded = repository.load("600000.SH")

    assert first.storage_format in {"parquet", "csv"}
    assert loaded["close"].tolist() == [10.0, 10.2]
    with pytest.raises(CacheError, match="显式使用覆盖"):
        repository.save(original)

    replacement = _frame([("2024-01-04", 11.0)])
    repository.overwrite(replacement)
    assert repository.load("600000.SH")["close"].tolist() == [11.0]


def test_repository_incremental_update_replaces_overlap_and_adds_new_date(tmp_path: Path) -> None:
    repository = MarketDataRepository(tmp_path)
    repository.save(_frame([("2024-01-02", 10.0), ("2024-01-03", 10.2)]))

    result = repository.incremental_update(
        _frame([("2024-01-03", 10.8), ("2024-01-04", 11.0)])
    )
    loaded = repository.load("600000.SH")

    assert result.rows_updated == 1
    assert result.rows_added == 1
    assert loaded["trade_date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2024-01-02",
        "2024-01-03",
        "2024-01-04",
    ]
    assert loaded["close"].tolist() == [10.0, 10.8, 11.0]


def test_repository_falls_back_to_csv_when_parquet_write_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = MarketDataRepository(tmp_path)

    def fail_parquet(frame: pd.DataFrame, path: Path) -> None:
        raise ImportError("pyarrow unavailable")

    monkeypatch.setattr(repository, "_atomic_parquet", fail_parquet)
    result = repository.save(_frame([("2024-01-02", 10.0)]))

    assert result.storage_format == "csv"
    assert result.path.suffix == ".csv"
    assert result.fallback_reason is not None
    assert repository.load("600000.SH")["close"].tolist() == [10.0]


def test_safe_stock_filename_prevents_path_traversal() -> None:
    filename = safe_stock_filename("../../evil/600000.SH")

    assert "/" not in filename
    assert "\\" not in filename
    assert ".." not in filename
    assert filename != "600000.SH"
