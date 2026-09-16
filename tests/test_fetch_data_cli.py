from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.fetch_data import main


def test_csv_cli_works_without_tushare_token(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.delenv("TUSHARE_TOKEN", raising=False)
    csv_path = tmp_path / "input.csv"
    cache_dir = tmp_path / "cache"
    pd.DataFrame(
        [
            {
                "ts_code": "600000.SH",
                "trade_date": "2024-01-02",
                "open": 10.0,
                "high": 10.5,
                "low": 9.8,
                "close": 10.2,
                "vol": 1000,
                "amount": 10100,
            }
        ]
    ).to_csv(csv_path, index=False)

    exit_code = main(
        [
            "--cache-dir",
            str(cache_dir),
            "--format",
            "csv",
            "csv",
            "--file",
            str(csv_path),
        ]
    )

    assert exit_code == 0
    assert (cache_dir / "600000.SH.csv").is_file()
    assert '"passed": true' in capsys.readouterr().out
