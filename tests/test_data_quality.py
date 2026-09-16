from __future__ import annotations

import pandas as pd

from src.data import generate_quality_report


def test_quality_report_detects_ohlc_price_and_volume_errors() -> None:
    frame = pd.DataFrame(
        [
            {
                "ts_code": "600000.SH",
                "trade_date": "2024-01-02",
                "open": 10.0,
                "high": 9.0,
                "low": 10.5,
                "close": 11.0,
                "vol": -1,
                "amount": -2,
            },
            {
                "ts_code": "600000.SH",
                "trade_date": "2024-01-03",
                "open": 0.0,
                "high": 10.0,
                "low": -0.1,
                "close": 9.5,
                "vol": 100,
                "amount": 950,
            },
        ]
    )

    report = generate_quality_report(frame)

    assert report.total_rows == 2
    assert report.start_date == "2024-01-02"
    assert report.end_date == "2024-01-03"
    assert report.non_positive_prices["open"] == 1
    assert report.negative_volume == 1
    assert report.negative_amount == 1
    assert report.ohlc_logic_errors["high_below_open"] == 1
    assert report.ohlc_logic_errors["high_below_close"] == 1
    assert report.ohlc_logic_errors["high_below_low"] == 1
    assert report.ohlc_logic_errors["low_above_open"] == 1
    assert report.ohlc_logic_errors["low_above_high"] == 1
    assert report.ohlc_logic_errors["total_rows"] == 1
    assert report.passed is False

