from __future__ import annotations

import pandas as pd
import pytest
from requests.exceptions import ConnectionError as RequestsConnectionError

from src.data import (
    TushareAPIError,
    TushareClient,
    TushareEmptyDataError,
    TushareNetworkError,
    TushareTokenMissingError,
)


class FakeAPI:
    def __init__(self, result: object) -> None:
        self.result = result
        self.kwargs: dict[str, str] | None = None

    def daily(self, **kwargs: str) -> object:
        self.kwargs = kwargs
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def _tushare_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["600000.SH", "20240103", 10.2, 10.8, 10.0, 10.5, 1200, 12600, 0.3],
            ["600000.SH", "20240102", 10.0, 10.5, 9.8, 10.2, 1000, 10100, 0.2],
        ],
        columns=[
            "ts_code",
            "trade_date",
            "open",
            "high",
            "low",
            "close",
            "vol",
            "amount",
            "pct_chg",
        ],
    )


def test_tushare_client_requires_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TUSHARE_TOKEN", raising=False)

    with pytest.raises(TushareTokenMissingError, match="本地 CSV"):
        TushareClient()


def test_tushare_client_returns_sorted_standard_data() -> None:
    api = FakeAPI(_tushare_frame())
    client = TushareClient("token", pro_api_factory=lambda token: api)

    result = client.fetch_daily("600000.sh", "2024-01-02", "20240103")

    assert result.columns.tolist() == [
        "ts_code",
        "trade_date",
        "open",
        "high",
        "low",
        "close",
        "vol",
        "amount",
    ]
    assert result["trade_date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2024-01-02",
        "2024-01-03",
    ]
    assert api.kwargs == {
        "ts_code": "600000.SH",
        "start_date": "20240102",
        "end_date": "20240103",
    }


def test_tushare_client_rejects_illegal_stock_code_before_network_call() -> None:
    api = FakeAPI(_tushare_frame())
    client = TushareClient("token", pro_api_factory=lambda token: api)

    with pytest.raises(TushareAPIError, match="股票代码格式不合法"):
        client.fetch_daily("600000", "2024-01-02", "2024-01-03")
    assert api.kwargs is None


def test_tushare_client_reports_empty_data() -> None:
    client = TushareClient(
        "token", pro_api_factory=lambda token: FakeAPI(pd.DataFrame())
    )

    with pytest.raises(TushareEmptyDataError, match="未返回数据"):
        client.fetch_daily("600000.SH", "2024-01-02", "2024-01-03")


def test_tushare_client_reports_network_failure() -> None:
    client = TushareClient(
        "token",
        pro_api_factory=lambda token: FakeAPI(RequestsConnectionError("offline")),
    )

    with pytest.raises(TushareNetworkError, match="网络请求失败"):
        client.fetch_daily("600000.SH", "2024-01-02", "2024-01-03")


def test_tushare_client_reports_api_failure() -> None:
    client = TushareClient(
        "token", pro_api_factory=lambda token: FakeAPI(RuntimeError("permission denied"))
    )

    with pytest.raises(TushareAPIError, match="接口调用失败"):
        client.fetch_daily("600000.SH", "2024-01-02", "2024-01-03")
