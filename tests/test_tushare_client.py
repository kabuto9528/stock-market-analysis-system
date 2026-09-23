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


@pytest.mark.parametrize(
    ("plain_code", "expected_code"),
    [
        ("600519", "600519.SH"),
        ("000001", "000001.SZ"),
        ("300750", "300750.SZ"),
        ("920001", "920001.BJ"),
    ],
)
def test_tushare_client_infers_exchange_for_common_plain_codes(
    plain_code: str, expected_code: str
) -> None:
    frame = _tushare_frame().assign(ts_code=expected_code)
    api = FakeAPI(frame)
    client = TushareClient("token", pro_api_factory=lambda token: api)

    result = client.fetch_daily(plain_code, "2024-01-02", "2024-01-03")

    assert result["ts_code"].unique().tolist() == [expected_code]
    assert api.kwargs is not None and api.kwargs["ts_code"] == expected_code


def test_tushare_client_retries_transient_network_failure() -> None:
    class FlakyAPI:
        def __init__(self) -> None:
            self.calls = 0

        def daily(self, **kwargs: str) -> pd.DataFrame:
            self.calls += 1
            if self.calls < 3:
                raise RequestsConnectionError("temporary offline")
            return _tushare_frame()

    api = FlakyAPI()
    delays: list[float] = []
    client = TushareClient(
        "token",
        pro_api_factory=lambda token: api,
        max_attempts=3,
        retry_delay_seconds=0.25,
        sleep_func=delays.append,
    )

    result = client.fetch_daily("600000", "2024-01-02", "2024-01-03")

    assert len(result) == 2
    assert api.calls == 3
    assert delays == [0.25, 0.5]


def test_tushare_client_rejects_illegal_stock_code_before_network_call() -> None:
    api = FakeAPI(_tushare_frame())
    client = TushareClient("token", pro_api_factory=lambda token: api)

    with pytest.raises(TushareAPIError, match="股票代码格式不合法"):
        client.fetch_daily("ABC", "2024-01-02", "2024-01-03")
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
        max_attempts=2,
        retry_delay_seconds=0,
        sleep_func=lambda seconds: None,
    )

    with pytest.raises(TushareNetworkError, match="已尝试 2 次"):
        client.fetch_daily("600000.SH", "2024-01-02", "2024-01-03")


def test_tushare_client_reports_api_failure() -> None:
    client = TushareClient(
        "token", pro_api_factory=lambda token: FakeAPI(RuntimeError("permission denied"))
    )

    with pytest.raises(TushareAPIError, match="接口调用失败"):
        client.fetch_daily("600000.SH", "2024-01-02", "2024-01-03")
