"""Tushare 日线行情客户端。"""

from __future__ import annotations

import os
import re
import time
from collections.abc import Callable
from datetime import date
from typing import Any

import pandas as pd

from .errors import (
    MarketDataValidationError,
    TushareAPIError,
    TushareEmptyDataError,
    TushareNetworkError,
    TushareTokenMissingError,
)
from .schema import canonicalize_market_data

_TUSHARE_CODE_PATTERN = re.compile(r"^\d{6}\.(SH|SZ|BJ)$", re.IGNORECASE)
_PLAIN_CODE_PATTERN = re.compile(r"^\d{6}$")


def normalize_tushare_code(ts_code: str) -> str:
    """标准化股票代码；常见 A 股纯 6 位代码可自动推断交易所。"""

    code = str(ts_code).strip().upper()
    if _TUSHARE_CODE_PATTERN.fullmatch(code):
        return code
    if _PLAIN_CODE_PATTERN.fullmatch(code):
        if code.startswith(("60", "68")):
            return f"{code}.SH"
        if code.startswith(("00", "30")):
            return f"{code}.SZ"
        if code.startswith(("43", "83", "87", "92")):
            return f"{code}.BJ"
        raise TushareAPIError(
            f"无法自动判断股票代码 {ts_code!r} 的交易所；请显式填写 .SH、.SZ 或 .BJ。"
        )
    raise TushareAPIError(
        f"股票代码格式不合法：{ts_code!r}；示例：600000、600000.SH、000001.SZ。"
    )


class TushareClient:
    """只负责获取并标准化 Tushare 日线数据，不负责缓存或预处理。"""

    def __init__(
        self,
        token: str | None = None,
        *,
        pro_api_factory: Callable[[str], Any] | None = None,
        request_timeout_seconds: float = 15.0,
        max_attempts: int = 3,
        retry_delay_seconds: float = 0.5,
        sleep_func: Callable[[float], None] = time.sleep,
    ) -> None:
        selected_token = (token if token is not None else os.getenv("TUSHARE_TOKEN", "")).strip()
        if not selected_token:
            raise TushareTokenMissingError(
                "未配置 Tushare Token。请设置环境变量 TUSHARE_TOKEN；"
                "无 Token 时可使用本地 CSV 导入。"
            )
        self._token = selected_token
        self._pro_api_factory = pro_api_factory
        if request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds 必须大于 0。")
        if max_attempts <= 0:
            raise ValueError("max_attempts 必须为正整数。")
        if retry_delay_seconds < 0:
            raise ValueError("retry_delay_seconds 不能小于 0。")
        self._request_timeout_seconds = float(request_timeout_seconds)
        self._max_attempts = int(max_attempts)
        self._retry_delay_seconds = float(retry_delay_seconds)
        self._sleep_func = sleep_func

    @staticmethod
    def _parse_date(value: str | date, field_name: str) -> pd.Timestamp:
        try:
            parsed = pd.Timestamp(value).normalize()
        except (TypeError, ValueError) as exc:
            raise TushareAPIError(
                f"{field_name} 无法解析：{value!r}；请使用 YYYY-MM-DD 或 YYYYMMDD。"
            ) from exc
        if pd.isna(parsed):
            raise TushareAPIError(f"{field_name} 不能为空。")
        return parsed

    def _create_api(self) -> Any:
        if self._pro_api_factory is not None:
            return self._pro_api_factory(self._token)
        try:
            import tushare as ts

            return ts.pro_api(self._token, timeout=self._request_timeout_seconds)
        except Exception as exc:
            raise TushareAPIError(f"Tushare 客户端初始化失败：{exc}") from exc

    def fetch_daily(
        self,
        ts_code: str,
        start_date: str | date,
        end_date: str | date,
    ) -> pd.DataFrame:
        """按股票代码和闭区间日期获取标准日线行情。"""

        code = normalize_tushare_code(ts_code)
        start = self._parse_date(start_date, "start_date")
        end = self._parse_date(end_date, "end_date")
        if start > end:
            raise TushareAPIError("start_date 不能晚于 end_date。")

        api = self._create_api()
        network_errors: tuple[type[BaseException], ...] = (
            ConnectionError,
            TimeoutError,
            OSError,
        )
        try:
            from requests.exceptions import RequestException

            network_errors += (RequestException,)
        except ImportError:  # pragma: no cover - tushare 正常安装时 requests 必然存在
            pass

        frame: Any = None
        for attempt in range(1, self._max_attempts + 1):
            try:
                frame = api.daily(
                    ts_code=code,
                    start_date=start.strftime("%Y%m%d"),
                    end_date=end.strftime("%Y%m%d"),
                )
                break
            except Exception as exc:
                if not isinstance(exc, network_errors):
                    raise TushareAPIError(
                        f"Tushare 接口调用失败（{code}，{start.date()} 至 {end.date()}）：{exc}"
                    ) from exc
                if attempt >= self._max_attempts:
                    raise TushareNetworkError(
                        f"Tushare 网络请求失败（{code}，{start.date()} 至 {end.date()}，"
                        f"已尝试 {self._max_attempts} 次）：{exc}"
                    ) from exc
                self._sleep_func(self._retry_delay_seconds * attempt)

        if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
            raise TushareEmptyDataError(
                f"Tushare 未返回数据：{code}，{start.date()} 至 {end.date()}。"
                "请检查代码、日期范围、权限或是否为交易日。"
            )
        try:
            normalized, _ = canonicalize_market_data(frame, expected_ts_code=code)
        except MarketDataValidationError as exc:
            raise TushareAPIError(f"Tushare 返回数据格式异常：{exc}") from exc
        return normalized
