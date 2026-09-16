"""Tushare 日线行情客户端。"""

from __future__ import annotations

import os
import re
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


class TushareClient:
    """只负责获取并标准化 Tushare 日线数据，不负责缓存或预处理。"""

    def __init__(
        self,
        token: str | None = None,
        *,
        pro_api_factory: Callable[[str], Any] | None = None,
    ) -> None:
        selected_token = (token if token is not None else os.getenv("TUSHARE_TOKEN", "")).strip()
        if not selected_token:
            raise TushareTokenMissingError(
                "未配置 Tushare Token。请设置环境变量 TUSHARE_TOKEN；"
                "无 Token 时可使用本地 CSV 导入。"
            )
        self._token = selected_token
        self._pro_api_factory = pro_api_factory

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

            return ts.pro_api(self._token)
        except Exception as exc:
            raise TushareAPIError(f"Tushare 客户端初始化失败：{exc}") from exc

    def fetch_daily(
        self,
        ts_code: str,
        start_date: str | date,
        end_date: str | date,
    ) -> pd.DataFrame:
        """按股票代码和闭区间日期获取标准日线行情。"""

        code = ts_code.strip().upper()
        if not _TUSHARE_CODE_PATTERN.fullmatch(code):
            raise TushareAPIError(
                f"股票代码格式不合法：{ts_code!r}；示例：600000.SH、000001.SZ。"
            )
        start = self._parse_date(start_date, "start_date")
        end = self._parse_date(end_date, "end_date")
        if start > end:
            raise TushareAPIError("start_date 不能晚于 end_date。")

        try:
            api = self._create_api()
            frame = api.daily(
                ts_code=code,
                start_date=start.strftime("%Y%m%d"),
                end_date=end.strftime("%Y%m%d"),
            )
        except TushareAPIError:
            raise
        except Exception as exc:
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
            if isinstance(exc, network_errors):
                raise TushareNetworkError(
                    f"Tushare 网络请求失败（{code}，{start.date()} 至 {end.date()}）：{exc}"
                ) from exc
            raise TushareAPIError(
                f"Tushare 接口调用失败（{code}，{start.date()} 至 {end.date()}）：{exc}"
            ) from exc

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
