from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import ProxyHandler, Request, build_opener

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.cleaning import clean_market_data  # noqa: E402
from src.data.errors import MarketDataError  # noqa: E402
from src.data.tushare_client import TushareClient  # noqa: E402
from src.evaluation.stage8 import Stage8Error, load_stage8_config  # noqa: E402

EASTMONEY_ENDPOINT = "https://push2his.eastmoney.com/api/qt/stock/kline/get"


def _fetch_eastmoney(ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    market = "1" if ts_code.endswith(".SH") else "0"
    symbol = ts_code.split(".")[0]
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    rows: list[dict[str, object]] = []
    opener = build_opener(ProxyHandler({}))
    for year in range(start.year, end.year + 1):
        chunk_start = max(start, pd.Timestamp(year=year, month=1, day=1))
        chunk_end = min(end, pd.Timestamp(year=year, month=12, day=31))
        params = {
            "secid": f"{market}.{symbol}",
            "fields1": "f1,f2,f3,f4,f5,f6",
            "fields2": "f51,f52,f53,f54,f55,f56,f57",
            "klt": "101",
            "fqt": "0",
            "beg": chunk_start.strftime("%Y%m%d"),
            "end": chunk_end.strftime("%Y%m%d"),
            "lmt": "1000",
        }
        request = Request(
            f"{EASTMONEY_ENDPOINT}?{urlencode(params)}",
            headers={"User-Agent": "Mozilla/5.0", "Referer": "https://quote.eastmoney.com/"},
        )
        last_error: Exception | None = None
        payload = None
        for attempt in range(1, 6):
            try:
                with opener.open(request, timeout=30) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                break
            except (OSError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt < 5:
                    time.sleep(float(attempt))
        if payload is None:
            raise Stage8Error(f"获取 {ts_code} {year} 年行情失败：{last_error}")
        data = payload.get("data")
        if not data or not data.get("klines"):
            continue
        for item in data["klines"]:
            trade_date, open_, close, high, low, vol, amount_yuan = item.split(",")[:7]
            rows.append(
                {
                    "ts_code": ts_code,
                    "trade_date": trade_date,
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "vol": vol,
                    "amount": float(amount_yuan) / 1000.0,
                }
            )
    if not rows:
        raise Stage8Error(f"公开行情接口未返回 {ts_code} 的日线数据。")
    return pd.DataFrame(rows).drop_duplicates(subset=["trade_date"], keep="last")


def _fetch_baostock(client: object, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    prefix = "sh" if ts_code.endswith(".SH") else "sz"
    symbol = f"{prefix}.{ts_code.split('.')[0]}"
    fields = "date,code,open,high,low,close,volume,amount,tradestatus"
    result = client.query_history_k_data_plus(
        symbol,
        fields,
        start_date=start_date,
        end_date=end_date,
        frequency="d",
        adjustflag="3",
    )
    if result.error_code != "0":
        raise Stage8Error(f"BaoStock 获取 {ts_code} 失败：{result.error_msg}")
    rows = []
    while result.next():
        values = dict(zip(result.fields, result.get_row_data(), strict=True))
        if values["tradestatus"] != "1" or not values["close"]:
            continue
        rows.append(
            {
                "ts_code": ts_code,
                "trade_date": values["date"],
                "open": values["open"],
                "high": values["high"],
                "low": values["low"],
                "close": values["close"],
                "vol": float(values["volume"]) / 100.0,
                "amount": float(values["amount"]) / 1000.0,
            }
        )
    if not rows:
        raise Stage8Error(f"BaoStock 未返回 {ts_code} 的有效交易日数据。")
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="按阶段 8 配置获取真实日线 CSV。")
    parser.add_argument("--config", default="configs/stage8_formal.yaml")
    parser.add_argument(
        "--source", choices=("tushare", "baostock", "eastmoney"), default="tushare"
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    baostock_client = None
    try:
        config = load_stage8_config(args.config)
        tushare_client = TushareClient() if args.source == "tushare" else None
        if args.source == "baostock":
            baostock_client = importlib.import_module("baostock")
            login = baostock_client.login()
            if login.error_code != "0":
                raise Stage8Error(f"BaoStock 登录失败：{login.error_msg}")
        for stock in config.stocks:
            if stock.file.exists() and not args.overwrite:
                print(f"跳过已存在文件：{stock.file}")
                continue
            start_date = config.start_date.date().isoformat()
            end_date = config.end_date.date().isoformat()
            if args.source == "tushare":
                assert tushare_client is not None
                frame = tushare_client.fetch_daily(stock.ts_code, start_date, end_date)
                source_name = "Tushare pro.daily"
                endpoint = "Tushare pro.daily"
            elif args.source == "baostock":
                assert baostock_client is not None
                frame = _fetch_baostock(baostock_client, stock.ts_code, start_date, end_date)
                source_name = "BaoStock historical K data"
                endpoint = "query_history_k_data_plus"
            else:
                frame = _fetch_eastmoney(stock.ts_code, start_date, end_date)
                source_name = "Eastmoney public historical K-line API"
                endpoint = EASTMONEY_ENDPOINT
            cleaned = clean_market_data(frame, expected_ts_code=stock.ts_code)
            stock.file.parent.mkdir(parents=True, exist_ok=True)
            cleaned.data.to_csv(stock.file, index=False, encoding="utf-8-sig")
            provenance = {
                "ts_code": stock.ts_code,
                "source": source_name,
                "source_endpoint": endpoint,
                "adjustment": "none",
                "requested_start_date": start_date,
                "requested_end_date": end_date,
                "actual_start_date": cleaned.data.iloc[0]["trade_date"].date().isoformat(),
                "actual_end_date": cleaned.data.iloc[-1]["trade_date"].date().isoformat(),
                "sample_count": len(cleaned.data),
                "volume_unit": "手（BaoStock/东方财富原始量按来源转换）",
                "amount_unit": "千元（原始成交额为元，保存前除以 1000）",
                "fetched_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            }
            stock.file.with_suffix(".source.json").write_text(
                json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(
                f"已写入 {stock.ts_code}: {len(cleaned.data)} 行，"
                f"{cleaned.data.iloc[0]['trade_date'].date()} 至 "
                f"{cleaned.data.iloc[-1]['trade_date'].date()} -> {stock.file}"
            )
        return 0
    except (Stage8Error, MarketDataError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"正式数据准备失败：{exc}", file=sys.stderr)
        return 2
    finally:
        if baostock_client is not None:
            baostock_client.logout()


if __name__ == "__main__":
    raise SystemExit(main())
