"""阶段 2 行情获取与本地缓存命令行工具。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ConfigError, load_config
from src.data import (
    MarketDataError,
    MarketDataRepository,
    TushareClient,
    generate_quality_report,
    load_market_csv,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="从 Tushare 或本地 CSV 获取标准日线行情并写入安全缓存。"
    )
    parser.add_argument("--cache-dir", type=Path, help="缓存目录；默认读取 configs/default.yaml")
    parser.add_argument(
        "--format",
        choices=("parquet", "csv"),
        default="parquet",
        dest="preferred_format",
        help="首选缓存格式，默认 parquet；不可用时自动回退 CSV。",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--overwrite", action="store_true", help="显式覆盖已有缓存。")
    mode.add_argument("--incremental", action="store_true", help="安全合并并更新已有缓存。")

    subparsers = parser.add_subparsers(dest="source", required=True)
    tushare_parser = subparsers.add_parser("tushare", help="从 Tushare 获取日线行情。")
    tushare_parser.add_argument("--ts-code", required=True)
    tushare_parser.add_argument("--start-date", required=True)
    tushare_parser.add_argument("--end-date", required=True)

    csv_parser = subparsers.add_parser("csv", help="从本地 CSV 导入行情，无需 Token。")
    csv_parser.add_argument("--file", required=True, type=Path)
    csv_parser.add_argument("--ts-code", help="可选；校验 CSV 内股票代码是否一致。")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_config()
        cache_dir = args.cache_dir or config.paths.cache_dir
        repository = MarketDataRepository(
            cache_dir, preferred_format=args.preferred_format
        )

        if args.source == "csv":
            loaded = load_market_csv(args.file, expected_ts_code=args.ts_code)
            frame = loaded.data
            source_report = loaded.quality_report
        else:
            client = TushareClient(config.tushare_token)
            frame = client.fetch_daily(args.ts_code, args.start_date, args.end_date)
            source_report = generate_quality_report(frame)

        if args.incremental:
            result = repository.incremental_update(frame, quality_report=source_report)
        elif args.overwrite:
            result = repository.overwrite(frame, quality_report=source_report)
        else:
            result = repository.save(frame, quality_report=source_report)

        output = {
            "cache": {
                "path": str(result.path),
                "storage_format": result.storage_format,
                "row_count": result.row_count,
                "rows_added": result.rows_added,
                "rows_updated": result.rows_updated,
                "duplicate_rows_removed": result.duplicate_rows_removed,
                "fallback_reason": result.fallback_reason,
            },
            "source_quality": source_report.to_dict(),
            "quality_report_path": str(
                repository.quality_report_path(str(frame["ts_code"].iloc[0]))
            ),
        }
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return 0
    except (MarketDataError, ConfigError) as exc:
        parser.exit(2, f"错误：{exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
