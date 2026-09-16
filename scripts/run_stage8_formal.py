from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.stage8 import (  # noqa: E402
    Stage8Error,
    load_stage8_config,
    prepare_available_stocks,
    run_formal_experiments,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="阶段 8 正式实验：先预检，显式 --execute 后运行。")
    parser.add_argument("--config", default="configs/stage8_formal.yaml")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--run-id")
    parser.add_argument("--execute", action="store_true", help="预检通过后执行全部正式实验。")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        config = load_stage8_config(args.config)
        prepared, inventory = prepare_available_stocks(config)
        payload = {
            "config": str(config.path),
            "ready_stock_count": len(prepared),
            "minimum_stock_count": config.minimum_stocks,
            "inventory": inventory.fillna("").to_dict(orient="records"),
            "formal_data_ready": len(prepared) >= config.minimum_stocks,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        if not args.execute:
            return 0 if payload["formal_data_ready"] else 2
        run_root = run_formal_experiments(
            config, run_id=args.run_id, device=args.device
        )
        print(json.dumps({"status": "completed", "run_root": str(run_root)}, ensure_ascii=False))
        return 0
    except Stage8Error as exc:
        print(f"阶段 8 阻塞：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
