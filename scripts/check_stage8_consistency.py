from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.stage8 import Stage8Error, check_stage8_consistency  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="检查阶段 8 表格、实验 ID、预测文件和指标一致性。")
    parser.add_argument("run_dir", help="例如 artifacts/stage8/formal_20260916_180000")
    args = parser.parse_args()
    try:
        report = check_stage8_consistency(PROJECT_ROOT / args.run_dir)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["status"] == "passed" else 2
    except (Stage8Error, OSError, ValueError) as exc:
        print(f"一致性检查失败：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
