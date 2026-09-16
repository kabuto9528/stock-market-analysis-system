"""本地行情缓存仓库，Parquet 优先、CSV 回退。"""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .errors import CacheError, CacheNotFoundError, MarketDataValidationError
from .quality import DataQualityReport, generate_quality_report
from .schema import STANDARD_MARKET_COLUMNS, canonicalize_market_data, coerce_market_data

_SAFE_CODE = re.compile(r"^[A-Z0-9][A-Z0-9._-]{0,63}$")


def safe_stock_filename(ts_code: str) -> str:
    """把证券代码转换为无目录语义且碰撞概率低的安全文件名。"""

    code = str(ts_code).strip().upper()
    if not code:
        raise CacheError("股票代码不能为空，无法生成缓存文件名。")
    if _SAFE_CODE.fullmatch(code) and ".." not in code:
        return code
    slug = re.sub(r"[^A-Z0-9]+", "_", code).strip("_-")[:48] or "STOCK"
    digest = hashlib.sha256(code.encode("utf-8")).hexdigest()[:12]
    return f"{slug}_{digest}"


@dataclass(frozen=True)
class CacheWriteResult:
    """一次缓存写入的可追溯结果。"""

    path: Path
    storage_format: str
    row_count: int
    rows_added: int
    rows_updated: int
    duplicate_rows_removed: int
    fallback_reason: str | None = None


class MarketDataRepository:
    """以原子替换方式保存、覆盖和增量更新单股票日线缓存。"""

    def __init__(self, root_dir: str | Path, *, preferred_format: str = "parquet") -> None:
        self.root_dir = Path(root_dir).expanduser().resolve()
        self.root_dir.mkdir(parents=True, exist_ok=True)
        if preferred_format not in {"parquet", "csv"}:
            raise CacheError("preferred_format 只能是 parquet 或 csv。")
        self.preferred_format = preferred_format

    def _base(self, ts_code: str) -> str:
        return safe_stock_filename(ts_code)

    def _data_path(self, ts_code: str, storage_format: str) -> Path:
        return self.root_dir / f"{self._base(ts_code)}.{storage_format}"

    def _manifest_path(self, ts_code: str) -> Path:
        return self.root_dir / f"{self._base(ts_code)}.cache.json"

    def quality_report_path(self, ts_code: str) -> Path:
        return self.root_dir / f"{self._base(ts_code)}.quality.json"

    def exists(self, ts_code: str) -> bool:
        return self._manifest_path(ts_code).is_file() or any(
            self._data_path(ts_code, fmt).is_file() for fmt in ("parquet", "csv")
        )

    @staticmethod
    def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
        temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            temp.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            os.replace(temp, path)
        finally:
            if temp.exists():
                temp.unlink()

    @staticmethod
    def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
        temp = path.with_name(f".{path.stem}.{uuid.uuid4().hex}.tmp.parquet")
        try:
            frame.to_parquet(temp, index=False)
            os.replace(temp, path)
        finally:
            if temp.exists():
                temp.unlink()

    @staticmethod
    def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
        temp = path.with_name(f".{path.stem}.{uuid.uuid4().hex}.tmp.csv")
        try:
            frame.to_csv(temp, index=False, encoding="utf-8", date_format="%Y-%m-%d")
            os.replace(temp, path)
        finally:
            if temp.exists():
                temp.unlink()

    def _active_format(self, ts_code: str) -> str | None:
        manifest_path = self._manifest_path(ts_code)
        if manifest_path.is_file():
            try:
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
                storage_format = payload["storage_format"]
            except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
                raise CacheError(f"缓存清单损坏：{manifest_path}；{exc}") from exc
            if storage_format not in {"parquet", "csv"}:
                raise CacheError(f"缓存清单格式字段非法：{storage_format!r}")
            if not self._data_path(ts_code, storage_format).is_file():
                raise CacheError(
                    f"缓存清单指向的数据文件不存在：{self._data_path(ts_code, storage_format)}"
                )
            return storage_format
        for storage_format in (self.preferred_format, "csv", "parquet"):
            if self._data_path(ts_code, storage_format).is_file():
                return storage_format
        return None

    def load(self, ts_code: str) -> pd.DataFrame:
        """读取活动缓存并再次校验标准字段、日期类型和排序唯一性。"""

        storage_format = self._active_format(ts_code)
        if storage_format is None:
            raise CacheNotFoundError(f"未找到股票 {ts_code} 的本地行情缓存。")
        path = self._data_path(ts_code, storage_format)
        try:
            if storage_format == "parquet":
                raw = pd.read_parquet(path)
            else:
                raw = pd.read_csv(path, dtype=object, encoding="utf-8")
            coerced = coerce_market_data(raw, expected_ts_code=ts_code)
        except (OSError, ValueError, ImportError, MarketDataValidationError) as exc:
            raise CacheError(f"缓存读取或校验失败：{path}；{exc}") from exc
        if coerced.duplicated(subset=["ts_code", "trade_date"]).any():
            raise CacheError(f"缓存包含重复交易日期，拒绝静默修复：{path}")
        return coerced.loc[:, STANDARD_MARKET_COLUMNS].reset_index(drop=True)

    def save(
        self,
        frame: pd.DataFrame,
        *,
        overwrite: bool = False,
        quality_report: DataQualityReport | None = None,
        rows_added: int | None = None,
        rows_updated: int = 0,
    ) -> CacheWriteResult:
        """保存标准行情；默认拒绝覆盖，写入成功后再切换活动清单。"""

        try:
            canonical, duplicates_removed = canonicalize_market_data(frame)
        except MarketDataValidationError as exc:
            raise CacheError(f"缓存写入前校验失败：{exc}") from exc
        if canonical.empty:
            raise CacheError("不能保存空行情数据。")
        ts_code = str(canonical["ts_code"].iloc[0])
        if self.exists(ts_code) and not overwrite:
            raise CacheError(
                f"股票 {ts_code} 的缓存已存在；请显式使用覆盖或增量更新。"
            )

        storage_format = self.preferred_format
        fallback_reason: str | None = None
        path = self._data_path(ts_code, storage_format)
        if storage_format == "parquet":
            try:
                self._atomic_parquet(canonical, path)
            except Exception as exc:
                fallback_reason = f"Parquet 不可用，已回退 CSV：{type(exc).__name__}: {exc}"
                storage_format = "csv"
                path = self._data_path(ts_code, storage_format)
                try:
                    self._atomic_csv(canonical, path)
                except Exception as csv_exc:
                    raise CacheError(f"Parquet 与 CSV 缓存写入均失败：{csv_exc}") from csv_exc
        else:
            try:
                self._atomic_csv(canonical, path)
            except Exception as exc:
                raise CacheError(f"CSV 缓存写入失败：{exc}") from exc

        report = quality_report or generate_quality_report(canonical)
        self._atomic_json(self.quality_report_path(ts_code), report.to_dict())
        manifest = {
            "ts_code": ts_code,
            "storage_format": storage_format,
            "file_name": path.name,
            "row_count": len(canonical),
            "start_date": canonical["trade_date"].min().strftime("%Y-%m-%d"),
            "end_date": canonical["trade_date"].max().strftime("%Y-%m-%d"),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "quality_report": self.quality_report_path(ts_code).name,
            "fallback_reason": fallback_reason,
        }
        self._atomic_json(self._manifest_path(ts_code), manifest)
        return CacheWriteResult(
            path=path,
            storage_format=storage_format,
            row_count=len(canonical),
            rows_added=len(canonical) if rows_added is None else rows_added,
            rows_updated=rows_updated,
            duplicate_rows_removed=duplicates_removed,
            fallback_reason=fallback_reason,
        )

    def overwrite(
        self,
        frame: pd.DataFrame,
        *,
        quality_report: DataQualityReport | None = None,
    ) -> CacheWriteResult:
        """显式覆盖同一股票的活动缓存。"""

        return self.save(frame, overwrite=True, quality_report=quality_report)

    def incremental_update(
        self,
        incoming: pd.DataFrame,
        *,
        quality_report: DataQualityReport | None = None,
    ) -> CacheWriteResult:
        """先完整校验增量，再按日期合并；重叠日期由新数据替换并原子落盘。"""

        try:
            normalized, incoming_duplicates = canonicalize_market_data(incoming)
        except MarketDataValidationError as exc:
            raise CacheError(f"增量数据校验失败，原缓存未修改：{exc}") from exc
        if normalized.empty:
            raise CacheError("增量数据为空，原缓存未修改。")
        ts_code = str(normalized["ts_code"].iloc[0])
        if not self.exists(ts_code):
            return self.save(
                normalized,
                quality_report=quality_report,
                rows_added=len(normalized),
            )

        existing = self.load(ts_code)
        existing_dates = set(existing["trade_date"])
        incoming_dates = set(normalized["trade_date"])
        rows_updated = len(existing_dates & incoming_dates)
        rows_added = len(incoming_dates - existing_dates)
        combined = pd.concat([existing, normalized], ignore_index=True)
        merged, merge_duplicates = canonicalize_market_data(combined, expected_ts_code=ts_code)
        final_report = generate_quality_report(merged)
        result = self.save(
            merged,
            overwrite=True,
            quality_report=final_report,
            rows_added=rows_added,
            rows_updated=rows_updated,
        )
        return CacheWriteResult(
            **{
                **asdict(result),
                "duplicate_rows_removed": incoming_duplicates + merge_duplicates,
            }
        )
