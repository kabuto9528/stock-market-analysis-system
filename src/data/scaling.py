"""仅训练集拟合的 MinMaxScaler 管理与安全 JSON 持久化。"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
from uuid import uuid4

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from .errors import MarketDataValidationError
from .splitting import TemporalSplit

SCALER_FORMAT_VERSION = 1
UNIVARIATE_FEATURES: tuple[str, ...] = ("close",)
MULTIVARIATE_FEATURES: tuple[str, ...] = (
    "open",
    "high",
    "low",
    "close",
    "vol",
    "amount",
)


def validate_feature_columns(feature_columns: Sequence[str]) -> tuple[str, ...]:
    columns = tuple(str(column) for column in feature_columns)
    if columns not in (UNIVARIATE_FEATURES, MULTIVARIATE_FEATURES):
        raise MarketDataValidationError(
            "特征列必须严格为单变量 ('close',) 或多变量 "
            "('open', 'high', 'low', 'close', 'vol', 'amount')，且顺序不可改变。"
        )
    return columns


def _finite_matrix(frame: pd.DataFrame, columns: tuple[str, ...], label: str) -> np.ndarray:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise MarketDataValidationError(f"{label}缺少字段：{'、'.join(missing)}。")
    values = frame.loc[:, columns].to_numpy(dtype=np.float64, copy=True)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise MarketDataValidationError(f"{label}包含缺失值或正负无穷。")
    return values


def _scaler_state(scaler: MinMaxScaler) -> dict[str, Any]:
    return {
        "feature_range": [float(scaler.feature_range[0]), float(scaler.feature_range[1])],
        "clip": bool(scaler.clip),
        "min": scaler.min_.astype(float).tolist(),
        "scale": scaler.scale_.astype(float).tolist(),
        "data_min": scaler.data_min_.astype(float).tolist(),
        "data_max": scaler.data_max_.astype(float).tolist(),
        "data_range": scaler.data_range_.astype(float).tolist(),
        "n_features_in": int(scaler.n_features_in_),
        "n_samples_seen": int(scaler.n_samples_seen_),
    }


def _restore_scaler(state: object, expected_features: int) -> MinMaxScaler:
    if not isinstance(state, dict):
        raise MarketDataValidationError("Scaler 文件中的缩放参数必须是对象。")
    required = {
        "feature_range",
        "clip",
        "min",
        "scale",
        "data_min",
        "data_max",
        "data_range",
        "n_features_in",
        "n_samples_seen",
    }
    if set(state) != required:
        raise MarketDataValidationError("Scaler 文件的缩放参数字段不完整或包含未知字段。")
    feature_range = state["feature_range"]
    if not isinstance(feature_range, list) or len(feature_range) != 2:
        raise MarketDataValidationError("Scaler feature_range 格式错误。")
    if int(state["n_features_in"]) != expected_features:
        raise MarketDataValidationError("Scaler 参数维度与特征列数量不一致。")

    scaler = MinMaxScaler(
        feature_range=(float(feature_range[0]), float(feature_range[1])),
        clip=bool(state["clip"]),
    )
    arrays: dict[str, np.ndarray] = {}
    for key in ("min", "scale", "data_min", "data_max", "data_range"):
        value = np.asarray(state[key], dtype=np.float64)
        if value.shape != (expected_features,) or not np.isfinite(value).all():
            raise MarketDataValidationError(f"Scaler 参数 {key} 的维度或数值非法。")
        arrays[key] = value
    scaler.min_ = arrays["min"]
    scaler.scale_ = arrays["scale"]
    scaler.data_min_ = arrays["data_min"]
    scaler.data_max_ = arrays["data_max"]
    scaler.data_range_ = arrays["data_range"]
    scaler.n_features_in_ = expected_features
    scaler.n_samples_seen_ = int(state["n_samples_seen"])
    if scaler.n_samples_seen_ <= 0:
        raise MarketDataValidationError("Scaler 训练样本数必须为正数。")
    return scaler


def _canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@dataclass
class ScalerManager:
    """封装特征和目标缩放器，并冻结训练区间及列顺序。"""

    feature_columns: tuple[str, ...]
    target_column: str
    training_start_date: str
    training_end_date: str
    feature_scaler: MinMaxScaler
    target_scaler: MinMaxScaler

    @classmethod
    def fit(
        cls,
        split: TemporalSplit,
        feature_columns: Sequence[str],
        target_column: str = "close",
    ) -> "ScalerManager":
        """只从 TemporalSplit.train 读取数据，接口不接受全量 DataFrame。"""

        if not isinstance(split, TemporalSplit):
            raise MarketDataValidationError(
                "ScalerManager.fit 只接受 split_by_time 返回的 TemporalSplit，禁止传入全量 DataFrame。"
            )
        features = validate_feature_columns(feature_columns)
        if target_column != "close":
            raise MarketDataValidationError("本项目预测目标列必须是 close。")
        train = split.train.frame
        feature_values = _finite_matrix(train, features, "训练集特征")
        target_values = _finite_matrix(train, (target_column,), "训练集目标")
        feature_scaler = MinMaxScaler()
        target_scaler = MinMaxScaler()
        feature_scaler.fit(feature_values)
        target_scaler.fit(target_values)
        return cls(
            feature_columns=features,
            target_column=target_column,
            training_start_date=split.train.boundary.start_date,
            training_end_date=split.train.boundary.end_date,
            feature_scaler=feature_scaler,
            target_scaler=target_scaler,
        )

    def _validate_requested_features(self, feature_columns: Sequence[str] | None) -> None:
        if feature_columns is None:
            return
        requested = tuple(feature_columns)
        if requested != self.feature_columns:
            raise MarketDataValidationError(
                f"特征顺序变化：保存顺序={self.feature_columns}，请求顺序={requested}。"
            )

    def transform_features(
        self,
        frame: pd.DataFrame,
        *,
        feature_columns: Sequence[str] | None = None,
    ) -> np.ndarray:
        self._validate_requested_features(feature_columns)
        encountered = tuple(column for column in frame.columns if column in self.feature_columns)
        if set(encountered) == set(self.feature_columns) and encountered != self.feature_columns:
            raise MarketDataValidationError(
                f"特征顺序变化：保存顺序={self.feature_columns}，输入顺序={encountered}。"
            )
        values = _finite_matrix(frame, self.feature_columns, "待转换特征")
        transformed = self.feature_scaler.transform(values)
        if not np.isfinite(transformed).all():
            raise MarketDataValidationError("特征归一化结果包含非有限值。")
        return transformed

    def transform_target(self, values: pd.DataFrame | pd.Series | np.ndarray) -> np.ndarray:
        if isinstance(values, pd.DataFrame):
            if tuple(values.columns) != (self.target_column,):
                raise MarketDataValidationError(
                    f"目标列必须严格为 ({self.target_column!r},)，当前为 {tuple(values.columns)}。"
                )
            array = values.to_numpy(dtype=np.float64, copy=True)
        elif isinstance(values, pd.Series):
            if values.name not in (None, self.target_column):
                raise MarketDataValidationError(f"目标序列名称必须是 {self.target_column}。")
            array = values.to_numpy(dtype=np.float64, copy=True).reshape(-1, 1)
        else:
            array = np.asarray(values, dtype=np.float64)
            if array.ndim == 1:
                array = array.reshape(-1, 1)
        if array.ndim != 2 or array.shape[1] != 1 or not np.isfinite(array).all():
            raise MarketDataValidationError("目标值必须是有限的一列数值。")
        return self.target_scaler.transform(array)

    def inverse_transform_target(self, values: np.ndarray | Sequence[float]) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        original_shape = array.shape
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        if array.ndim != 2 or array.shape[1] != 1 or not np.isfinite(array).all():
            raise MarketDataValidationError("待反归一化目标必须是有限的一列数值。")
        restored = self.target_scaler.inverse_transform(array)
        return restored.reshape(original_shape) if len(original_shape) == 1 else restored

    def to_payload(self) -> dict[str, Any]:
        return {
            "format_version": SCALER_FORMAT_VERSION,
            "feature_columns": list(self.feature_columns),
            "target_column": self.target_column,
            "training_interval": {
                "start_date": self.training_start_date,
                "end_date": self.training_end_date,
            },
            "feature_scaler": _scaler_state(self.feature_scaler),
            "target_scaler": _scaler_state(self.target_scaler),
        }

    def save(self, path: str | Path) -> Path:
        """保存为带 SHA-256 完整性校验的 JSON，加载时不执行 pickle 代码。"""

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_payload()
        canonical = _canonical_json(payload)
        envelope = {
            "sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
            "payload": payload,
        }
        temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
        try:
            temporary.write_text(
                json.dumps(envelope, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            os.replace(temporary, target)
        finally:
            if temporary.exists():
                temporary.unlink()
        return target

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        expected_feature_columns: Sequence[str] | None = None,
        expected_target_column: str = "close",
    ) -> "ScalerManager":
        source = Path(path)
        try:
            envelope = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise MarketDataValidationError(f"无法读取 Scaler JSON：{source}。") from exc
        if not isinstance(envelope, dict) or set(envelope) != {"sha256", "payload"}:
            raise MarketDataValidationError("Scaler 文件封装格式错误。")
        payload = envelope["payload"]
        if not isinstance(payload, dict):
            raise MarketDataValidationError("Scaler payload 必须是对象。")
        actual_hash = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
        if envelope["sha256"] != actual_hash:
            raise MarketDataValidationError("Scaler 文件完整性校验失败，内容可能已损坏或被修改。")
        required = {
            "format_version",
            "feature_columns",
            "target_column",
            "training_interval",
            "feature_scaler",
            "target_scaler",
        }
        if set(payload) != required or payload["format_version"] != SCALER_FORMAT_VERSION:
            raise MarketDataValidationError("Scaler 文件版本或字段不受支持。")
        features = validate_feature_columns(payload["feature_columns"])
        if expected_feature_columns is not None and tuple(expected_feature_columns) != features:
            raise MarketDataValidationError(
                f"特征顺序变化：文件顺序={features}，期望顺序={tuple(expected_feature_columns)}。"
            )
        target_column = str(payload["target_column"])
        if target_column != expected_target_column or target_column != "close":
            raise MarketDataValidationError("Scaler 目标列与期望不一致。")
        interval = payload["training_interval"]
        if not isinstance(interval, dict) or set(interval) != {"start_date", "end_date"}:
            raise MarketDataValidationError("Scaler 训练区间格式错误。")
        start = str(interval["start_date"])
        end = str(interval["end_date"])
        try:
            if pd.Timestamp(start) > pd.Timestamp(end):
                raise MarketDataValidationError("Scaler 训练区间起始日期晚于结束日期。")
        except ValueError as exc:
            raise MarketDataValidationError("Scaler 训练区间日期无法解析。") from exc
        return cls(
            feature_columns=features,
            target_column=target_column,
            training_start_date=start,
            training_end_date=end,
            feature_scaler=_restore_scaler(payload["feature_scaler"], len(features)),
            target_scaler=_restore_scaler(payload["target_scaler"], 1),
        )
