"""LSTM 最佳权重与独立 Scaler 的可校验持久化。"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
from uuid import uuid4

import torch

from src.data import ScalerManager, validate_feature_columns

from .errors import ModelArtifactError, ModelConfigurationError
from .lstm import LSTMModelConfig, LSTMRegressor

MODEL_FORMAT_VERSION = 1


@dataclass(frozen=True)
class ModelMetadata:
    feature_columns: tuple[str, ...]
    target_column: str
    window_size: int
    random_seed: int
    best_epoch: int
    best_validation_loss: float
    model_parameters: dict[str, Any]


@dataclass(frozen=True)
class LoadedLSTMArtifact:
    model: LSTMRegressor
    scaler: ScalerManager
    metadata: ModelMetadata


def _positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ModelArtifactError(f"模型文件字段 {field} 必须为正整数。")
    return value


def _validate_metadata(
    payload: object,
    *,
    expected_feature_columns: Sequence[str] | None = None,
    expected_window_size: int | None = None,
    expected_model_parameters: Mapping[str, Any] | None = None,
) -> ModelMetadata:
    if not isinstance(payload, dict):
        raise ModelArtifactError("模型 metadata 必须是对象。")
    required = {
        "feature_columns",
        "target_column",
        "window_size",
        "random_seed",
        "best_epoch",
        "best_validation_loss",
        "model_parameters",
    }
    if set(payload) != required:
        raise ModelArtifactError("模型 metadata 字段不完整或包含未知字段。")
    try:
        features = validate_feature_columns(payload["feature_columns"])
    except Exception as exc:
        raise ModelArtifactError("模型文件中的特征列或顺序非法。") from exc
    target_column = str(payload["target_column"])
    if target_column != "close":
        raise ModelArtifactError("模型目标列必须为 close。")
    window_size = _positive_int(payload["window_size"], "window_size")
    random_seed = payload["random_seed"]
    if isinstance(random_seed, bool) or not isinstance(random_seed, int) or random_seed < 0:
        raise ModelArtifactError("模型文件 random_seed 必须为非负整数。")
    best_epoch = _positive_int(payload["best_epoch"], "best_epoch")
    try:
        best_validation_loss = float(payload["best_validation_loss"])
    except (TypeError, ValueError) as exc:
        raise ModelArtifactError("模型文件 best_validation_loss 必须为数值。") from exc
    if not math.isfinite(best_validation_loss) or best_validation_loss < 0:
        raise ModelArtifactError("模型文件 best_validation_loss 必须为有限非负数。")
    parameters = payload["model_parameters"]
    if not isinstance(parameters, dict):
        raise ModelArtifactError("模型结构参数必须是对象。")
    try:
        config = LSTMModelConfig(**parameters)
    except (TypeError, ModelConfigurationError) as exc:
        raise ModelArtifactError("模型结构参数非法。") from exc
    normalized_parameters = config.to_dict()
    if config.input_size != len(features):
        raise ModelArtifactError("模型 input_size 与特征数量不一致。")
    if expected_feature_columns is not None and tuple(expected_feature_columns) != features:
        raise ModelArtifactError(
            f"特征顺序不一致：模型={features}，期望={tuple(expected_feature_columns)}。"
        )
    if expected_window_size is not None and expected_window_size != window_size:
        raise ModelArtifactError(
            f"窗口长度不一致：模型={window_size}，期望={expected_window_size}。"
        )
    if expected_model_parameters is not None:
        expected = dict(expected_model_parameters)
        if expected != normalized_parameters:
            raise ModelArtifactError(
                f"模型结构参数不一致：模型={normalized_parameters}，期望={expected}。"
            )
    return ModelMetadata(
        feature_columns=features,
        target_column=target_column,
        window_size=window_size,
        random_seed=random_seed,
        best_epoch=best_epoch,
        best_validation_loss=best_validation_loss,
        model_parameters=normalized_parameters,
    )


def save_model_artifact(
    path: str | Path,
    model: LSTMRegressor,
    *,
    feature_columns: Sequence[str],
    target_column: str,
    window_size: int,
    random_seed: int,
    best_epoch: int,
    best_validation_loss: float,
) -> Path:
    """原子保存当前模型；训练器会先把模型恢复到最佳验证 epoch。"""

    features = validate_feature_columns(feature_columns)
    metadata = _validate_metadata(
        {
            "feature_columns": list(features),
            "target_column": target_column,
            "window_size": window_size,
            "random_seed": random_seed,
            "best_epoch": best_epoch,
            "best_validation_loss": best_validation_loss,
            "model_parameters": model.get_config(),
        }
    )
    if metadata.model_parameters["input_size"] != len(features):
        raise ModelArtifactError("模型 input_size 与待保存特征数量不一致。")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
    cpu_state = {
        name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()
    }
    envelope = {
        "format_version": MODEL_FORMAT_VERSION,
        "state_dict": cpu_state,
        "metadata": {
            "feature_columns": list(metadata.feature_columns),
            "target_column": metadata.target_column,
            "window_size": metadata.window_size,
            "random_seed": metadata.random_seed,
            "best_epoch": metadata.best_epoch,
            "best_validation_loss": metadata.best_validation_loss,
            "model_parameters": metadata.model_parameters,
        },
    }
    try:
        torch.save(envelope, temporary)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


def load_model_artifact(
    model_path: str | Path,
    scaler_path: str | Path,
    *,
    expected_feature_columns: Sequence[str] | None = None,
    expected_window_size: int | None = None,
    expected_model_parameters: Mapping[str, Any] | None = None,
    map_location: str | torch.device = "cpu",
) -> LoadedLSTMArtifact:
    """加载模型和独立 Scaler，并交叉校验特征数、顺序、窗口与结构参数。"""

    source = Path(model_path)
    try:
        envelope = torch.load(source, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError, EOFError) as exc:
        raise ModelArtifactError(f"无法安全读取模型文件：{source}。") from exc
    if not isinstance(envelope, dict) or set(envelope) != {
        "format_version",
        "state_dict",
        "metadata",
    }:
        raise ModelArtifactError("模型文件封装格式错误。")
    if envelope["format_version"] != MODEL_FORMAT_VERSION:
        raise ModelArtifactError("模型文件版本不受支持。")
    metadata = _validate_metadata(
        envelope["metadata"],
        expected_feature_columns=expected_feature_columns,
        expected_window_size=expected_window_size,
        expected_model_parameters=expected_model_parameters,
    )
    state_dict = envelope["state_dict"]
    if not isinstance(state_dict, dict) or not state_dict:
        raise ModelArtifactError("模型 state_dict 缺失或为空。")
    try:
        model = LSTMRegressor(**metadata.model_parameters)
        model.load_state_dict(state_dict, strict=True)
        model.to(torch.device(map_location))
        model.eval()
    except (RuntimeError, TypeError, ValueError) as exc:
        raise ModelArtifactError("state_dict 与模型结构参数不匹配。") from exc
    try:
        scaler = ScalerManager.load(
            scaler_path,
            expected_feature_columns=metadata.feature_columns,
            expected_target_column=metadata.target_column,
        )
    except Exception as exc:
        raise ModelArtifactError("Scaler 与模型 metadata 不一致或无法读取。") from exc
    if scaler.feature_scaler.n_features_in_ != metadata.model_parameters["input_size"]:
        raise ModelArtifactError("Scaler 特征数量与模型 input_size 不一致。")
    return LoadedLSTMArtifact(model=model, scaler=scaler, metadata=metadata)
