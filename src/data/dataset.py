"""无未来信息的滑动窗口、PyTorch Dataset 与 DataLoader 辅助函数。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .errors import MarketDataValidationError
from .scaling import ScalerManager, validate_feature_columns
from .splitting import TemporalSplit


@dataclass(frozen=True)
class WindowedSamples:
    """一个分区的模型输入、缩放目标及可审计日期元数据。"""

    split: str
    X: np.ndarray
    y: np.ndarray
    raw_targets: np.ndarray
    target_dates: np.ndarray
    previous_close: np.ndarray
    input_start_dates: np.ndarray
    input_end_dates: np.ndarray
    feature_columns: tuple[str, ...]
    target_column: str
    window_size: int

    def __post_init__(self) -> None:
        sample_count = self.X.shape[0]
        if self.X.ndim != 3:
            raise MarketDataValidationError("窗口 X 必须是三维数组：(样本数, 窗口长度, 特征数)。")
        if self.X.shape[1:] != (self.window_size, len(self.feature_columns)):
            raise MarketDataValidationError("窗口 X 的长度或特征维度与元数据不一致。")
        if self.y.shape != (sample_count, 1):
            raise MarketDataValidationError("窗口 y 必须是形状 (样本数, 1) 的数组。")
        arrays = (
            self.raw_targets,
            self.target_dates,
            self.previous_close,
            self.input_start_dates,
            self.input_end_dates,
        )
        if any(len(array) != sample_count for array in arrays):
            raise MarketDataValidationError("窗口值与日期元数据的样本数量不一致。")
        if sample_count == 0:
            raise MarketDataValidationError(f"{self.split} 分区没有可用窗口样本。")
        if not np.isfinite(self.X).all() or not np.isfinite(self.y).all():
            raise MarketDataValidationError(f"{self.split} 窗口包含非有限值。")
        if not np.isfinite(self.raw_targets).all() or not np.isfinite(self.previous_close).all():
            raise MarketDataValidationError(f"{self.split} 原始目标或 previous_close 包含非有限值。")
        if not np.all(self.input_end_dates < self.target_dates):
            raise MarketDataValidationError(f"{self.split} 存在输入结束日期不早于目标日期的样本。")

    def __len__(self) -> int:
        return self.X.shape[0]


@dataclass(frozen=True)
class WindowedDataBundle:
    """训练、验证、测试窗口集合。"""

    train: WindowedSamples
    validation: WindowedSamples
    test: WindowedSamples

    def __iter__(self) -> Iterator[WindowedSamples]:
        yield self.train
        yield self.validation
        yield self.test


class StockWindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """只将数值窗口交给模型，日期审计信息保留在 samples 属性中。"""

    def __init__(self, samples: WindowedSamples) -> None:
        self.samples = samples
        self.features = torch.as_tensor(samples.X, dtype=torch.float32)
        self.targets = torch.as_tensor(samples.y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.targets[index]


def _samples_for_indices(
    *,
    split_name: str,
    target_indices: range,
    data: pd.DataFrame,
    scaled_features: np.ndarray,
    scaled_targets: np.ndarray,
    feature_columns: tuple[str, ...],
    target_column: str,
    window_size: int,
) -> WindowedSamples:
    x_values: list[np.ndarray] = []
    y_values: list[np.ndarray] = []
    raw_targets: list[float] = []
    target_dates: list[np.datetime64] = []
    previous_close: list[float] = []
    input_start_dates: list[np.datetime64] = []
    input_end_dates: list[np.datetime64] = []

    dates = data["trade_date"].to_numpy(dtype="datetime64[ns]")
    close = data[target_column].to_numpy(dtype=np.float64)
    for target_index in target_indices:
        input_start = target_index - window_size
        if input_start < 0:
            continue
        input_end = target_index - 1
        if not dates[input_end] < dates[target_index]:
            raise MarketDataValidationError(
                f"{split_name} 样本日期非法：输入结束日期必须严格早于目标日期。"
            )
        x_values.append(scaled_features[input_start:target_index])
        y_values.append(scaled_targets[target_index])
        raw_targets.append(float(close[target_index]))
        target_dates.append(dates[target_index])
        previous_close.append(float(close[input_end]))
        input_start_dates.append(dates[input_start])
        input_end_dates.append(dates[input_end])

    if not x_values:
        raise MarketDataValidationError(
            f"{split_name} 分区样本不足，无法用 window_size={window_size} 构造目标窗口。"
        )
    return WindowedSamples(
        split=split_name,
        X=np.asarray(x_values, dtype=np.float32),
        y=np.asarray(y_values, dtype=np.float32).reshape(-1, 1),
        raw_targets=np.asarray(raw_targets, dtype=np.float64),
        target_dates=np.asarray(target_dates, dtype="datetime64[ns]"),
        previous_close=np.asarray(previous_close, dtype=np.float64),
        input_start_dates=np.asarray(input_start_dates, dtype="datetime64[ns]"),
        input_end_dates=np.asarray(input_end_dates, dtype="datetime64[ns]"),
        feature_columns=feature_columns,
        target_column=target_column,
        window_size=window_size,
    )


def assert_target_dates_disjoint(bundle: WindowedDataBundle) -> None:
    """确认训练、验证、测试目标日期两两互斥。"""

    sets = {
        samples.split: set(samples.target_dates.tolist())
        for samples in bundle
    }
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if sets[left] & sets[right]:
            raise MarketDataValidationError(f"{left} 与 {right} 的目标日期发生重叠。")


def build_windows(
    split: TemporalSplit,
    scaler: ScalerManager,
    feature_columns: Sequence[str],
    *,
    window_size: int = 60,
    target_column: str = "close",
) -> WindowedDataBundle:
    """按目标日期所属分区构造窗口，验证/测试可读取前一区间尾部上下文。"""

    if isinstance(window_size, bool) or not isinstance(window_size, int) or window_size <= 0:
        raise MarketDataValidationError("window_size 必须为正整数。")
    if not isinstance(split, TemporalSplit):
        raise MarketDataValidationError("build_windows 只接受 split_by_time 返回的 TemporalSplit。")
    features = validate_feature_columns(feature_columns)
    if features != scaler.feature_columns:
        raise MarketDataValidationError(
            f"特征顺序变化：Scaler={scaler.feature_columns}，窗口={features}。"
        )
    if target_column != scaler.target_column or target_column != "close":
        raise MarketDataValidationError("窗口目标列必须与 Scaler 一致且为 close。")
    if scaler.training_start_date != split.train.boundary.start_date or (
        scaler.training_end_date != split.train.boundary.end_date
    ):
        raise MarketDataValidationError("Scaler 的训练日期区间与当前训练集不一致。")
    if len(split.train.frame) <= window_size:
        raise MarketDataValidationError(
            f"训练集仅 {len(split.train.frame)} 行，必须大于 window_size={window_size}。"
        )

    data = split.data
    scaled_features = scaler.transform_features(data, feature_columns=features)
    scaled_targets = scaler.transform_target(data[[target_column]])
    validation_start = split.validation.boundary.start_index
    test_start = split.test.boundary.start_index
    total = len(data)

    bundle = WindowedDataBundle(
        train=_samples_for_indices(
            split_name="train",
            target_indices=range(window_size, validation_start),
            data=data,
            scaled_features=scaled_features,
            scaled_targets=scaled_targets,
            feature_columns=features,
            target_column=target_column,
            window_size=window_size,
        ),
        validation=_samples_for_indices(
            split_name="validation",
            target_indices=range(validation_start, test_start),
            data=data,
            scaled_features=scaled_features,
            scaled_targets=scaled_targets,
            feature_columns=features,
            target_column=target_column,
            window_size=window_size,
        ),
        test=_samples_for_indices(
            split_name="test",
            target_indices=range(test_start, total),
            data=data,
            scaled_features=scaled_features,
            scaled_targets=scaled_targets,
            feature_columns=features,
            target_column=target_column,
            window_size=window_size,
        ),
    )
    assert_target_dates_disjoint(bundle)
    return bundle


def create_dataloader(
    samples: WindowedSamples,
    *,
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = 0,
    seed: int = 42,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    """创建可复现 DataLoader；验证集和测试集禁止 shuffle。"""

    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise MarketDataValidationError("batch_size 必须为正整数。")
    if isinstance(num_workers, bool) or not isinstance(num_workers, int) or num_workers < 0:
        raise MarketDataValidationError("num_workers 必须为非负整数。")
    if samples.split != "train" and shuffle:
        raise MarketDataValidationError("验证集和测试集 DataLoader 禁止 shuffle。")
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        StockWindowDataset(samples),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator,
    )


def create_dataloaders(
    bundle: WindowedDataBundle,
    *,
    batch_size: int,
    train_shuffle: bool = False,
    num_workers: int = 0,
    seed: int = 42,
) -> dict[str, DataLoader[tuple[torch.Tensor, torch.Tensor]]]:
    return {
        "train": create_dataloader(
            bundle.train,
            batch_size=batch_size,
            shuffle=train_shuffle,
            num_workers=num_workers,
            seed=seed,
        ),
        "validation": create_dataloader(
            bundle.validation,
            batch_size=batch_size,
            num_workers=num_workers,
            seed=seed,
        ),
        "test": create_dataloader(
            bundle.test,
            batch_size=batch_size,
            num_workers=num_workers,
            seed=seed,
        ),
    }
