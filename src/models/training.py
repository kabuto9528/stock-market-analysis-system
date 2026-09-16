"""仅依赖训练集和验证集的 LSTM 训练与 Early Stopping。"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from .errors import ModelConfigurationError
from .lstm import LSTMRegressor

DeviceChoice = Literal["auto", "cpu", "cuda"] | torch.device


@dataclass(frozen=True)
class EpochRecord:
    """单个 epoch 的训练审计记录。"""

    epoch: int
    train_loss: float
    validation_loss: float
    elapsed_seconds: float


@dataclass(frozen=True)
class TrainingResult:
    """训练结束状态；model 已恢复为验证损失最低的权重。"""

    history: tuple[EpochRecord, ...]
    best_epoch: int
    best_validation_loss: float
    stopped_early: bool
    total_elapsed_seconds: float
    device: str


@dataclass(frozen=True)
class TrainerConfig:
    learning_rate: float = 0.001
    max_epochs: int = 100
    early_stopping_patience: int = 10
    min_delta: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ModelConfigurationError("learning_rate 必须为有限正数。")
        for name in ("max_epochs", "early_stopping_patience"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ModelConfigurationError(f"{name} 必须为正整数。")
        if not math.isfinite(self.min_delta) or self.min_delta < 0:
            raise ModelConfigurationError("min_delta 必须为有限非负数。")


def select_device(choice: DeviceChoice = "auto") -> torch.device:
    """自动选 CUDA，否则回退 CPU；显式请求不可用 CUDA 时给出明确错误。"""

    if isinstance(choice, torch.device):
        device = choice
    elif choice == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif choice in {"cpu", "cuda"}:
        device = torch.device(choice)
    else:
        raise ModelConfigurationError("device 只能是 auto、cpu、cuda 或 torch.device。")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ModelConfigurationError("请求使用 CUDA，但当前环境没有可用 CUDA 设备。")
    return device


def _validate_loader(loader: DataLoader, expected_split: str) -> None:
    if not isinstance(loader, DataLoader):
        raise ModelConfigurationError(f"{expected_split} 必须是 PyTorch DataLoader。")
    if len(loader.dataset) == 0:
        raise ModelConfigurationError(f"{expected_split} DataLoader 不能为空。")
    samples = getattr(loader.dataset, "samples", None)
    if samples is not None and getattr(samples, "split", None) != expected_split:
        raise ModelConfigurationError(
            f"Early Stopping 只允许使用 validation；收到 {getattr(samples, 'split', None)}。"
        )


def _mean_loss(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    optimizer: Adam | None,
) -> float:
    total_loss = 0.0
    total_samples = 0
    training = optimizer is not None
    model.train(training)
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for features, targets in loader:
            features = features.to(device=device, dtype=torch.float32)
            targets = targets.to(device=device, dtype=torch.float32)
            if training:
                optimizer.zero_grad(set_to_none=True)
            predictions = model(features)
            loss = criterion(predictions, targets)
            if not torch.isfinite(loss):
                raise ModelConfigurationError("训练或验证损失出现 NaN/Inf。")
            if training:
                loss.backward()
                optimizer.step()
            batch_size = int(features.shape[0])
            total_loss += float(loss.detach().cpu()) * batch_size
            total_samples += batch_size
    if total_samples == 0:
        raise ModelConfigurationError("DataLoader 未产生任何样本。")
    return total_loss / total_samples


class LSTMTrainer:
    """固定使用 MSELoss 与 Adam，并以验证损失执行 Early Stopping。"""

    def __init__(
        self,
        config: TrainerConfig | None = None,
        *,
        device: DeviceChoice = "auto",
    ) -> None:
        self.config = config or TrainerConfig()
        self.device = select_device(device)

    def fit(
        self,
        model: LSTMRegressor,
        train_loader: DataLoader,
        validation_loader: DataLoader,
    ) -> TrainingResult:
        _validate_loader(train_loader, "train")
        _validate_loader(validation_loader, "validation")
        model.to(self.device)
        criterion = nn.MSELoss()
        optimizer = Adam(model.parameters(), lr=self.config.learning_rate)
        best_state: dict[str, torch.Tensor] | None = None
        best_loss = math.inf
        best_epoch = 0
        epochs_without_improvement = 0
        history: list[EpochRecord] = []
        training_started = time.perf_counter()

        for epoch in range(1, self.config.max_epochs + 1):
            epoch_started = time.perf_counter()
            train_loss = _mean_loss(
                model, train_loader, criterion, self.device, optimizer=optimizer
            )
            validation_loss = _mean_loss(
                model, validation_loader, criterion, self.device, optimizer=None
            )
            history.append(
                EpochRecord(
                    epoch=epoch,
                    train_loss=train_loss,
                    validation_loss=validation_loss,
                    elapsed_seconds=time.perf_counter() - epoch_started,
                )
            )
            if validation_loss < best_loss - self.config.min_delta:
                best_loss = validation_loss
                best_epoch = epoch
                best_state = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in model.state_dict().items()
                }
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.config.early_stopping_patience:
                    break

        if best_state is None or best_epoch <= 0 or not math.isfinite(best_loss):
            raise ModelConfigurationError("训练未产生有效的最佳验证模型。")
        model.load_state_dict(copy.deepcopy(best_state), strict=True)
        model.to(self.device)
        return TrainingResult(
            history=tuple(history),
            best_epoch=best_epoch,
            best_validation_loss=best_loss,
            stopped_early=len(history) < self.config.max_epochs,
            total_elapsed_seconds=time.perf_counter() - training_started,
            device=str(self.device),
        )
