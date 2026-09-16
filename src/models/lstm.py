"""使用 batch_first=True 的统一单变量/多变量 LSTM 回归模型。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn

from .errors import ModelConfigurationError


@dataclass(frozen=True)
class LSTMModelConfig:
    """可持久化的 LSTM 网络结构参数。"""

    input_size: int
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2

    def __post_init__(self) -> None:
        for name in ("input_size", "hidden_size", "num_layers"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ModelConfigurationError(f"{name} 必须为正整数。")
        if isinstance(self.dropout, bool) or not isinstance(self.dropout, (int, float)):
            raise ModelConfigurationError("dropout 必须为数值。")
        if not 0 <= float(self.dropout) < 1:
            raise ModelConfigurationError("dropout 必须位于 [0, 1) 区间。")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LSTMRegressor(nn.Module):
    """用输入窗口最后时刻的 LSTM 输出回归下一交易日单个收盘价。"""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.config = LSTMModelConfig(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=float(dropout),
        )
        effective_dropout = self.config.dropout if self.config.num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=self.config.input_size,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            dropout=effective_dropout,
            batch_first=True,
        )
        self.output_layer = nn.Linear(self.config.hidden_size, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """接收 (batch, window, input_size)，返回 (batch, 1)。"""

        if inputs.ndim != 3:
            raise ModelConfigurationError(
                "LSTM 输入必须是三维张量：(batch, window_size, input_size)。"
            )
        if inputs.shape[1] <= 0:
            raise ModelConfigurationError("LSTM 输入窗口不能为空。")
        if inputs.shape[2] != self.config.input_size:
            raise ModelConfigurationError(
                f"输入特征数为 {inputs.shape[2]}，模型要求 {self.config.input_size}。"
            )
        sequence_output, _ = self.lstm(inputs)
        return self.output_layer(sequence_output[:, -1, :])

    def get_config(self) -> dict[str, Any]:
        return self.config.to_dict()
