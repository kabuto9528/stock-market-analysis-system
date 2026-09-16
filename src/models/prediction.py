"""LSTM 批量回测预测和显式目标日期的下一交易日预测。"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import torch

from src.data import ScalerManager, WindowedSamples, validate_feature_columns

from .errors import PredictionError
from .lstm import LSTMRegressor
from .training import DeviceChoice, select_device

PREDICTION_COLUMNS = (
    "target_date",
    "previous_close",
    "actual",
    "predicted",
    "model_name",
)


class LSTMPredictor:
    """绑定模型、Scaler 和训练元数据，拒绝加载后特征或窗口漂移。"""

    def __init__(
        self,
        model: LSTMRegressor,
        scaler: ScalerManager,
        *,
        feature_columns: Sequence[str],
        target_column: str = "close",
        window_size: int,
        device: DeviceChoice = "auto",
        model_name: str = "LSTM",
    ) -> None:
        self.feature_columns = validate_feature_columns(feature_columns)
        self.target_column = str(target_column)
        if self.target_column != "close" or scaler.target_column != self.target_column:
            raise PredictionError("预测目标列必须与 Scaler 一致且为 close。")
        if scaler.feature_columns != self.feature_columns:
            raise PredictionError("预测特征顺序与 Scaler 不一致。")
        if isinstance(window_size, bool) or not isinstance(window_size, int) or window_size <= 0:
            raise PredictionError("window_size 必须为正整数。")
        if model.config.input_size != len(self.feature_columns):
            raise PredictionError("模型 input_size 与预测特征数量不一致。")
        self.window_size = window_size
        self.model = model
        self.scaler = scaler
        self.device = select_device(device)
        self.model_name = str(model_name).strip() or "LSTM"
        self.model.to(self.device)
        self.model.eval()

    def _predict_scaled(self, windows: np.ndarray, batch_size: int) -> np.ndarray:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise PredictionError("batch_size 必须为正整数。")
        values = np.asarray(windows, dtype=np.float32)
        expected_shape = (self.window_size, len(self.feature_columns))
        if values.ndim != 3 or values.shape[1:] != expected_shape:
            raise PredictionError(
                f"预测窗口必须为 (样本数, {expected_shape[0]}, {expected_shape[1]})。"
            )
        if len(values) == 0 or not np.isfinite(values).all():
            raise PredictionError("预测窗口不能为空且必须全部为有限值。")
        outputs: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(values), batch_size):
                batch = torch.as_tensor(
                    values[start : start + batch_size],
                    dtype=torch.float32,
                    device=self.device,
                )
                outputs.append(self.model(batch).detach().cpu().numpy())
        return np.concatenate(outputs, axis=0)

    def predict_samples(
        self,
        samples: WindowedSamples,
        *,
        batch_size: int = 256,
    ) -> pd.DataFrame:
        """批量预测窗口样本，反归一化 close，并保留目标日期和真实价格。"""

        if samples.feature_columns != self.feature_columns:
            raise PredictionError("窗口特征列或顺序与模型不一致。")
        if samples.target_column != self.target_column:
            raise PredictionError("窗口目标列与模型不一致。")
        if samples.window_size != self.window_size:
            raise PredictionError("窗口长度与模型训练 metadata 不一致。")
        scaled = self._predict_scaled(samples.X, batch_size)
        predicted = self.scaler.inverse_transform_target(scaled).reshape(-1)
        return pd.DataFrame(
            {
                "target_date": pd.to_datetime(samples.target_dates),
                "previous_close": samples.previous_close.astype(float),
                "actual": samples.raw_targets.astype(float),
                "predicted": predicted.astype(float),
                "model_name": self.model_name,
            },
            columns=PREDICTION_COLUMNS,
        )

    def predict_next_trading_day(
        self,
        data: pd.DataFrame,
        *,
        target_date: str | pd.Timestamp,
    ) -> pd.DataFrame:
        """用最近 window_size 条记录预测调用方给出的下一交易日；actual 保持 NaN。"""

        missing = [
            column
            for column in ("trade_date", self.target_column, *self.feature_columns)
            if column not in data.columns
        ]
        if missing:
            raise PredictionError("下一交易日预测缺少字段：" + "、".join(dict.fromkeys(missing)) + "。")
        if len(data) < self.window_size:
            raise PredictionError(
                f"行情仅 {len(data)} 行，不足 window_size={self.window_size}。"
            )
        dates = pd.to_datetime(data["trade_date"], errors="coerce")
        if dates.isna().any() or not dates.is_monotonic_increasing or dates.duplicated().any():
            raise PredictionError("行情日期必须可解析、严格升序且不重复。")
        parsed_target = pd.Timestamp(target_date)
        if pd.isna(parsed_target) or parsed_target <= dates.iloc[-1]:
            raise PredictionError("target_date 必须严格晚于最后一条已知行情日期。")
        recent = data.iloc[-self.window_size :]
        scaled_features = self.scaler.transform_features(
            recent, feature_columns=self.feature_columns
        )
        scaled_prediction = self._predict_scaled(scaled_features[None, :, :], 1)
        predicted = float(self.scaler.inverse_transform_target(scaled_prediction).reshape(-1)[0])
        return pd.DataFrame(
            {
                "target_date": [parsed_target],
                "previous_close": [float(recent.iloc[-1][self.target_column])],
                "actual": [np.nan],
                "predicted": [predicted],
                "model_name": [self.model_name],
            },
            columns=PREDICTION_COLUMNS,
        )
