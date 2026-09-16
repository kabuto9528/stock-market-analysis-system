"""LSTM 模型、训练、预测与持久化异常。"""


class LSTMError(Exception):
    """阶段 5 LSTM 模块统一异常基类。"""


class ModelConfigurationError(LSTMError, ValueError):
    """模型结构、训练参数或加载期望不合法。"""


class ModelArtifactError(LSTMError):
    """模型或 Scaler 文件缺失、损坏或不一致。"""


class PredictionError(LSTMError, ValueError):
    """预测输入与已训练模型不一致。"""
