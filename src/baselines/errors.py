"""传统基线模型的可预期异常。"""


class BaselineError(Exception):
    """基线模型错误基类。"""


class BaselineDataError(BaselineError, ValueError):
    """输入历史、日期、目标日期或回测协议不合法。"""


class BaselineNotFittedError(BaselineError, RuntimeError):
    """模型尚未使用训练分区拟合。"""


class BaselineFitError(BaselineError, RuntimeError):
    """统计模型拟合失败。"""


class BaselinePredictionError(BaselineError, RuntimeError):
    """统计模型预测或状态更新失败。"""
