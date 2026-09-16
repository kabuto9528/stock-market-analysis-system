"""评价、日期对齐和实验记录异常。"""


class EvaluationError(Exception):
    """阶段 6 评价模块异常基类。"""


class EvaluationInputError(EvaluationError):
    """指标或预测表输入不符合约束。"""


class BacktestError(EvaluationError):
    """统一测试集回测失败。"""


class ExperimentStoreError(EvaluationError):
    """实验记录保存或读取失败。"""
