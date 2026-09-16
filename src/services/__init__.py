"""业务服务层公共接口。"""

from .backtesting import AlignmentReport, BacktestResult, UnifiedTestBacktestService
from .web import (
    AnalysisWebService, ExperimentDashboard, ExperimentWebService, MarketContext,
    MarketDataWebService, TrainingRequest, TrainingSummary, TrainingWebService,
    WebApplicationService, WebServiceError,
)

__all__ = [
    "AlignmentReport", "BacktestResult", "UnifiedTestBacktestService",
    "AnalysisWebService", "ExperimentDashboard", "ExperimentWebService",
    "MarketContext", "MarketDataWebService", "TrainingRequest", "TrainingSummary",
    "TrainingWebService", "WebApplicationService", "WebServiceError",
]
