"""股票行情获取、清洗、时间划分、缩放、窗口与缓存接口。"""

from .cleaning import (
    CleanMarketDataResult,
    CleaningReport,
    ObservedDateGap,
    clean_market_data,
)
from .csv_loader import CSVLoadResult, load_market_csv
from .dataset import (
    StockWindowDataset,
    WindowedDataBundle,
    WindowedSamples,
    assert_target_dates_disjoint,
    build_windows,
    create_dataloader,
    create_dataloaders,
)
from .errors import (
    CSVImportError,
    CacheError,
    CacheNotFoundError,
    MarketDataError,
    MarketDataValidationError,
    TushareAPIError,
    TushareClientError,
    TushareEmptyDataError,
    TushareNetworkError,
    TushareTokenMissingError,
)
from .quality import DataQualityReport, generate_quality_report
from .repository import CacheWriteResult, MarketDataRepository, safe_stock_filename
from .scaling import (
    MULTIVARIATE_FEATURES,
    UNIVARIATE_FEATURES,
    ScalerManager,
    validate_feature_columns,
)
from .schema import (
    NUMERIC_COLUMNS,
    PRICE_COLUMNS,
    STANDARD_MARKET_COLUMNS,
    canonicalize_market_data,
    coerce_market_data,
)
from .splitting import (
    DataPartition,
    DateBoundary,
    TemporalSplit,
    assert_partition_dates_disjoint,
    split_by_time,
)
from .tushare_client import TushareClient, normalize_tushare_code

__all__ = [
    "CSVImportError",
    "CSVLoadResult",
    "CacheError",
    "CacheNotFoundError",
    "CacheWriteResult",
    "CleanMarketDataResult",
    "CleaningReport",
    "DataPartition",
    "DataQualityReport",
    "DateBoundary",
    "MULTIVARIATE_FEATURES",
    "MarketDataError",
    "MarketDataRepository",
    "MarketDataValidationError",
    "NUMERIC_COLUMNS",
    "ObservedDateGap",
    "PRICE_COLUMNS",
    "STANDARD_MARKET_COLUMNS",
    "ScalerManager",
    "StockWindowDataset",
    "TushareAPIError",
    "TushareClient",
    "TushareClientError",
    "TushareEmptyDataError",
    "TushareNetworkError",
    "TushareTokenMissingError",
    "TemporalSplit",
    "UNIVARIATE_FEATURES",
    "WindowedDataBundle",
    "WindowedSamples",
    "assert_partition_dates_disjoint",
    "assert_target_dates_disjoint",
    "build_windows",
    "canonicalize_market_data",
    "clean_market_data",
    "coerce_market_data",
    "create_dataloader",
    "create_dataloaders",
    "generate_quality_report",
    "load_market_csv",
    "normalize_tushare_code",
    "safe_stock_filename",
    "split_by_time",
    "validate_feature_columns",
]
