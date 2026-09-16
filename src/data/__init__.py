"""股票行情获取、本地 CSV 导入、质量检查与缓存接口。"""

from .csv_loader import CSVLoadResult, load_market_csv
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
from .schema import (
    NUMERIC_COLUMNS,
    PRICE_COLUMNS,
    STANDARD_MARKET_COLUMNS,
    canonicalize_market_data,
    coerce_market_data,
)
from .tushare_client import TushareClient

__all__ = [
    "CSVImportError",
    "CSVLoadResult",
    "CacheError",
    "CacheNotFoundError",
    "CacheWriteResult",
    "DataQualityReport",
    "MarketDataError",
    "MarketDataRepository",
    "MarketDataValidationError",
    "NUMERIC_COLUMNS",
    "PRICE_COLUMNS",
    "STANDARD_MARKET_COLUMNS",
    "TushareAPIError",
    "TushareClient",
    "TushareClientError",
    "TushareEmptyDataError",
    "TushareNetworkError",
    "TushareTokenMissingError",
    "canonicalize_market_data",
    "coerce_market_data",
    "generate_quality_report",
    "load_market_csv",
    "safe_stock_filename",
]
