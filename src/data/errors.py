"""数据模块共享异常类型。"""


class MarketDataError(Exception):
    """行情数据模块可预期错误的基类。"""


class MarketDataValidationError(MarketDataError, ValueError):
    """字段、日期、数值或证券代码不符合标准行情约定。"""


class CSVImportError(MarketDataValidationError):
    """本地 CSV 无法转换为标准行情数据。"""


class CacheError(MarketDataError):
    """行情缓存读写失败或缓存内容不可信。"""


class CacheNotFoundError(CacheError, FileNotFoundError):
    """指定股票的本地行情缓存不存在。"""


class TushareClientError(MarketDataError):
    """Tushare 客户端错误的基类。"""


class TushareTokenMissingError(TushareClientError):
    """环境变量中未配置 Tushare Token。"""


class TushareNetworkError(TushareClientError):
    """访问 Tushare 时发生网络错误。"""


class TushareEmptyDataError(TushareClientError):
    """Tushare 查询成功但未返回行情。"""


class TushareAPIError(TushareClientError):
    """Tushare SDK 或远端接口返回异常。"""
