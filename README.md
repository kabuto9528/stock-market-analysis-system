# 基于 LSTM 神经网络的股票价格预测系统

本项目使用截至第 t 个交易日的数据预测第 t+1 个交易日收盘价。当前完成阶段 2：可从 Tushare 或本地 CSV 获取标准日线行情，生成质量报告，并写入 Parquet 优先、CSV 回退的本地缓存。

> 本系统仅用于教学与科研演示，不构成任何投资建议。

## 运行环境

- Python 3.12
- Windows PowerShell（以下命令以 PowerShell 为例）

## 创建并激活虚拟环境

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

如果 PowerShell 阻止激活脚本，可在当前终端执行：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## 安装依赖

先单独安装匹配本机驱动的 GPU 版 PyTorch，再安装其余依赖：

```powershell
python -m pip install torch --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

`pyarrow` 是 Parquet 可选依赖；未安装或 Parquet 写入不受支持时，行情仓库会自动回退为 UTF-8 CSV，不影响离线流程。

## Tushare Token

复制示例文件并填写 Token，禁止提交 `.env`：

```powershell
Copy-Item .env.example .env
```

```text
TUSHARE_TOKEN=你的Token
```

Token 仅从环境变量或 `.env` 读取。**没有 Token 时仍可通过本地 CSV 完成后续流程**，CSV 导入和缓存读取不会初始化 Tushare 客户端。

## 标准行情字段

CSV 至少包含以下字段；允许携带额外列，但缓存只保留标准列：

```text
ts_code,trade_date,open,high,low,close,vol,amount
```

- `trade_date` 支持 `YYYY-MM-DD` 和 `YYYYMMDD`，导入后统一升序。
- 单个文件只能包含一个 `ts_code`。
- 重复交易日期会在质量报告中记录，缓存前保留输入中最后一条记录。
- 非正价格、负成交量/成交额和 OHLC 逻辑异常会写入质量报告，不会被伪造或静默修正。

## 获取或导入行情

从本地 CSV 导入（无需 Token）：

```powershell
python scripts/fetch_data.py --cache-dir data/cache csv --file data/example.csv --ts-code 600000.SH
```

从 Tushare 获取：

```powershell
python scripts/fetch_data.py tushare --ts-code 600000.SH --start-date 2020-01-01 --end-date 2025-12-31
```

已有缓存默认拒绝覆盖。需要显式选择覆盖或增量更新：

```powershell
python scripts/fetch_data.py --overwrite csv --file data/example.csv
python scripts/fetch_data.py --incremental tushare --ts-code 600000.SH --start-date 2026-01-01 --end-date 2026-09-16
```

命令输出缓存格式、行数、增量新增/替换数量和数据质量报告。缓存目录同时保存 `*.cache.json` 活动清单和 `*.quality.json` 质量报告。证券代码会转换为安全文件名，不能借助 `..`、斜杠或反斜杠逃逸缓存目录。

## 缓存策略

- 默认优先写 Parquet；缺少引擎或格式写入失败时回退 CSV。
- 保存、覆盖和增量更新均先完成字段与类型校验，再以临时文件原子替换。
- 增量更新按 `ts_code + trade_date` 合并；重叠日期由新数据替换，新日期按升序写入。
- 活动格式由缓存清单记录，不依赖同时存在的旧格式副本。

## 配置文件

默认配置位于 `configs/default.yaml`，包括 70%/15%/15% 时间顺序划分比例、60 个交易日窗口、LSTM 默认超参数、随机种子和项目相对存储目录。运行目录由 `src/config.py` 解析，配置中不得写用户绝对路径或固定盘符。

## 数据预处理

阶段 3 数据层按以下顺序调用：

```python
cleaned = clean_market_data(raw_frame)
split = split_by_time(cleaned.data, (0.70, 0.15, 0.15))
scaler = ScalerManager.fit(split, ("close",))
windows = build_windows(split, scaler, ("close",), window_size=60)
```

Scaler 只能从 `TemporalSplit.train` 拟合；验证集与测试集只做转换。多变量特征顺序固定为
`open, high, low, close, vol, amount`。详细规则和跨边界日期示例见 `docs/数据处理说明.md`。

## 运行测试

测试不访问真实网络，Tushare 使用 mock：

```powershell
python -m pytest tests/test_preprocessing.py -q
python -m pytest -q
```

## 启动 Streamlit

```powershell
python -m streamlit run app.py
```

页面刷新不会自动获取数据或启动训练。阶段 3 已完成数据清洗、时间划分、缩放和窗口构造，尚未进入基线模型或 LSTM 训练。

## 目录说明

- `src/data/`：标准字段、CSV、Tushare、质量报告、缓存、清洗、时间划分、Scaler 与窗口数据集
- `scripts/fetch_data.py`：阶段 2 命令行入口
- `data/cache/`：本地行情缓存（不入版本库）
- `src/models/`、`src/baselines/`、`src/evaluation/`：后续阶段实现
- `configs/`：YAML 配置
- `tests/`：不访问真实网络的自动测试
- `artifacts/`：模型与实验产物（不入版本库）
