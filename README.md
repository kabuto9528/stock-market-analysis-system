# 基于 LSTM 神经网络的股票价格预测系统

使用截至第 t 个交易日的数据，预测第 t+1 个交易日收盘价。项目已完成阶段 0—9，包含行情导入、清洗、顺序划分、训练集缩放、滑动窗口、传统基线、PyTorch LSTM、统一回测、实验追溯、Streamlit 展示及离线答辩资源。

> 仅用于教学与科研演示，不构成投资建议、收益承诺或交易依据。

## 1. 环境与安装

- Python 3.12
- Windows PowerShell（核心 Python 模块也可在其他平台运行）

```powershell
Set-Location <项目目录>
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

GPU 用户可先按本机 CUDA 环境安装匹配的 PyTorch，再安装 `requirements.txt`。CPU 环境可直接运行测试、离线演示和小型训练。完整说明见 `docs/安装说明.md`。

## 2. 数据与配置

标准字段：`ts_code、trade_date、open、high、low、close、vol、amount`。数据按 `trade_date` 升序并去重，训练/验证/测试按 70%/15%/15% 顺序划分，不使用 shuffle。Scaler 只在训练集拟合。

Tushare Token 只允许写入未提交的 `.env`：

```powershell
Copy-Item .env.example .env
```

```text
TUSHARE_TOKEN=你的Token
```

无 Token 时可使用 CSV、本地缓存或内置离线答辩资源。

## 3. 启动系统

Windows 下可直接双击项目根目录的 `启动系统.bat`。脚本会打开独立的 Edge 应用窗口；关闭该窗口后，本次 Streamlit 后台进程会自动结束。若 8501 已被占用，会自动选择后续可用端口。

也可以手动启动：

```powershell
python -m streamlit run app.py
```

推荐离线答辩路径：

1. “数据获取”选择“离线演示”，加载 `demo/offline_market.csv`。
2. “数据分析”查看 K 线、均线、成交量、收益率和相关性。
3. “预测分析”选择 `offline_demo_stage9`，直接加载预训练模型和已生成预测。
4. “模型对比”展示公共测试日期上的 Naive、MA、SES、ARIMA 和单变量 LSTM。
5. 不点击训练按钮即可完成演示；页面刷新不会自动训练。

首次启动会把 `demo/offline_demo_stage9/` 复制到运行目录 `artifacts/experiments/`（若目标不存在），不会覆盖用户实验。

## 4. 命令行闭环

```powershell
python scripts/run_experiment.py `
  --file demo/offline_market.csv `
  --config tests/fixtures/stage6_test_config.yaml `
  --output-dir artifacts/experiments `
  --experiment-id local_e2e_check `
  --result-kind test `
  --device cpu
```

该命令覆盖 CSV 读取、清洗、划分、缩放、窗口、小型 LSTM 训练、预测、传统基线、指标和实验保存。`test` 结果不得冒充论文正式结论。

## 5. 测试与正式实验

```powershell
python -m pytest -q -p no:cacheprovider --basetemp artifacts/test_tmp
python scripts/check_stage8_consistency.py artifacts/stage8/formal_20260916_1805
```

若系统临时目录权限受限，应把 `--basetemp` 指向项目内可写目录。正式实验使用真实 BaoStock 行情，结果见 `artifacts/stage8/formal_20260916_1805/` 与 `docs/实验结果分析.md`。正式结论显示单变量 LSTM 并未优于 Naive，项目未隐瞒负结果。

## 6. 关键正确性约束

- 验证集只用于调参和 Early Stopping；测试集固定后只做最终评价。
- 保存验证损失最低 epoch 的权重，不保存最后 epoch 冒充最佳模型。
- 输入窗口结束日期严格早于目标日期；目标日真实值不进入特征。
- Naive/MA 先预测再读取已实现值；全部模型只在公共目标日期上比较。
- 主比较只含传统单变量模型与仅 close 的单变量 LSTM；多变量 LSTM 单列为特征消融。
- 排名依据 DA、相对 Naive RMSE 提升率及 RMSE；R² 仅作参考。

## 7. 文档入口

- 安装：`docs/安装说明.md`
- 使用：`docs/系统使用说明.md`
- 测试：`docs/系统测试报告.md`
- 常见问题：`docs/常见问题.md`
- 答辩流程：`docs/答辩演示流程.md`
- 交付清单：`docs/最终交付清单.md`
- 已知问题：`docs/已知问题清单.md`
