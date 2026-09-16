# 阶段 7：Streamlit 可视化系统

- 日期：2026-09-16
- 状态：验收通过；未进入阶段 8
- 执行方式：单 Agent 串行，未创建或委派子 Agent

## 完成内容

1. 完成首页、数据获取、数据分析、模型训练、预测分析和模型对比六个页面。
2. 页面只通过 `src/services/web.py` 调用数据、训练、预测、评价和实验服务，未复制核心算法。
3. 支持 Tushare、CSV、本地缓存和固定小型离线样例；显示数据预览与质量报告。
4. 实现 Plotly K 线、收盘价与 MA5/10/20、成交量、收益率分布和相关性热力图。
5. 模型训练支持单/多变量及全部要求参数；只有表单按钮提交才训练，刷新不自动训练。
6. 训练显示 Epoch 进度、训练/验证损失、最优 Epoch、最低验证损失、耗时和设备。
7. 训练产物保存最佳验证模型、Scaler、历史、指标、逐日预测和下一交易日预测。
8. 预测页提供真实/预测曲线、误差图、指标卡和预训练模型加载。
9. 对比页提供 RMSE、MAE、DA、相对提升率图与 CSV 下载；兼容实验可合并单/多变量 LSTM。
10. 使用 session state 保存行情与最近实验；service 使用 cache_resource，分析和实验读取使用 cache_data。
11. Token、数据不足、模型缺失、配置不匹配和训练失败均转换为中文友好错误。
12. 更新 README、系统使用说明、当前进度和阶段 7 论文素材卡。

## 正确性决策

- 主比较仍只允许传统单变量模型与单变量 LSTM；多变量 LSTM 标记为特征消融。
- 验证集只用于 Early Stopping，保存验证损失最低权重；测试集固定后评价一次。
- 下一日期按工作日估算并明确提示节假日以交易所日历为准。
- 离线样例及交互训练均标记 `test`，不作为阶段 8 正式结论。

## 验收自检

- [x] 六个页面和普通用户操作闭环可用。
- [x] 页面未直接导入 data/models/evaluation/baselines 核心模块。
- [x] 数据来源、质量报告、全部分析图和离线入口可用。
- [x] 参数配置、主动训练、实时进度和最佳模型信息可用。
- [x] 测试预测、下一日预测、模型对比和结果下载可用。
- [x] session state、cache_data、cache_resource 已接入。
- [x] 中文错误、中文图表标题/坐标和风险声明已接入。
- [x] service 与页面辅助逻辑测试通过。
- [x] Streamlit 启动冒烟检查通过。
- [x] 全量测试通过。

## 测试结论

- 阶段 7 专项：`python -m pytest tests/test_web_services.py -q` → 5 passed。
- 相关回归：`python -m pytest tests/test_lstm.py tests/test_evaluation.py tests/test_web_services.py -q` → 28 passed。
- 全量测试：`python -m pytest -q` → 85 passed in 7.46s。
- 编译检查：`python -m compileall -q app.py pages src` → 通过。
- Streamlit 页面脚本：AppTest 执行首页与 5 个功能页，均无异常。
- Streamlit 冒烟：本机 `127.0.0.1:8765` 返回 HTTP 200 → 通过。

## 遗留与下一阶段条件

- 未运行正式大规模实验；现有样例和交互结果不得写入论文正式结果表。
- 阶段 8 应使用真实行情、多股票、固定多种子协议生成正式图表和结论。

