# 阶段 5：PyTorch LSTM 模型、训练器、预测器和持久化

- 日期：2026-09-16
- 状态：验收通过；未进入阶段 6
- 执行方式：单 Agent 串行，未创建或委派子 Agent

## 完成内容

1. 实现 `batch_first=True` 的统一 `LSTMRegressor`，单变量与多变量仅由特征配置和 `input_size` 区分。
2. 模型输入为三维窗口，读取最后时刻 LSTM 输出，经线性层回归一个下一交易日收盘价。
3. 实现 Python、NumPy、PyTorch、全部可用 CUDA 设备的随机种子和确定性设置。
4. 实现固定 MSELoss、Adam 的训练器，逐 epoch 记录训练损失、验证损失和耗时。
5. Early Stopping 只允许 validation 分区；训练结束恢复最低验证损失 epoch 的权重。
6. 支持 auto/cpu/cuda 设备选择，无 GPU 时自动回退 CPU。
7. 模型文件保存 state_dict、结构参数、特征列、目标列、窗口、种子、最佳 epoch 和验证损失。
8. Scaler 独立保存；加载时校验特征数量、顺序、窗口长度、模型参数和 state_dict。
9. 实现保留日期并反归一化 close 的批量预测，以及显式 target_date 的下一交易日预测。
10. 新增本地 CSV 单变量/多变量训练脚本，并独立保存逐 epoch 训练历史。
11. 自动测试明确拒绝使用 test DataLoader 执行 Early Stopping。
12. 测试合成序列只用于自动验证，未作为正式实验结果。

## 主要接口

- `LSTMRegressor`、`LSTMModelConfig`
- `set_random_seed(seed)`、`select_device(device)`
- `LSTMTrainer.fit(model, train_loader, validation_loader)`
- `save_model_artifact(...)`、`load_model_artifact(...)`
- `LSTMPredictor.predict_samples(...)`
- `LSTMPredictor.predict_next_trading_day(...)`
- `python scripts/train_model.py --file ... --mode univariate|multivariate`

## 修改文件

- 新增：`src/models/errors.py`、`lstm.py`、`reproducibility.py`、`training.py`
- 新增：`src/models/persistence.py`、`prediction.py`、`scripts/train_model.py`
- 新增：`tests/test_lstm.py`、`docs/LSTM模型说明.md`、`docs/paper/素材卡_阶段5.md`
- 修改：`src/models/__init__.py`、`configs/default.yaml`、`README.md`、`当前进度.md`

## 测试

- 阶段 5：`.\.venv\Scripts\python.exe -m pytest tests/test_lstm.py -q` → `9 passed in 5.65s`。
- 全量：`.\.venv\Scripts\python.exe -m pytest -q` → `66 passed in 9.59s`。
- 编译检查：`.\.venv\Scripts\python.exe -m compileall -q src/models scripts/train_model.py` → 通过。
- 测试不访问真实网络，未生成或伪造正式行情指标。

## 关键结论

- Early Stopping 测试构造“最佳 epoch 后继续恶化”的序列，确认最终模型恢复最佳 epoch 权重。
- 保存加载前后的预测逐元素一致；错误特征顺序、窗口和结构参数均会被拒绝。
- 相同种子下两次短训练损失和预测完全一致；CPU 小样本训练通过。
- CLI 输出明确记录测试集参与训练或早停的样本数为 0。

## 遗留与下一阶段条件

- 尚未计算 RMSE、MAE、R²、DA 或相对 Naive 提升率；这些属于阶段 6。
- 尚未开展真实行情正式训练，未形成模型优劣结论；阶段 8 才执行正式实验。
- 阶段 6 必须按公共 target_date 对齐结果，且主对比只包含单变量 LSTM 与单变量基线。
