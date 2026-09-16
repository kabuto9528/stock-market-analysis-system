# LSTM 模型说明

## 1. 任务与输入

系统使用截至第 t 个交易日的固定长度窗口，预测第 t+1 个交易日的一个收盘价。阶段 5 复用阶段 3 的顺序划分、仅训练集拟合 Scaler 和滑动窗口结果，不重新实现数据切分。

两类实验使用同一个 `LSTMRegressor`：

- 单变量：特征严格为 `close`，`input_size=1`；
- 多变量：特征顺序严格为 `open, high, low, close, vol, amount`，`input_size=6`。

传统 Naive、MA、SES、ARIMA 只能与单变量 LSTM 进入主对比；多变量 LSTM 只用于特征消融实验。

## 2. 网络结构

模型位于 `src/models/lstm.py`，核心结构为：

1. `nn.LSTM(..., batch_first=True)`；
2. 读取窗口最后一个时刻的 LSTM 输出；
3. 通过 `Linear(hidden_size, 1)` 输出一个归一化后的下一交易日收盘价。

输入形状固定为 `(batch, window_size, input_size)`，输出形状固定为 `(batch, 1)`。`input_size`、`hidden_size`、`num_layers` 和 `dropout` 均可配置。PyTorch 只在多层 LSTM 的层间应用 dropout，因此 `num_layers=1` 时有效 dropout 自动为 0，但 metadata 仍保留用户配置值。

## 3. 随机种子与设备

`set_random_seed` 同时设置：

- Python `random`；
- NumPy；
- PyTorch CPU；
- 可用 CUDA 设备的 `manual_seed` 与 `manual_seed_all`；
- PyTorch 确定性算法和 cuDNN deterministic 配置。

默认种子为 42。设备参数支持 `auto`、`cpu`、`cuda`：`auto` 在 CUDA 可用时选择 GPU，否则选择 CPU；显式请求不可用 CUDA 会报错，不会静默改变实验条件。

## 4. 训练器与 Early Stopping

`LSTMTrainer` 固定使用 MSELoss 和 Adam。每个 epoch 记录：

- epoch 编号；
- 按样本数加权的训练 MSE；
- 按样本数加权的验证 MSE；
- epoch 耗时。

Early Stopping 只接收名为 `validation` 的阶段 3 窗口 DataLoader。项目 DataLoader 若把 `test` 传到验证位置会立即报错。测试集不参与梯度更新、超参数选择、Early Stopping 或最佳模型选择。

验证损失严格下降时，训练器把当时的 `state_dict` 深拷贝到 CPU 内存。连续 `early_stopping_patience` 个 epoch 未改善后停止；无论是否提前停止，返回前都会恢复验证损失最低 epoch 的权重。因此后续保存的是最佳验证模型，不是最后一个 epoch。

## 5. 持久化与加载校验

模型和 Scaler 必须分开保存：

- 模型：PyTorch `.pt` 文件；
- Scaler：阶段 3 定义的带 SHA-256 完整性校验 JSON；
- CLI 训练历史：独立 `.history.json`。

模型文件保存：

- `state_dict`；
- `input_size, hidden_size, num_layers, dropout`；
- 特征列及严格顺序；
- 目标列 `close`；
- 窗口长度；
- 随机种子；
- 最佳 epoch；
- 最佳验证损失。

加载使用 `torch.load(..., weights_only=True)`，并校验文件版本、字段集合、state_dict、特征数量、特征顺序、窗口长度和模型结构参数。随后加载独立 Scaler，再次交叉检查 Scaler 特征顺序和模型 `input_size`。任一项不一致都拒绝预测。

## 6. 预测接口

`LSTMPredictor.predict_samples` 对 `WindowedSamples` 执行批量预测，将模型输出通过目标 Scaler 反归一化为真实价格尺度，并保留：

`target_date, previous_close, actual, predicted, model_name`。

这五个字段可在阶段 6 与传统基线结果按公共目标日期对齐。

`predict_next_trading_day` 使用最近 `window_size` 条已知行情预测一个未来目标日。由于系统不能仅靠自然日可靠推断交易所开市日，调用方必须显式传入经过交易日历确认的 `target_date`；该日期必须晚于最后一条已知行情。未来真实值尚不存在，因此返回结果中的 `actual` 为 NaN。

## 7. 本地 CSV 训练

单变量：

`python scripts/train_model.py --file data/example.csv --mode univariate --device auto`

多变量：

`python scripts/train_model.py --file data/example.csv --mode multivariate --device auto`

可选参数包括 `--ts-code`、`--config`、`--output-dir`、`--name` 和 `--overwrite`。脚本执行顺序为 CSV 校验与清洗、70/15/15 顺序划分、仅训练集拟合 Scaler、构造窗口、训练/验证、恢复最佳权重、分别保存模型/Scaler/历史。脚本不会用测试集决定 Early Stopping，也不会输出正式测试指标。

## 8. 自动测试边界

阶段 5 自动测试使用小型合成序列，仅验证代码行为，不代表正式行情、模型精度或论文实验结论。测试覆盖单变量/多变量前向传播、输出形状、小样本 CPU 训练、Early Stopping 最佳权重恢复、禁止测试集早停、保存加载一致性、加载期 metadata 校验、批量及下一交易日预测、固定种子复现和本地 CSV CLI。
