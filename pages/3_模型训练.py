import pandas as pd
import streamlit as st
from src.services import TrainingRequest
from src.web_runtime import get_web_service, init_session, require_market_context, show_error

st.set_page_config(page_title="模型训练", page_icon="🧠", layout="wide")
init_session(); service = get_web_service(); context = require_market_context(); defaults = service.config
st.title(f"LSTM 模型训练：{context.ts_code}")
st.warning("训练只会在点击“开始训练”后执行；页面刷新不会自动启动训练。测试集不参与调参或 Early Stopping。")

with st.form("training_form"):
    mode_cn = st.radio("输入特征", ["单变量（仅 close）", "多变量（OHLC、成交量、成交额）"], horizontal=True)
    c1, c2, c3 = st.columns(3)
    window = c1.number_input("窗口（交易日）", 2, 250, defaults.window_size)
    hidden = c2.selectbox("隐藏单元", [32, 64, 128], index=1)
    layers = c3.selectbox("LSTM 层数", [1, 2], index=1)
    c4, c5, c6 = st.columns(3)
    dropout = c4.number_input("Dropout", 0.0, 0.9, float(defaults.lstm.dropout), 0.05)
    batch = c5.selectbox("Batch Size", [8, 16, 32, 64, 128], index=2)
    lr = c6.selectbox("学习率", [0.01, 0.001, 0.0001], index=1, format_func=lambda x: f"{x:g}")
    c7, c8, c9 = st.columns(3)
    epochs = c7.number_input("最大 Epoch", 1, 500, defaults.lstm.max_epochs)
    patience = c8.number_input("Early Stopping Patience", 1, 100, defaults.lstm.early_stopping_patience)
    seed = c9.number_input("随机种子", 1, 999999, defaults.random_seed)
    device = st.selectbox("训练设备", ["auto", "cpu", "cuda"], help="auto 优先使用可用 CUDA，否则使用 CPU。")
    submitted = st.form_submit_button("开始训练", type="primary")

if submitted:
    request = TrainingRequest(
        mode="univariate" if mode_cn.startswith("单变量") else "multivariate",
        window_size=int(window), hidden_size=int(hidden), num_layers=int(layers),
        dropout=float(dropout), batch_size=int(batch), learning_rate=float(lr),
        max_epochs=int(epochs), patience=int(patience), random_seed=int(seed), device=device,
    )
    progress = st.progress(0, text="准备训练……")
    status = st.empty(); live_chart = st.empty(); rows = []
    def on_epoch(record):
        rows.append({"epoch": record.epoch, "train_loss": record.train_loss, "validation_loss": record.validation_loss})
        progress.progress(min(record.epoch / request.max_epochs, 1.0), text=f"Epoch {record.epoch}/{request.max_epochs}")
        status.caption(f"训练损失 {record.train_loss:.6f}；验证损失 {record.validation_loss:.6f}")
        live_chart.plotly_chart(service.experiments.loss_figure(pd.DataFrame(rows)), width="stretch")
    try:
        with st.spinner("正在训练并保存验证损失最低的模型……"):
            summary = service.training.train(context.frame, request, progress_callback=on_epoch)
        progress.progress(1.0, text="训练完成")
        st.session_state.last_training_summary = summary
        st.session_state.last_experiment_id = summary.experiment_id
        st.success(f"训练完成，实验 ID：{summary.experiment_id}")
    except Exception as exc:
        progress.empty(); show_error(exc)

summary = st.session_state.get("last_training_summary")
if summary is not None:
    st.subheader("最近一次训练结果")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("最优 Epoch", summary.best_epoch)
    c2.metric("最低验证损失", f"{summary.best_validation_loss:.6f}")
    c3.metric("总耗时", f"{summary.elapsed_seconds:.2f} 秒")
    c4.metric("设备", summary.device)
    st.plotly_chart(service.experiments.loss_figure(summary.history), width="stretch")
    st.caption("交互训练结果标记为 test，不作为阶段 8 的正式实验结论。")
