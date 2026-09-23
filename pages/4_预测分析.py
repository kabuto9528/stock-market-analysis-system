import streamlit as st
from src.web_runtime import cached_experiment, get_web_service, init_session, require_market_context, show_error

st.set_page_config(page_title="预测分析", page_icon="🔮", layout="wide")
init_session(); service = get_web_service(); context = require_market_context()
st.title("预测分析")
experiments = service.experiments.list_experiments_for_stock(context.ts_code)
if not experiments:
    st.info(f"尚无股票 {context.ts_code} 的实验结果。请先训练，或加载与该股票匹配的预训练资源。")
    st.stop()
default_id = st.session_state.get("last_experiment_id")
if context.source.startswith("内置固定小型测试样例") and "offline_demo_stage9" in experiments:
    default_id = "offline_demo_stage9"
index = experiments.index(default_id) if default_id in experiments else 0
experiment_id = st.selectbox("选择实验 / 预训练模型", experiments, index=index)
try:
    dashboard = cached_experiment(experiment_id, service.experiments.stamp(experiment_id))
    models = dashboard.predictions["model_name"].drop_duplicates().tolist()
    model = st.selectbox("选择模型", models, index=len(models)-1)
    metric = dashboard.comparison[dashboard.comparison["model_name"] == model].iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("RMSE", f"{metric['rmse']:.4f}")
    c2.metric("MAE", f"{metric['mae']:.4f}")
    c3.metric("方向准确率 DA", f"{metric['direction_accuracy']:.2%}")
    c4.metric("相对 Naive 提升率", f"{metric['relative_naive_rmse_improvement_pct']:.2f}%")
    st.caption("R² 仅作价格水平拟合度参考，不用于模型排名。")
    st.plotly_chart(
        service.experiments.prediction_figure(dashboard.predictions, model),
        width="stretch",
        key="prediction_actual_vs_predicted",
    )
    st.plotly_chart(
        service.experiments.error_figure(dashboard.predictions, model),
        width="stretch",
        key="prediction_daily_error",
    )
    st.subheader("下一交易日预测")
    if not dashboard.next_prediction.empty:
        st.dataframe(dashboard.next_prediction, width="stretch", hide_index=True)
    else:
        st.info("该历史实验未保存下一交易日预测。可用当前数据加载预训练模型实时计算。")
        if st.button("加载预训练模型并预测下一交易日"):
            with st.spinner("正在校验并加载预训练模型……"):
                output = service.training.load_pretrained_prediction(experiment_id, context.frame)
            st.dataframe(output, width="stretch", hide_index=True)
except Exception as exc:
    show_error(exc)
