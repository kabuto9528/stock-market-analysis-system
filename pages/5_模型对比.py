import streamlit as st
from src.web_runtime import (
    cached_experiment,
    get_web_service,
    init_session,
    require_market_context,
    show_error,
)

st.set_page_config(page_title="模型对比", page_icon="⚖️", layout="wide")
init_session(); service = get_web_service(); context = require_market_context()
st.title(f"模型对比与结果下载：{context.ts_code}")
experiments = service.experiments.list_experiments_for_stock(context.ts_code)
if not experiments:
    st.info(f"尚无股票 {context.ts_code} 的实验结果。请先完成一次训练或放入匹配的实验产物。")
    st.stop()
default_id = st.session_state.get("last_experiment_id")
if context.source.startswith("内置固定小型测试样例") and "offline_demo_stage9" in experiments:
    default_id = "offline_demo_stage9"
experiment_index = experiments.index(default_id) if default_id in experiments else 0
experiment_id = st.selectbox("主实验", experiments, index=experiment_index)
secondary_options = ["不合并"] + [item for item in experiments if item != experiment_id]
secondary_id = st.selectbox(
    "兼容的特征消融实验（可选）", secondary_options,
    help="选择同一股票、区间和划分的另一特征模式，可在一张表中展示单变量与多变量 LSTM。",
)
try:
    dashboard = cached_experiment(experiment_id, service.experiments.stamp(experiment_id))
    secondary = None if secondary_id == "不合并" else cached_experiment(secondary_id, service.experiments.stamp(secondary_id))
    comparison, combined_predictions = service.experiments.combine_compatible(dashboard, secondary)
    multi_mask = comparison["model_name"].astype(str).str.contains("多变量")
    main_comparison = comparison.loc[~multi_mask].copy()
    lstm_ablation = comparison.loc[comparison["model_name"].astype(str).str.startswith("LSTM")].copy()

    st.subheader("单变量主比较指标表")
    st.dataframe(main_comparison, width="stretch", hide_index=True)
    st.caption("主比较只包含 Naive、MA、SES、ARIMA 与仅 close 的单变量 LSTM；R² 不参与排名。")
    figures = service.experiments.metric_figures(main_comparison)
    c1, c2 = st.columns(2)
    c1.plotly_chart(figures["rmse"], width="stretch", key="comparison_main_rmse")
    c2.plotly_chart(figures["mae"], width="stretch", key="comparison_main_mae")
    c3, c4 = st.columns(2)
    c3.plotly_chart(
        figures["direction_accuracy"], width="stretch", key="comparison_main_da"
    )
    c4.plotly_chart(
        figures["relative_naive_rmse_improvement_pct"],
        width="stretch",
        key="comparison_main_naive_improvement",
    )

    if multi_mask.any():
        st.subheader("特征消融：单变量与多变量 LSTM")
        st.dataframe(lstm_ablation, width="stretch", hide_index=True)
        st.caption("多变量 LSTM 仅在特征消融区展示，不与传统单变量模型混入主排名。")
        ablation_figures = service.experiments.metric_figures(lstm_ablation)
        a1, a2 = st.columns(2)
        a1.plotly_chart(
            ablation_figures["rmse"], width="stretch", key="comparison_ablation_rmse"
        )
        a2.plotly_chart(
            ablation_figures["direction_accuracy"],
            width="stretch",
            key="comparison_ablation_da",
        )
    else:
        st.info("当前未合并兼容的多变量实验；选择同股票、同区间和同划分的多变量实验后可查看特征消融。")

    st.subheader("下载")
    d1, d2, d3 = st.columns(3)
    d1.download_button("下载主比较指标 CSV", service.experiments.csv_bytes(main_comparison), f"{experiment_id}_main_comparison.csv", "text/csv")
    d2.download_button("下载逐日预测 CSV", service.experiments.csv_bytes(combined_predictions), f"{experiment_id}_daily_predictions.csv", "text/csv")
    d3.download_button("下载实验记录 JSON", service.experiments.json_bytes(dashboard.record.to_dict()), f"{experiment_id}_experiment_record.json", "application/json")
    if multi_mask.any():
        st.download_button("下载特征消融指标 CSV", service.experiments.csv_bytes(lstm_ablation), f"{experiment_id}_feature_ablation.csv", "text/csv")
    st.info(f"实验状态：{dashboard.record.status}；结果类型：{dashboard.record.result_kind}。test 结果不是正式论文结论。")
except Exception as exc:
    show_error(exc)
