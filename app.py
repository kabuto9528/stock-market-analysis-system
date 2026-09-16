"""Streamlit 首页。"""
import streamlit as st

from src.config import ConfigError, load_config
from src.web_runtime import init_session

st.set_page_config(page_title="LSTM 股票价格预测系统", page_icon="📈", layout="wide")
init_session()
try:
    config = load_config()
except ConfigError as exc:
    st.error(f"配置加载失败：{exc}")
    st.stop()

st.title(config.project_name)
st.caption("使用截至第 t 个交易日的数据，预测第 t+1 个交易日的收盘价。")

left, right = st.columns([1.3, 1])
with left:
    st.subheader("项目介绍")
    st.write("本系统面向股票时间序列教学、科研与毕业答辩演示，整合行情获取、数据质量检查、可视化分析、LSTM 训练、测试集预测与传统模型对比。")
    st.subheader("技术路线")
    st.markdown("**行情数据 → 清洗与顺序划分 → 训练集拟合 Scaler → 滑动窗口 → LSTM / 传统基线 → 公共测试日期评价 → Plotly 可视化**")
with right:
    st.subheader("当前状态")
    st.success(config.development_status)
    st.info("支持 Tushare、CSV、本地缓存和预训练模型；交互训练产物标记为测试结果，正式论文实验在阶段 8 执行。")

st.subheader("使用步骤")
st.markdown("""
1. 在 **数据获取** 页载入行情并查看质量报告。
2. 在 **数据分析** 页查看 K 线、均线、成交量、收益率与相关性。
3. 在 **模型训练** 页设置参数，并点击按钮主动训练；页面刷新不会自动训练。
4. 在 **预测分析** 页查看测试集预测、误差、指标与下一交易日估算。
5. 在 **模型对比** 页查看模型指标并下载实验结果和逐日预测。
""")
st.warning("投资风险声明：本系统仅用于教学与科研演示。股票市场具有高波动和不确定性，预测结果不构成投资建议、收益承诺或交易依据，使用者应独立判断并自行承担风险。")
