"""Streamlit 应用入口；阶段 1 仅提供静态项目首页。"""

import streamlit as st

from src.config import ConfigError, load_config

st.set_page_config(
    page_title="LSTM 股票价格预测系统",
    page_icon="📈",
    layout="wide",
)

try:
    config = load_config()
except ConfigError as exc:
    st.error(f"配置加载失败：{exc}")
    st.stop()

st.title(config.project_name)
st.caption("使用截至第 t 个交易日的数据，预测第 t+1 个交易日的收盘价。")

st.subheader("使用流程")
st.markdown(
    """
1. 配置 Tushare Token 或准备本地 CSV 行情文件。
2. 清洗数据并按时间顺序构造训练、验证、测试集。
3. 显式启动模型训练与基线回测。
4. 查看预测曲线、评价指标和可追溯实验产物。
"""
)

st.subheader("当前开发状态")
st.info(config.development_status)
st.write("本阶段已完成 Python 3.12 工程骨架与配置基础设施；真实数据获取和模型训练尚未实现。")

st.warning("本系统仅用于教学与科研演示，预测结果不构成任何投资建议。")
