import streamlit as st
from src.web_runtime import cached_analysis, init_session, require_market_context, show_error

st.set_page_config(page_title="数据分析", page_icon="📊", layout="wide")
init_session(); context = require_market_context()
st.title(f"数据分析：{context.ts_code}")
st.caption(f"来源：{context.source}；所有图表由 Plotly 生成，可缩放、悬停和下载。")
try:
    with st.spinner("正在生成分析图表……"):
        figures = cached_analysis(context.frame)
    st.plotly_chart(figures["kline"], width="stretch")
    st.plotly_chart(figures["close_ma"], width="stretch")
    c1, c2 = st.columns(2)
    c1.plotly_chart(figures["volume"], width="stretch")
    c2.plotly_chart(figures["returns"], width="stretch")
    st.plotly_chart(figures["correlation"], width="stretch")
except Exception as exc:
    show_error(exc)
