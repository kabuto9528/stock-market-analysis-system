"""Streamlit 状态、缓存与错误展示辅助。"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src.services import MarketContext, WebApplicationService, WebServiceError


@st.cache_resource(show_spinner=False)
def get_web_service() -> WebApplicationService:
    return WebApplicationService()


def cached_analysis(frame: pd.DataFrame):
    """生成分析图表。

    Plotly Figure 不放入 ``st.cache_data``：图表生成耗时很短，而缓存大型图对象会增加
    序列化成本，并可能在代码热重载或跨页面切换时让前端长期停留在骨架屏。
    """

    return get_web_service().analysis.figures(frame)


@st.cache_data(show_spinner=False)
def cached_experiment(experiment_id: str, stamp: int):
    del stamp
    return get_web_service().experiments.load(experiment_id)


def init_session() -> None:
    st.session_state.setdefault("market_context", None)
    st.session_state.setdefault("last_experiment_id", None)
    st.session_state.setdefault("last_training_summary", None)


def set_market_context(context: MarketContext) -> None:
    previous = st.session_state.get("market_context")
    if previous is None or previous.ts_code != context.ts_code or previous.source != context.source:
        st.session_state.last_training_summary = None
        st.session_state.last_experiment_id = None
    st.session_state.market_context = context


def get_market_context() -> MarketContext | None:
    return st.session_state.get("market_context")


def show_error(exc: Exception) -> None:
    if isinstance(exc, WebServiceError):
        st.error(exc.user_message())
    else:
        st.error("操作失败，请检查输入和本地配置后重试。")
        with st.expander("简要错误信息"):
            st.code(str(exc)[:500])


def require_market_context() -> MarketContext:
    context = get_market_context()
    if context is None:
        st.info("请先到“数据获取”页加载 Tushare、CSV、本地缓存或离线演示数据。")
        st.stop()
    return context
