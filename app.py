"""Streamlit 首页。"""
import streamlit as st

from src.config import ConfigError, load_config
from src.web_runtime import init_session

st.set_page_config(page_title="股票价格预测系统", page_icon="📈", layout="wide")
init_session()
try:
    load_config()
except ConfigError as exc:
    st.error(f"配置加载失败：{exc}")
    st.stop()

st.markdown(
    """
    <style>
    .stMainBlockContainer { padding-top: 2.8rem; padding-bottom: 2rem; }
    .st-key-home_hero {
        padding: 2rem 2.2rem;
        border-radius: 22px;
        border: 1px solid rgba(105, 151, 216, .25);
        background: radial-gradient(ellipse at top right, rgba(64, 124, 214, .2), transparent 65%),
                    linear-gradient(120deg, rgba(79, 131, 206, .1), rgba(59, 156, 150, .06));
        margin-bottom: 1rem;
    }
    .st-key-home_hero h1 {
        font-size: clamp(1.8rem, 3.2vw, 2.75rem);
        letter-spacing: -.04em;
        padding-top: .6rem;
        padding-bottom: .7rem;
    }
    .home-eyebrow { color: #77a9e8; font-size: .8rem; letter-spacing: .14em; font-weight: 600; }
    .home-intro { opacity: .72; line-height: 1.85; font-size: .96rem; margin: 0 0 1.2rem; }
    .home-tags { display: flex; flex-wrap: wrap; gap: .5rem; }
    .home-tags span {
        border: 1px solid rgba(125, 153, 190, .25);
        border-radius: 999px; padding: .3rem .75rem; font-size: .76rem; opacity: .85;
    }
    .home-overview { border-left: 1px solid rgba(125, 153, 190, .25); padding-left: 1.7rem; }
    .home-overview-label { font-size: .78rem; opacity: .55; margin: .2rem 0 1rem; }
    .home-overview-row { display: flex; gap: .85rem; margin-bottom: 1rem; align-items: center; }
    .home-step-number { color: #77a9e8; font-size: .8rem; font-weight: 600; }
    .home-overview-row strong { font-size: .92rem; font-weight: 600; }
    .home-overview-row small { display: block; opacity: .6; font-size: .76rem; margin-top: .2rem; }
    [class*="st-key-home_feature_"] {
        border-radius: 16px;
        border: 1px solid rgba(125, 153, 190, .22);
        padding: 1.15rem;
        height: 100%;
        background: rgba(125, 153, 190, .035);
    }
    [class*="st-key-home_feature_"] h3 { font-size: 1.05rem; padding-top: .2rem; }
    [class*="st-key-home_feature_"] [data-testid="stCaptionContainer"] { min-height: 3.8rem; }
    .home-feature-number { color: #77a9e8; font-size: .8rem; letter-spacing: .08em; }
    .st-key-home_guide { margin-top: .5rem; }
    @media (max-width: 900px) {
        .st-key-home_hero { padding: 1.5rem; }
        .home-overview { padding-left: 0; border-left: 0; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.container(key="home_hero"):
    intro, overview = st.columns([1.7, 1], gap="large")
    with intro:
        st.markdown('<div class="home-eyebrow">行情 · 分析 · 预测</div>', unsafe_allow_html=True)
        st.title("股票价格预测系统")
        st.markdown(
            '<p class="home-intro">从历史行情出发，分析价格走势，训练预测模型。<br>'
            '使用截至当前交易日的数据，预测下一交易日的收盘价。</p>'
            '<div class="home-tags"><span>Tushare 行情</span><span>交互图表</span>'
            '<span>LSTM 预测</span><span>多模型对比</span></div>',
            unsafe_allow_html=True,
        )
    with overview:
        st.markdown(
            """
            <div class="home-overview">
                <div class="home-overview-label">一站式研究流程</div>
                <div class="home-overview-row"><span class="home-step-number">01</span>
                    <div><strong>了解历史行情</strong><small>价格走势、成交量与收益率</small></div></div>
                <div class="home-overview-row"><span class="home-step-number">02</span>
                    <div><strong>构建预测模型</strong><small>自主训练或使用预训练模型</small></div></div>
                <div class="home-overview-row"><span class="home-step-number">03</span>
                    <div><strong>评估预测表现</strong><small>查看预测误差与模型对比</small></div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.subheader("功能导航")
features = [
    ("数据获取", "在线获取或导入行情，检查数据质量。", "pages/1_数据获取.py"),
    ("数据分析", "查看 K 线、均线、成交量与收益率。", "pages/2_数据分析.py"),
    ("模型训练", "设置模型参数，训练自己的 LSTM 模型。", "pages/3_模型训练.py"),
    ("预测分析", "查看预测走势、误差与下一交易日估算。", "pages/4_预测分析.py"),
    ("模型对比", "对比模型表现，导出指标与预测结果。", "pages/5_模型对比.py"),
]
for number, (column, (title, description, page)) in enumerate(zip(st.columns(5), features), start=1):
    with column, st.container(key=f"home_feature_{number}"):
        st.markdown(f'<span class="home-feature-number">0{number}</span>', unsafe_allow_html=True)
        st.subheader(title)
        st.caption(description)
        st.page_link(page, label="进入页面 →", use_container_width=True)

with st.container(key="home_guide", border=True):
    st.markdown("**第一次使用？从载入行情开始。**")
    st.caption("依次完成数据获取、走势分析、模型训练和预测评估。也可加载离线演示数据，在预测分析页直接查看预训练模型结果。")

st.caption("仅用于教学与科研。股票市场存在不确定性，预测结果不构成投资建议、收益承诺或交易依据。")
