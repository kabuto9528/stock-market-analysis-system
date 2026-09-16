import datetime as dt
import streamlit as st

from src.web_runtime import get_web_service, init_session, set_market_context, show_error

st.set_page_config(page_title="数据获取", page_icon="🗂️", layout="wide")
init_session(); service = get_web_service()
st.title("数据获取与质量检查")

source = st.radio("数据来源", ["Tushare 在线获取", "CSV 上传", "本地缓存", "离线演示"], horizontal=True)
try:
    context = None
    if source == "Tushare 在线获取":
        c1, c2, c3 = st.columns(3)
        code = c1.text_input("股票代码", "600000.SH", help="示例：600000.SH、000001.SZ")
        start = c2.date_input("起始日期", dt.date(2020, 1, 1))
        end = c3.date_input("结束日期", dt.date.today())
        save_cache = st.checkbox("获取成功后写入本地缓存", value=True)
        if st.button("从 Tushare 获取", type="primary"):
            with st.spinner("正在获取并校验行情数据……"):
                context = service.data.fetch_tushare(code, start, end, save_cache=save_cache)
    elif source == "CSV 上传":
        code = st.text_input("预期股票代码（可选）", help="填写后会校验 CSV 中的 ts_code")
        upload = st.file_uploader("上传 UTF-8 CSV", type=["csv"])
        save_cache = st.checkbox("导入成功后保存到本地缓存")
        if st.button("导入 CSV", type="primary", disabled=upload is None):
            context = service.data.import_csv_bytes(upload.name, upload.getvalue(), code.strip() or None, save_cache=save_cache)
    elif source == "本地缓存":
        codes = service.data.list_cached_stocks()
        if not codes:
            st.info("当前没有本地行情缓存。可先使用 Tushare 或 CSV，并勾选保存缓存。")
        else:
            selected = st.selectbox("缓存股票", codes)
            if st.button("加载本地缓存", type="primary"):
                context = service.data.load_cache(selected)
    else:
        st.caption("固定小型样例仅用于离线功能演示和自动测试，不是正式实验数据或论文结论。")
        if st.button("加载离线演示样例", type="primary"):
            context = service.data.load_offline_demo()
    if context is not None:
        set_market_context(context)
        st.success(f"已加载 {context.ts_code}，共 {len(context.frame)} 行，来源：{context.source}")
except Exception as exc:
    show_error(exc)

context = st.session_state.get("market_context")
if context is not None:
    st.subheader("数据预览")
    st.dataframe(context.frame.tail(30), width="stretch", hide_index=True)
    st.subheader("质量报告")
    st.dataframe(service.data.quality_table(context.quality), width="stretch", hide_index=True)
    if context.duplicate_rows_removed:
        st.info(f"导入时已按交易日期去除重复记录 {context.duplicate_rows_removed} 行。")
