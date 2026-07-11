"""금융 공시 RAG 데모."""
import time

import streamlit as st

from src.chain import ask
from src.config import DATA_DIR

DOCS = {
    "삼성전자": ("samsung", DATA_DIR / "삼성전자1분기.pdf"),
    "SK하이닉스": ("sk_hynix", DATA_DIR / "sk하이닉스1분기.pdf"),
}

st.set_page_config(page_title="공시 분석 RAG", layout="wide")
st.title("📊 금융 공시 분석 시스템")
st.caption("로컬 LLM (llama3.3:70b) · 하이브리드 검색 (BM25 + 벡터) · 완전 온프레미스")

with st.sidebar:
    st.header("설정")
    heavy = st.toggle("70B 모델 사용", value=True,
                      help="끄면 8B (빠르지만 표 해석 약함)")
    mode = st.radio("모드", ["단일 문서", "기업 비교"])

question = st.text_input("질문", placeholder="예: 영업이익")

if mode == "단일 문서":
    target = st.selectbox("문서", list(DOCS.keys()))

if st.button("분석", type="primary") and question:
    targets = [target] if mode == "단일 문서" else list(DOCS.keys())
    cols = st.columns(len(targets))

    for col, name in zip(cols, targets):
        with col:
            st.subheader(name)
            collection, path = DOCS[name]
            with st.spinner("분석 중..."):
                t = time.time()
                ans = ask(collection, question, pdf_path=path, heavy=heavy)
                elapsed = time.time() - t

            st.success(ans.text)
            st.caption(f"⏱ {elapsed:.1f}초")

            with st.expander("📄 근거 출처"):
                for page, snippet in ans.sources:
                    st.markdown(f"**p.{page}**")
                    st.text(snippet.strip()[:200])