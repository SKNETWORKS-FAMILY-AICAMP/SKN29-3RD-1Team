import requests
import streamlit as st
from app.config import FASTAPI_BASE_PATH

API_URL = FASTAPI_BASE_PATH + "/preview/solve"

st.set_page_config(
    page_title="AI 알고리즘 튜터",
    layout="wide"
)

st.title("🤖 AI 알고리즘 튜터")

problem = st.text_area(
    "문제를 붙여넣으세요",
    height=300,
    placeholder="프로그래머스 / 백준 문제를 붙여넣으세요..."
)

if st.button("문제 풀이 시작", type="primary"):

    if not problem.strip():
        st.warning("문제를 입력해주세요.")
        st.stop()

    with st.spinner("AI가 문제를 분석하고 있습니다..."):

        try:
            response = requests.post(
                API_URL,
                json={
                    "problem": problem
                },
                timeout=300
            )

            response.raise_for_status()

            result = response.json()

        except Exception as e:
            st.error(f"API 호출 실패\n\n{e}")
            st.stop()

    st.success("풀이 완료")

    st.divider()

    st.subheader("🧠 사고 과정")

    st.markdown(result["think_process"])

    st.divider()

    st.subheader("🔍 코드 생성 및 디버깅 이력")

    traces = result["execution_traces"]

    for idx, trace in enumerate(traces, start=1):

        success_icon = "✅" if trace["success"] else "❌"

        with st.expander(
            f"{success_icon} 시도 {idx}",
            expanded=(idx == len(traces))
        ):
            st.code(
                trace["code"],
                language="python"
            )

            st.text("실행 결과")

            st.code(
                trace["result"],
                language="text"
            )

    st.divider()

    st.subheader("💻 최종 코드")

    st.code(
        result["code"],
        language="python"
    )

    st.divider()

    st.subheader("📚 RAG 학습 문서")

    seperator = "\n\n======================================================================================\n\n"
    
    if result["concept_docs"].strip():

        docs = result["concept_docs"].split(seperator)

        for idx, doc in enumerate(docs, start=1):

            with st.expander(
                f"문서 {idx}",
                expanded=(idx == 1)
            ):
                st.markdown(doc)

    else:
        st.info("관련 학습 문서가 없습니다.")

    st.divider()

    st.subheader("🎓 최종 해설")

    st.markdown(result["final_answer"])