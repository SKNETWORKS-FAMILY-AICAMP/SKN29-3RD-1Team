import json
from pathlib import Path

import streamlit as st


st.set_page_config(
    page_title="Agent State Viewer",
    layout="wide"
)


# =========================
# Utils
# =========================

def load_jsonl(file_obj):
    rows = []

    for line in file_obj:
        if isinstance(line, bytes):
            line = line.decode("utf-8")

        line = line.strip()

        if not line:
            continue

        rows.append(json.loads(line))

    return rows


def load_jsonl_path(path: str):
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            rows.append(json.loads(line))

    return rows


def render_text(title: str, value):
    if not value:
        return

    st.subheader(title)
    st.markdown(value)


def render_code(title: str, code: str):
    if not code:
        return

    st.subheader(title)
    st.code(code, language="python")


# =========================
# Sidebar
# =========================

st.sidebar.title("Agent Trace Viewer")

uploaded_file = st.sidebar.file_uploader(
    "Upload JSONL",
    type=["jsonl"]
)

path_input = st.sidebar.text_input(
    "Or File Path",
    value=""
)

rows = []

if uploaded_file is not None:
    rows = load_jsonl(uploaded_file)

elif path_input:
    try:
        rows = load_jsonl_path(path_input)
    except Exception as e:
        st.error(f"파일 로드 실패: {e}")
        st.stop()

if not rows:
    st.info("jsonl 파일을 업로드하거나 경로를 입력하세요.")
    st.stop()


# =========================
# Row Selection
# =========================

total_rows = len(rows)

st.sidebar.markdown("---")
st.sidebar.write(f"Total Rows: {total_rows}")

if "current_idx" not in st.session_state:
    st.session_state.current_idx = 0


col_prev, col_next = st.sidebar.columns(2)

with col_prev:
    if st.button("⬅ Prev"):
        st.session_state.current_idx = max(
            0,
            st.session_state.current_idx - 1
        )

with col_next:
    if st.button("Next ➡"):
        st.session_state.current_idx = min(
            total_rows - 1,
            st.session_state.current_idx + 1
        )

selected_idx = st.sidebar.slider(
    "Row",
    min_value=0,
    max_value=total_rows - 1,
    value=st.session_state.current_idx
)

st.session_state.current_idx = selected_idx

row = rows[selected_idx]

timestamp = row.get("timestamp", "")
agent = row.get("agent", "")
state = row.get("state", {})


# =========================
# Header
# =========================

st.title("Agent State Dump Viewer")

meta1, meta2, meta3 = st.columns(3)

with meta1:
    st.metric("Row", selected_idx)

with meta2:
    st.metric("Agent", agent)

with meta3:
    st.metric("Timestamp", str(timestamp))


# =========================
# Problem
# =========================

problem = state.get("problem")

if problem:
    with st.expander("📄 Problem", expanded=True):
        st.markdown(problem)


# =========================
# Think
# =========================

think_process = state.get("think_process")

if think_process:
    with st.expander("🧠 Think Process", expanded=True):
        st.markdown(think_process)


# =========================
# Execution Summary
# =========================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Retry Count",
        state.get("retry_count", 0)
    )

with col2:
    st.metric(
        "Passed",
        str(state.get("code_passed", False))
    )

with col3:
    traces = state.get("execution_traces", [])
    st.metric(
        "Execution Traces",
        len(traces)
    )


# =========================
# Test
# =========================

test_input = state.get("test_input")
expected_output = state.get("expected_output")

if test_input or expected_output:

    with st.expander("🧪 Test Case", expanded=False):

        if test_input:
            st.subheader("Input")
            st.code(test_input)

        if expected_output:
            st.subheader("Expected Output")
            st.code(expected_output)


# =========================
# Generated Code
# =========================

render_code(
    "💻 Generated Code",
    state.get("code")
)


# =========================
# Execution Result
# =========================

execution_result = state.get("execution_result")

if execution_result:

    with st.expander("⚙ Execution Result", expanded=True):
        st.code(execution_result)


# =========================
# Execution Traces
# =========================

execution_traces = state.get(
    "execution_traces",
    []
)

if execution_traces:

    st.subheader("🔍 Execution Traces")

    for idx, trace in enumerate(execution_traces):

        with st.expander(
            f"Trace #{idx + 1}",
            expanded=False
        ):

            code = trace.get("code", "")
            result = trace.get("result", "")

            if code:
                st.markdown("##### Code")
                st.code(
                    code,
                    language="python"
                )

            if result:
                st.markdown("##### Result")
                st.code(result)


# =========================
# RAG
# =========================

algorithm_query = state.get("algorithm_query")
concept_docs = state.get("concept_docs")

if algorithm_query or concept_docs:

    with st.expander("📚 RAG", expanded=False):

        if algorithm_query:
            st.subheader("Algorithm Query")
            st.code(algorithm_query)

        if concept_docs:
            st.subheader("Concept Docs")
            st.markdown(concept_docs)


# =========================
# Final Answer
# =========================

final_answer = state.get("final_answer")

if final_answer:

    with st.expander(
        "✅ Final Answer",
        expanded=True
    ):
        st.markdown(final_answer)


# =========================
# Raw State
# =========================

with st.expander("🗄 Raw State"):
    st.json(state)