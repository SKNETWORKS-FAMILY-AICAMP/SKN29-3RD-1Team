import logging
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from app.llm.openai_model import get_openai_model
from app.llm.qwen_model import get_qwen_model
from app.llm.prompts import (
    CODE_GENERATION_PROMPT,
    DEBUG_PROMPT,
    FINAL_PROMPT,
    QUERY_PROMPT,
    THINK_PROMPT,
)
from app.mcp.tools import execute_python
from app.rag.retriever import get_retriever
from app.utils import extract_json

logger = logging.getLogger(__name__)


class SolveState(TypedDict, total=False):
    problem: str
    think_process: str
    code: str
    test_input: str
    expected_output: str
    execution_result: str
    algorithm_query: str
    concept_docs: str
    final_answer: str
    retry_count: int


def _content(response) -> str:
    """LangChain 모델별 응답 타입을 문자열로 정규화한다."""
    return getattr(response, "content", str(response))


def think_node(state: SolveState):
    llm = get_qwen_model("thinking")
    response = llm.invoke(THINK_PROMPT.format(problem=state["problem"]))
    text = _content(response)
    logger.debug("think response: %s", text)
    return {"think_process": text, "retry_count": state.get("retry_count", 0)}


def generate_code_node(state: SolveState):
    llm = get_openai_model()
    response = llm.invoke(
        CODE_GENERATION_PROMPT.format(
            problem=state["problem"],
            think_process=state["think_process"],
        )
    )
    text = _content(response)
    logger.debug("code generation response: %s", text)
    parsed = extract_json(text)
    return {
        "code": parsed["code"],
        "test_input": parsed.get("test_input", ""),
        "expected_output": parsed.get("expected_output", ""),
    }


def execute_node(state: SolveState):
    result = execute_python(
        code=state["code"],
        stdin=state.get("test_input", ""),
    )
    logger.debug("execution result: %s", result)
    return {"execution_result": result}


def execution_router(state: SolveState):
    if state.get("expected_output", "").strip() == state.get("execution_result", "").strip():
        return "success"
    return "retry"


def debug_node(state: SolveState):
    llm = get_openai_model()
    response = llm.invoke(
        DEBUG_PROMPT.format(
            problem=state["problem"],
            code=state["code"],
            test_input=state.get("test_input", ""),
            expected_output=state.get("expected_output", ""),
            actual_output=state.get("execution_result", ""),
        )
    )
    text = _content(response)
    logger.debug("debug response: %s", text)
    parsed = extract_json(text)
    return {
        "code": parsed["fixed_code"],
        "retry_count": state.get("retry_count", 0) + 1,
    }


def retry_router(state: SolveState):
    if state.get("retry_count", 0) >= 3:
        return "failed"
    return "execute"


def algorithm_query_node(state: SolveState):
    llm = get_qwen_model("query")
    response = llm.invoke(QUERY_PROMPT.format(problem=state["problem"]))
    text = _content(response).strip()
    logger.debug("algorithm query response: %s", text)
    return {"algorithm_query": text}


def retrieve_node(state: SolveState):
    retriever = get_retriever()
    docs = retriever.invoke(state["algorithm_query"])
    logger.debug("retrieved %d documents", len(docs))
    return {
        "concept_docs": "\n\n".join(doc.page_content for doc in docs)
    }


def final_answer_node(state: SolveState):
    llm = get_openai_model()
    response = llm.invoke(
        FINAL_PROMPT.format(
            problem=state["problem"],
            think_process=state["think_process"],
            code=state["code"],
            concept_docs=state.get("concept_docs", ""),
        )
    )
    text = _content(response)
    logger.debug("final answer response: %s", text)
    return {"final_answer": text}


def failed_answer_node(state: SolveState):
    return {
        "final_answer": (
            "코드 자동 실행 검증을 3회 시도했지만 샘플 출력과 일치하지 않았습니다.\n\n"
            f"마지막 코드:\n```python\n{state.get('code', '')}\n```\n\n"
            f"마지막 실행 결과:\n{state.get('execution_result', '')}"
        )
    }


builder = StateGraph(SolveState)

builder.add_node("think", think_node)
builder.add_node("generate_code", generate_code_node)
builder.add_node("execute", execute_node)
builder.add_node("debug", debug_node)
builder.add_node("algorithm_query", algorithm_query_node)
builder.add_node("retrieve", retrieve_node)
builder.add_node("final_answer", final_answer_node)
builder.add_node("failed_answer", failed_answer_node)

builder.add_edge(START, "think")
builder.add_edge("think", "generate_code")
builder.add_edge("generate_code", "execute")

builder.add_conditional_edges(
    "execute",
    execution_router,
    {
        "success": "algorithm_query",
        "retry": "debug",
    },
)

builder.add_conditional_edges(
    "debug",
    retry_router,
    {
        "execute": "execute",
        "failed": "failed_answer",
    },
)

builder.add_edge("algorithm_query", "retrieve")
builder.add_edge("retrieve", "final_answer")
builder.add_edge("final_answer", END)
builder.add_edge("failed_answer", END)

solve_graph = builder.compile()
