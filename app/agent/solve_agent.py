from typing import TypedDict
from app.llm.qwen_model import get_qwen_model
from app.llm.openai_model import get_openai_model
from app.rag.retriever import get_retriever
from app.mcp.tools import execute_python
from langgraph.graph import StateGraph, START, END
from app.utils import extract_json
from app.llm.prompts import (
    THINK_PROMPT,
    CODE_GENERATION_PROMPT,
    DEBUG_PROMPT,
    QUERY_PROMPT,
    FINAL_PROMPT,
)

import logging
logger = logging.getLogger(__name__)

class SolveState(TypedDict):
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

def think_node(state):

    llm = get_qwen_model("think")

    response = llm.invoke(
        THINK_PROMPT.format(
            problem=state["problem"]
        )
    )
    logger.debug(f"""Problem: {state['problem']}\n"Think process: {response.content}""")

    return {
        "think_process": response.content
    }

def generate_code_node(state):

    llm = get_openai_model()

    response = llm.invoke(
        CODE_GENERATION_PROMPT.format(
            problem=state["problem"],
            think_process=state["think_process"]
        ) # KeyError: '\n  "code"'
    )
    logger.debug(f"code generation response: {response.content}")

    parsed = extract_json(response.content)

    return {
        "code": parsed["code"],
        "test_input": parsed["test_input"],
        "expected_output": parsed["expected_output"]
    }

def execute_node(state):

    result = execute_python(
        code=state["code"],
        stdin=state["test_input"]
    )
    logger.debug(f"execution result: {result}")

    return {
        "execution_result": result
    }

def execution_router(state):

    if (
        state["execution_result"].strip()
        == state["expected_output"].strip()
    ):
        return "success"

    return "retry"

def debug_node(state):

    llm = get_openai_model()

    response = llm.invoke(
        DEBUG_PROMPT.format(
            problem=state["problem"],
            code=state["code"],
            test_input=state["test_input"],
            expected_output=state["expected_output"],
            actual_output=state["execution_result"]
        )
    )
    logger.debug(f"debug response: {response.content}")

    parsed = extract_json(response.content)

    return {
        "code": parsed["fixed_code"],
        "retry_count": state.get("retry_count", 0) + 1
    }

def retry_router(state):

    if state["retry_count"] >= 3:
        return "failed"

    return "execute"

def algorithm_query_node(state):

    llm = get_qwen_model("query")

    response = llm.invoke(
        QUERY_PROMPT.format(
            problem=state["problem"]
        )
    )
    logger.debug(f"algorithm query response: {response.content}")

    return {
        "algorithm_query": response.content
    }

def retrieve_node(state):

    retriever = get_retriever()
    docs = retriever.invoke(
        state["algorithm_query"]
    )
    logger.debug(f"retrieved {len(docs)} documents for query: {state['algorithm_query']}")
    logger.debug(f"retrieved docs content: {[doc.page_content[:20] for doc in docs]}")

    return {
        "concept_docs": "\n\n".join(
            doc.page_content
            for doc in docs
        )
    }

def final_answer_node(state):

    llm = get_openai_model()

    response = llm.invoke(
        FINAL_PROMPT.format(
            problem=state["problem"],
            think_process=state["think_process"],
            code=state["code"],
            concept_docs=state["concept_docs"]
        )
    )
    logger.debug(f"final answer response: {response.content}")

    return {
        "final_answer": response.content
    }

builder = StateGraph(SolveState)

builder.add_node("think", think_node)
builder.add_node("generate_code", generate_code_node)
builder.add_node("execute", execute_node)
builder.add_node("debug", debug_node)
builder.add_node("algorithm_query", algorithm_query_node)
builder.add_node("retrieve", retrieve_node)
builder.add_node("final_answer", final_answer_node)

builder.add_edge(START, "think")
builder.add_edge("think", "generate_code")
builder.add_edge("generate_code", "execute")

builder.add_conditional_edges(
    "execute",
    execution_router,
    {
        "success": "algorithm_query",
        "retry": "debug",
    }
)

builder.add_conditional_edges(
    "debug",
    retry_router,
    {
        "retry": "execute",
        "failed": END,
    }
)

builder.add_edge(
    "algorithm_query",
    "retrieve"
)

builder.add_edge(
    "retrieve",
    "final_answer"
)

builder.add_edge(
    "final_answer",
    END
)

solve_graph = builder.compile()

