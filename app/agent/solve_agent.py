from typing import TypedDict
from app.llm.qwen_model import get_qwen_model
from app.llm.openai_model import get_openai_model
from app.rag.retrieval_v2 import retrieve_v2
from app.mcp.tools import execute_python
from langgraph.graph import StateGraph, START, END
from app.utils import extract_json, pretty_print
from app.config import ENABLE_LOCAL_MODEL
from app.llm.prompts import (
    THINK_PROMPT_QWEN,
    THINK_PROMPT,
    CODE_GENERATION_PROMPT,
    DEBUG_PROMPT,
    QUERY_PROMPT,
    FINAL_PROMPT,
)

import logging
logger = logging.getLogger(__name__)


class ExecutionTrace(TypedDict):
    code: str
    result: str


class SolveState(TypedDict):
    problem: str

    # think
    think_process: str

    # code & execute
    test_input: str
    expected_output: str
    code: str
    execution_result: str
    execution_traces: list[ExecutionTrace]
    retry_count: int
    code_passed: bool

    # rag
    algorithm_query: str
    concept_docs: str

    # output
    final_answer: str
    

def think_node(state):
    
    if ENABLE_LOCAL_MODEL:
        llm = get_qwen_model("think")

        response = llm.invoke(
            THINK_PROMPT_QWEN.format(
                problem=state["problem"]
            )
        )
    else:
        llm = get_openai_model()

        response = llm.invoke(
            THINK_PROMPT.format(
                problem=state["problem"]
            )
        ).content

    logger.debug(f" ⭐ TEMP LOG: response_type={type(response)} response={response}")
    logger.debug(f"""Problem: {state['problem']}\n"Think process: {response}""")

    return {
        "think_process": response
    }

def generate_code_node(state):

    llm = get_openai_model()

    response = llm.invoke(
        CODE_GENERATION_PROMPT.format(
            problem=state["problem"],
            think_process=state["think_process"]
        )
    )

    logger.debug(f"code generation response: {response.content}")

    parsed = extract_json(response.content)

    return {
        "code": parsed["code"],
        "test_input": parsed["test_input"],
        "expected_output": parsed["expected_output"],
        "execution_traces": [],
        "retry_count": 0,
        "code_passed": False,
    }

def execute_node(state):

    result = execute_python(
        code=state["code"],
        stdin=state["test_input"]
    )

    logger.debug(f"execution result: {result}")

    traces = state.get("execution_traces", []).copy()

    traces.append({
        "code": state["code"],
        "result": result
    })

    return {
        "execution_result": result,
        "execution_traces": traces
    }

def normalize_output(text: str) -> str:
    return "\n".join(
        " ".join(line.split())
        for line in text.strip().splitlines()
    )

def execution_router(state):

    actual = normalize_output(state["execution_result"])
    expected = normalize_output(state["expected_output"])

    if actual == expected:
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
        "retry_count": state["retry_count"] + 1
    }

def retry_router(state):

    if state["retry_count"] >= 3:
        logger.info("retry 3회 초과, 문제풀이 실패")
        return "failed"

    return "retry"

def algorithm_query_node(state):

    llm = get_openai_model()

    response = llm.invoke(
        QUERY_PROMPT.format(
            problem=state["problem"],
            think_process=state['think_process'],
            code=state['code'],
        )
    )
    logger.debug(f"algorithm query response: {response.content}")

    return {
        "code_passed": True,
        "algorithm_query": response.content,
    }

def retrieve_node(state):

    docs = retrieve_v2(
        query=state["algorithm_query"],
        search_k=20,
        final_k=3,
    )['full_documents']

    logger.debug(f"retrieved {len(docs)} documents for query: {state['algorithm_query']}")

    seperator = "\n\n======================================================================================\n\n"
    return {
        "concept_docs": seperator.join(
            item["full_text"]
            for item in docs
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

