from __future__ import annotations

import json
import re
from typing import Any, Dict


# 중요:
# str.format()을 사용하므로 JSON 예시의 중괄호는 반드시 {{ }} 로 escape 해야 한다.
RETRIEVAL_REWRITE_PROMPT = """
너는 코딩테스트 학습 플랫폼의 RAG Query Rewriter다.

사용자 질문을 보고 검색 성능을 높이기 위한 정보를 JSON으로 만든다.

해야 할 일:
1. 핵심 알고리즘 또는 언어 기초 문법 key를 1~3개 추론한다.
2. Vector Search에 사용할 search_query를 만든다.
3. HyDE 방식의 짧은 가상 학습 문서를 만든다.

중요:
- 문제를 풀지 않는다.
- 정답 코드를 만들지 않는다.
- 문제 번호, 플랫폼명, 테스트케이스는 포함하지 않는다.
- search_query는 학습 문서 검색용 자연어 질문으로 만든다.
- hyde_document는 실제 문서처럼 개념, 사용 상황, 핵심 키워드를 포함한다.
- JSON만 출력한다.

사용 가능한 algorithm_key 예시:
dfs, bfs, dp, heap, priority_queue, binary_search, dijkstra,
bellman_ford, floyd_warshall, greedy, backtracking, union_find,
segment_tree, trie, graph, mst, topological_sort, bitmask,
stack, queue, deque, hash, sort, time_complexity, conditional_statement,
파이썬collections, 파이썬itertools, 집합

출력 형식:
{{
  "algorithm_candidates": ["algorithm_key"],
  "search_query": "검색용 자연어 질의",
  "hyde_document": "이 개념을 설명하는 짧은 가상 학습 문서"
}}

사용자 질문:
{query}
""".strip()


def build_retrieval_rewrite_prompt(query: str) -> str:
    return RETRIEVAL_REWRITE_PROMPT.format(query=query.strip())


def _extract_json(text: str) -> str:
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"JSON object not found: {text[:200]}")

    return text[start : end + 1]


def parse_retrieval_rewrite_response(response: Any) -> Dict[str, Any]:
    raw = response.content if hasattr(response, "content") else response
    if not isinstance(raw, str):
        raw = str(raw)

    data = json.loads(_extract_json(raw))

    candidates = data.get("algorithm_candidates") or []
    if isinstance(candidates, str):
        candidates = [candidates]
    candidates = [str(x).strip() for x in candidates if str(x).strip()]

    search_query = str(data.get("search_query", "")).strip()
    hyde_document = str(data.get("hyde_document", "")).strip()

    if not search_query:
        search_query = raw.strip()
    if not hyde_document:
        hyde_document = search_query

    return {
        "algorithm_candidates": candidates[:3],
        "search_query": search_query,
        "hyde_document": hyde_document,
    }


def rewrite_query_with_llm(llm: Any, query: str) -> Dict[str, Any]:
    prompt = build_retrieval_rewrite_prompt(query)
    response = llm.invoke(prompt)
    return parse_retrieval_rewrite_response(response)
