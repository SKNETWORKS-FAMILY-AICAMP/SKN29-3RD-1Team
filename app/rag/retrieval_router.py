from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.rag.retrieval_v1 import build_retrieval_plan, retrieve_v1, contains_any
from app.rag.retrieval_v2 import retrieve_v2


# 명확한 알고리즘명/자료구조명이 들어간 질문은 V1이 더 안정적이다.
CLEAR_ALGORITHM_KEYWORDS: List[str] = [
    "dfs", "bfs",
    "dp", "동적계획법", "동적 계획법",
    "heap", "힙",
    "priority queue", "priority_queue", "우선순위 큐", "우선순위큐",
    "binary search", "이분탐색", "이진탐색", "lower bound", "upper bound", "parametric",
    "dijkstra", "다익스트라",
    "bellman", "벨만포드", "벨만 포드",
    "floyd", "플로이드", "워셜",
    "greedy", "그리디",
    "backtracking", "백트래킹",
    "union find", "유니온 파인드", "서로소 집합",
    "segment tree", "세그먼트 트리",
    "trie", "트라이",
    "mst", "최소 신장 트리", "크루스칼", "프림",
    "topological", "위상정렬", "위상 정렬",
    "stack", "스택",
    "queue", "큐",
    "deque", "덱",
    "hash", "해시",
    "sort", "정렬",
]


# 문법/자연어/학습자료 추천형 질문은 V2가 더 적합하다.
V2_PREFERRED_KEYWORDS: List[str] = [
    "조건문", "if문", "if 문", "if else", "else if", "elif",
    "비교 연산자", "관계 연산자", "두 수 비교", "두 수를 비교", "값 비교", "크기 비교",
    "문법", "기초 문법", "라이브러리", "사용법",
    "더 공부", "공부할", "자료", "학습", "개념을 알고 싶", "뭘 공부", "무엇을 공부",
    "어떤 자료", "추천", "알고 싶어", "궁금해",
    "방법에 대해", "사용하여", "사용해서",
]


# 알고리즘명이 직접 없지만 의도가 자연어로 표현되는 케이스.
NATURAL_LANGUAGE_PATTERNS: List[str] = [
    "가장 급한", "먼저 처리", "중요한 작업",
    "같은 그룹", "그룹을 합치", "연결 여부",
    "정답 후보", "조건에 안 맞", "버리는",
    "접두사", "자동완성",
    "구간 합", "구간 최솟값", "여러 번",
    "최소 비용으로 연결",
    "값을 빠르게 찾", "빠르게 찾아",
]


def route_retriever(query: str) -> Dict[str, Any]:
    """사용자 질의를 보고 V1/V2 중 적절한 Retriever를 선택한다.

    선택 기준:
    - 명확한 알고리즘 키워드가 있으면 V1
    - 문법/자연어/학습자료 추천형이면 V2
    - Rule Planner 후보가 강하면 V1
    - 후보가 없거나 애매하면 V2
    """
    plan = build_retrieval_plan(query)

    has_v2_keyword = contains_any(query, V2_PREFERRED_KEYWORDS)
    has_natural_language_pattern = contains_any(query, NATURAL_LANGUAGE_PATTERNS)
    has_clear_algorithm_keyword = contains_any(query, CLEAR_ALGORITHM_KEYWORDS)

    candidates = plan.algorithm_candidates or []
    top_score = float(candidates[0].get("score", 0.0)) if candidates else 0.0

    # 문법/자연어 질의는 V2 우선
    if has_v2_keyword or has_natural_language_pattern:
        return {
            "selected_retriever": "v2",
            "routing_reason": "grammar_or_natural_language_query",
            "plan": plan,
        }

    # 명확한 알고리즘명이 있으면 V1 우선
    if has_clear_algorithm_keyword and top_score >= 2.0:
        return {
            "selected_retriever": "v1",
            "routing_reason": "clear_algorithm_keyword",
            "plan": plan,
        }

    # Rule 후보가 충분히 강하면 V1
    if top_score >= 3.0:
        return {
            "selected_retriever": "v1",
            "routing_reason": "high_confidence_rule_candidate",
            "plan": plan,
        }

    # 후보가 없거나 약하면 LLM Rewrite/HyDE가 있는 V2
    return {
        "selected_retriever": "v2",
        "routing_reason": "ambiguous_query_or_no_candidate",
        "plan": plan,
    }


def retrieve(
    query: str,
    llm: Optional[Any] = None,
    search_k: int = 20,
    final_k: int = 5,
    force_retriever: Optional[str] = None,
) -> Dict[str, Any]:
    """최종 통합 Retrieval 엔트리포인트.

    사용자는 V1/V2를 직접 고르지 않고 이 함수만 호출하면 된다.

    반환:
    {
        "selected_retriever": "v1" | "v2",
        "routing_reason": "...",
        "result": {...}
    }
    """
    route = route_retriever(query)

    selected = force_retriever or route["selected_retriever"]
    if selected not in {"v1", "v2"}:
        raise ValueError("force_retriever는 'v1', 'v2', None 중 하나여야 합니다.")

    if selected == "v1":
        result = retrieve_v1(
            query=query,
            search_k=search_k,
            final_k=final_k,
        )
    else:
        result = retrieve_v2(
            query=query,
            llm=llm,
            search_k=search_k,
            final_k=final_k,
            use_llm_rewrite=True,
            use_hyde=True,
            return_full_documents=True,
        )

    return {
        "query": query,
        "selected_retriever": selected,
        "routing_reason": route["routing_reason"] if force_retriever is None else "forced",
        "route_plan": route["plan"],
        "result": result,
    }


def print_routed_result(routed: Dict[str, Any]) -> None:
    """터미널 확인용 출력."""
    result = routed["result"]

    print("\n" + "=" * 90)
    print("Hybrid Retrieval Router")
    print("=" * 90)
    print(f"Query              : {routed['query']}")
    print(f"Selected Retriever : {routed['selected_retriever']}")
    print(f"Routing Reason     : {routed['routing_reason']}")

    if routed["selected_retriever"] == "v2":
        print(f"Search Query       : {result.get('search_query')}")
        print(f"Rewrite Result     : {result.get('rewrite_result')}")
        print(f"HyDE Used          : {bool(result.get('hyde_document'))}")

    print("\nVoted Docs")
    print("-" * 90)
    for i, item in enumerate(result.get("voted_docs") or [], 1):
        doc = item["representative_doc"]
        meta = doc.metadata
        print(
            f"[{i}] score={item.get('score')} | "
            f"algorithm_key={meta.get('algorithm_key')} | "
            f"source={meta.get('source_file')} | "
            f"section={meta.get('section_title')}"
        )

    full_documents = result.get("full_documents") or []
    if full_documents:
        print("\nFull Documents")
        print("-" * 90)
        for i, doc in enumerate(full_documents, 1):
            print(
                f"[{i}] doc_id={doc.get('doc_id')} | "
                f"chunks={doc.get('chunk_count')} | "
                f"source={doc.get('source_file')}"
            )


if __name__ == "__main__":
    test_queries = [
        "DFS와 BFS 차이가 뭐야?",
        "조건문을 사용하여 두 수를 비교하는 방법에 대해 더 공부할 수 있는 자료는 무엇인가요?",
        "가장 급한 작업을 먼저 처리하려면 뭘 공부해야 해?",
        "다익스트라 시간복잡도 알려줘",
    ]

    for q in test_queries:
        routed = retrieve(q)
        print_routed_result(routed)
