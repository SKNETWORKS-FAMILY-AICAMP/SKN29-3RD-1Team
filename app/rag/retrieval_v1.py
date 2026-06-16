from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from app.rag.vector_store import get_vector_store

# 프로젝트 metadata의 algorithm_key 기준 alias/동의어 사전.
# 역할: 사용자가 알고리즘명을 직접 말하지 않아도 후보 알고리즘을 추론한다.
ALGORITHM_ALIASES: Dict[str, List[str]] = {
    "time_complexity": [
        "시간복잡도", "시간 복잡도", "공간복잡도", "공간 복잡도", "복잡도",
        "빅오", "big-o", "big o", "big_o", "bigo", "점근", "입력 크기",
        "o(1)", "o(log n)", "o(logn)", "o(n)", "o(n log n)", "o(nlogn)", "o(n^2)",
        "로그 시간", "선형 시간", "상수 시간", "메모리 사용량", "공간 사용량",
    ],
    "heap": [
        "힙", "heap", "heapq", "최소값", "최솟값", "최댓값", "최대값",
        "가장 작은 값", "가장 큰 값", "작은 값을 빠르게", "큰 값을 빠르게",
        "최소 원소", "최대 원소", "꺼내는 자료구조",
    ],
    "priority_queue": [
        "우선순위 큐", "우선순위큐", "priorityqueue", "priority queue", "priority_queue", "pq",
        "우선순위대로", "급한 순서", "긴급한 순서", "응급실 대기열", "중요한 작업 먼저",
        "중요도", "우선도", "작업마다 중요", "급한 작업", "먼저 처리",
    ],
    "dfs": ["dfs", "깊이 우선", "깊이우선", "재귀 탐색", "깊게 탐색", "끝까지 탐색", "스택으로 탐색"],
    "bfs": ["bfs", "너비 우선", "너비우선", "가까운 노드", "레벨 순서", "동일 가중치", "가중치 없는 최단거리", "큐로 최단거리"],
    "binary_search": [
        "이분탐색", "이진탐색", "binary search", "lower bound", "upper bound", "parametric", "파라메트릭",
        "정렬된 배열", "절반씩", "범위를 반으로", "탐색 범위", "결정 문제", "가능한 최소", "가능한 최대",
        "조건을 만족하는 최소", "조건 만족 최소",
    ],
    "dp": [
        "dp", "동적계획법", "동적 계획법", "dynamic programming", "lcs", "lis", "메모이제이션", "memoization",
        "점화식", "작은 문제", "중복 부분 문제", "최적 부분 구조", "경우의 수", "최대 이익", "최소 비용",
        "가장 긴 공통 부분", "최장 공통 부분", "공통 부분수열",
    ],
    "backtracking": ["백트래킹", "backtracking", "가지치기", "순열", "조합", "가능한 경우", "되돌아가기", "조건에 맞지 않으면", "정답 후보"],
    "dijkstra": ["다익스트라", "dijkstra", "음수 간선 없는", "양수 가중치", "가중치 그래프", "하나의 시작점", "최단경로", "최단 경로"],
    "bellman_ford": ["벨만포드", "벨만 포드", "bellman", "음수 간선", "음수 가중치", "음수 사이클", "negative edge", "negative cycle"],
    "floyd_warshall": ["플로이드", "floyd", "워셜", "warshall", "모든 정점", "모든 쌍", "all pairs", "전체 정점"],
    "greedy": ["그리디", "greedy", "탐욕", "현재 최선", "그 순간", "지역 최적", "정렬 후 선택"],
    "sort": ["정렬", "sort", "sorting", "오름차순", "내림차순"],
    "selection_sort": ["선택 정렬", "선택정렬", "selection sort"],
    "bubble_sort": ["버블 정렬", "버블정렬", "bubble sort"],
    "insertion_sort": ["삽입 정렬", "삽입정렬", "insertion sort"],
    "merge_sort": ["병합 정렬", "병합정렬", "merge sort", "분할 정복 정렬", "n log n 정렬", "안정 정렬"],
    "quick_sort": ["퀵 정렬", "퀵정렬", "quick sort", "pivot", "피벗"],
    "stack": ["스택", "stack", "후입선출", "lifo", "괄호", "짝이 맞", "최근 것부터", "마지막에 들어온"],
    "queue": ["큐", "queue", "선입선출", "fifo", "먼저 들어온", "대기열"],
    "deque": ["덱", "deque", "양쪽", "앞뒤", "슬라이딩 윈도우", "monotonic queue", "모노톤 큐"],
    "hash": ["해시", "hash", "hashmap", "hashset", "딕셔너리", "dictionary", "map", "set", "중복", "빈도", "개수 세기", "빠르게 찾", "존재 여부"],
    "graph": ["그래프", "graph", "정점", "간선", "노드", "인접 리스트", "인접행렬"],
    "mst": ["최소 신장 트리", "최소 스패닝 트리", "mst", "크루스칼", "프림", "kruskal", "prim", "간선 선택"],
    "union_find": ["유니온 파인드", "union find", "disjoint set", "서로소 집합", "같은 집합", "연결 여부", "사이클 판단", "사이클 확인", "집합 합치기", "대표 원소"],
    "topological_sort": ["위상정렬", "위상 정렬", "topological", "선후관계", "순서가 정해진", "dag"],
    "segment_tree": ["세그먼트 트리", "segment tree", "구간 합", "구간합", "구간 최솟값", "구간 최댓값", "구간 쿼리", "range query", "업데이트 쿼리", "rmq"],
    "trie": ["트라이", "trie", "접두사", "prefix", "자동완성", "문자열 검색", "단어 검색"],
    "bitmask": ["비트마스킹", "bitmask", "비트 마스킹", "상태 압축", "부분집합", "비트 연산"],
    "순열,조합": ["순열", "조합", "순열 조합", "순열과 조합", "npr", "ncr", "permutation", "combination", "경우의 수 나열"],
    "파이썬collections": ["collections", "collection", "counter", "defaultdict", "namedtuple", "deque 라이브러리", "python collections"],
    "파이썬itertools": ["itertools", "permutations", "combinations", "product", "파이썬 순열", "파이썬 조합", "순열 조합 만들기"],
    "집합": ["집합", "set 자료구조", "set", "중복 제거"],
}

# normalize_algorithm_keys.py 실행 전/후 모두 호환되도록 legacy key까지 필터에서 확장한다.
FILTER_KEY_EXPANSIONS: Dict[str, List[str]] = {
    "time_complexity": ["time_complexity", "c/cpp시간복잡도", "파이썬시간복잡도", "java시간복잡도"],
}

INTENT_KEYWORDS = {
    "implementation": ["구현", "코드", "소스", "python", "java", "c++", "작성", "heapq", "라이브러리"],
    "comparison": ["차이", "비교", "다른", "vs", "대신", "같은 거야", "관계"],
    "problem_solving": ["문제", "풀이", "접근", "프로그래머스", "백준", "leetcode", "어떤 알고리즘", "뭘 써"],
    "complexity": ["시간복잡도", "공간복잡도", "복잡도", "big-o", "big o", "big_o", "빅오", "o("],
    "concept_explanation": ["설명", "개념", "이해", "원리", "언제", "무슨 뜻"],
}

LEVEL_KEYWORDS = {
    "beginner": ["처음", "초보", "쉽게", "입문", "기초", "beginner"],
    "intermediate": ["중급", "비교", "차이", "복잡도", "intermediate", "실전"],
    "advanced": ["고급", "최적화", "증명", "심화", "advanced", "pruning", "최악", "상한"],
}


@dataclass(frozen=True)
class RulePattern:
    """여러 곳에 흩어져 있던 if-rule을 데이터로 통합한 구조."""

    algorithm_key: str
    any_keywords: Tuple[str, ...]
    all_groups: Tuple[Tuple[str, ...], ...] = ()
    score: float = 2.0
    reason: str = "rule_pattern"


# 고신뢰/복합 조건을 한 곳에서만 관리한다.
# all_groups는 각 그룹에서 하나 이상 매칭되어야 한다는 뜻이다.
RULE_PATTERNS: Tuple[RulePattern, ...] = (
    RulePattern("priority_queue", ("중요도", "우선도", "급한 순서", "긴급한 순서", "중요한 작업 먼저", "응급실", "먼저 처리"), score=3.2),
    RulePattern("heap", ("전체 정렬은 필요 없", "정렬은 필요 없", "최솟값만", "최소값만", "최댓값만", "최대값만"), score=3.0),
    RulePattern("backtracking", ("정답 후보", "조건에 안 맞", "조건에 맞지 않", "버리는 풀이", "되돌아가면서", "가지치기"), score=3.2),
    RulePattern("dp", ("가장 긴 공통 부분", "최장 공통 부분", "lcs", "공통 부분수열", "공통 부분 문자열"), score=3.2),
    RulePattern("hash", ("같은 값이 이미", "이미 나왔", "중복 여부", "중복 확인", "방문 여부", "존재 여부"), score=3.1),
    RulePattern("segment_tree", ("최소값 구간", "최솟값 구간", "구간 최소", "구간 최솟값", "range minimum", "rmq", "여러 번 최소값", "여러 번 최솟값"), score=3.2),
    RulePattern("bellman_ford", ("음수 사이클", "negative cycle", "음수 간선", "음수 가중치"), score=3.2),
    RulePattern("union_find", ("어느 집합", "속하는지", "대표 원소", "find 연산", "루트 찾", "부모 찾", "연결 여부", "같은 집합", "집합을 합치", "서로소 집합"), score=3.1),
    RulePattern("bfs", ("큐를 쓰면", "큐로", "최단거리를 보장", "최단 거리 보장", "가중치 없는 최단거리"), score=3.0),
    RulePattern("binary_search", ("정렬된 배열", "정렬된 리스트", "정렬된 데이터"), all_groups=(("찾", "탐색", "검색", "원하는 값", "빠르게"),), score=3.1),
    RulePattern("binary_search", ("조건을 만족하는 최소", "조건 만족 최소", "조건을 만족하는 최대", "조건 만족 최대"), score=3.2),
    RulePattern("deque", ("양쪽", "앞뒤"), all_groups=(("삽입", "삭제", "넣", "빼"),), score=3.0),
    RulePattern("deque", ("슬라이딩 윈도우", "sliding window", "모노톤 큐", "monotonic queue"), score=3.1),
    RulePattern("merge_sort", ("분할 정복", "분할정복"), all_groups=(("안정", "n log n", "nlogn", "정렬"),), score=3.0),
    RulePattern("stack", ("최근에 들어온", "나중에 들어온", "마지막에 들어온", "후입선출", "lifo"), score=3.0),
    RulePattern("time_complexity", ("메모리를 얼마나", "메모리 사용량", "공간 사용량", "공간복잡도", "공간 복잡도", "로그 시간", "log time", "로그 복잡도"), score=3.0),
    RulePattern("파이썬collections", ("collections", "counter", "defaultdict", "namedtuple"), score=3.2),
    RulePattern("파이썬itertools", ("itertools", "permutations", "combinations", "product"), score=3.2),
    RulePattern("순열,조합", ("순열 조합", "순열과 조합", "순열", "조합"), all_groups=(("접근", "만들", "문제", "경우"),), score=3.0),
    RulePattern("mst", ("최소 신장", "스패닝", "크루스칼", "프림", "간선 선택"), score=2.6),
    RulePattern("trie", ("접두사", "prefix", "자동완성", "문자열 검색", "단어 검색"), score=2.8),
    RulePattern("bitmask", ("부분집합", "상태 압축", "비트 연산"), score=2.5),
)


@dataclass
class RetrievalPlan:
    need_rag: bool
    intent: str
    algorithm_candidates: List[Dict[str, Any]]
    target_level: str
    filter_used: Optional[Dict[str, Any]] = None
    expanded_query: Optional[str] = None
    rewrite_query: Optional[str] = None
    fallback_used: bool = False
    failure_reasons: Optional[List[str]] = None


RewriteModel = Callable[[str, List[Dict[str, Any]]], str]


def normalize_query(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def compact_query(text: str) -> str:
    return re.sub(r"[\s\-_]+", "", text.strip().lower())


def contains_any(query: str, keywords: Sequence[str]) -> bool:
    q = normalize_query(query)
    cq = compact_query(query)
    return any(k.lower() in q or compact_query(k) in cq for k in keywords)


def unique_candidates(candidates: List[Dict[str, Any]], max_candidates: int = 4) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for item in candidates:
        key = item["algorithm_key"]
        score = float(item.get("score", 1.0))
        reason = item.get("reason", "rule")
        if key not in merged or score > merged[key]["score"]:
            merged[key] = {"algorithm_key": key, "score": score, "reason": reason}

    return [
        {"algorithm_key": item["algorithm_key"], "score": round(item["score"], 3), "reason": item.get("reason", "rule")}
        for item in sorted(merged.values(), key=lambda x: x["score"], reverse=True)[:max_candidates]
    ]


def is_priority_queue_query(query: str) -> bool:
    return contains_any(query, ["우선순위 큐", "우선순위큐", "priority queue", "priorityqueue", "priority_queue", "heapq", "힙큐"])


def is_complexity_query(query: str) -> bool:
    q = normalize_query(query)
    if contains_any(query, ["시간복잡도", "시간 복잡도", "공간복잡도", "공간 복잡도", "복잡도", "빅오", "big-o", "big o", "bigo", "점근", "로그 시간", "선형 시간"]):
        return True
    return bool(re.search(r"o\s*\(\s*(1|log\s*n|n|n\s*log\s*n|n\^?2|n2)\s*\)", q))


def detect_intent(query: str) -> str:
    if is_complexity_query(query):
        return "complexity"

    q = normalize_query(query)
    for intent, keywords in INTENT_KEYWORDS.items():
        if any(k.lower() in q for k in keywords):
            return intent
    return "concept_explanation"


def detect_level(query: str) -> str:
    q = normalize_query(query)
    for level, keywords in LEVEL_KEYWORDS.items():
        if any(k.lower() in q for k in keywords):
            return level
    return "beginner"


def _rule_matches(query: str, pattern: RulePattern) -> bool:
    if not contains_any(query, pattern.any_keywords):
        return False
    return all(contains_any(query, group) for group in pattern.all_groups)


def _alias_score(query: str, aliases: Iterable[str]) -> float:
    q = normalize_query(query)
    cq = compact_query(query)
    score = 0.0
    for alias in aliases:
        a = alias.lower()
        if a in q or compact_query(alias) in cq:
            # 긴 alias일수록 우연한 substring matching 가능성이 낮기 때문에 약간 가산한다.
            score += 1.0 + min(len(a) / 20, 0.5)
    return score


def extract_algorithm_candidates(query: str, max_candidates: int = 4) -> List[Dict[str, Any]]:
    """통합 룰 기반 알고리즘 후보 추출기.

    기존 V1의 extract_high_confidence_candidates(), extract_rule_based_candidates(),
    ALGORITHM_ALIASES 분산 구조를 하나의 후보 스코어링 로직으로 단순화했다.
    """
    candidates: List[Dict[str, Any]] = []

    # 1) 복합/고신뢰 패턴
    for pattern in RULE_PATTERNS:
        if _rule_matches(query, pattern):
            candidates.append({"algorithm_key": pattern.algorithm_key, "score": pattern.score, "reason": pattern.reason})

    # 2) 의도별 보정: 넓은 표현 때문에 엉뚱한 일반 카테고리가 먼저 뜨는 문제 완화
    if is_priority_queue_query(query):
        candidates.extend([
            {"algorithm_key": "priority_queue", "score": 3.0, "reason": "priority_queue_phrase"},
            {"algorithm_key": "heap", "score": 2.4, "reason": "priority_queue_phrase"},
        ])

    if is_complexity_query(query):
        candidates.extend([
            {"algorithm_key": "time_complexity", "score": 2.8, "reason": "complexity_intent"},
            {"algorithm_key": "binary_search", "score": 1.2, "reason": "complexity_intent"},
            {"algorithm_key": "sort", "score": 1.1, "reason": "complexity_intent"},
        ])

    # 3) 최단거리 계열은 문맥에 따라 후보를 여러 개 둔다. hard-filter가 아니라 soft 후보라서 넓게 가져간다.
    if contains_any(query, ["최단거리", "최단 거리", "최단경로", "최단 경로"]):
        if contains_any(query, ["가중치 없", "가중치가 없", "동일 가중치", "같은 비용", "간선 비용 같"]):
            candidates.extend([{"algorithm_key": "bfs", "score": 2.8, "reason": "shortest_path_context"}, {"algorithm_key": "graph", "score": 1.0, "reason": "shortest_path_context"}])
        elif contains_any(query, ["음수", "negative"]):
            candidates.extend([{"algorithm_key": "bellman_ford", "score": 2.9, "reason": "shortest_path_context"}, {"algorithm_key": "dijkstra", "score": 1.4, "reason": "shortest_path_context"}, {"algorithm_key": "graph", "score": 1.0, "reason": "shortest_path_context"}])
        elif contains_any(query, ["모든 정점", "모든 쌍", "전체 정점", "all pairs"]):
            candidates.extend([{"algorithm_key": "floyd_warshall", "score": 2.9, "reason": "shortest_path_context"}, {"algorithm_key": "graph", "score": 1.0, "reason": "shortest_path_context"}])
        elif contains_any(query, ["가중치", "비용", "거리"]):
            candidates.extend([{"algorithm_key": "dijkstra", "score": 2.7, "reason": "shortest_path_context"}, {"algorithm_key": "bellman_ford", "score": 1.4, "reason": "shortest_path_context"}, {"algorithm_key": "floyd_warshall", "score": 1.2, "reason": "shortest_path_context"}, {"algorithm_key": "graph", "score": 1.0, "reason": "shortest_path_context"}])
        else:
            candidates.extend([{"algorithm_key": "bfs", "score": 1.8, "reason": "shortest_path_context"}, {"algorithm_key": "dijkstra", "score": 1.7, "reason": "shortest_path_context"}, {"algorithm_key": "graph", "score": 1.0, "reason": "shortest_path_context"}])

    # 4) alias dictionary 기반 후보 추가
    for algorithm_key, aliases in ALGORITHM_ALIASES.items():
        score = _alias_score(query, aliases)
        if score > 0:
            candidates.append({"algorithm_key": algorithm_key, "score": round(score, 3), "reason": "alias_dictionary"})

    return unique_candidates(candidates, max_candidates=max_candidates)


def should_use_rag(query: str) -> bool:
    q = normalize_query(query)
    if len(q) < 2:
        return False

    greetings = ["안녕", "하이", "hello", "hi", "고마워", "감사"]
    if q in greetings:
        return False

    return True


def build_retrieval_plan(query: str) -> RetrievalPlan:
    return RetrievalPlan(
        need_rag=should_use_rag(query),
        intent=detect_intent(query),
        algorithm_candidates=extract_algorithm_candidates(query),
        target_level=detect_level(query),
    )


def expand_filter_keys(keys: List[str]) -> List[str]:
    expanded: List[str] = []
    for key in keys:
        expanded.extend(FILTER_KEY_EXPANSIONS.get(key, [key]))
    return list(dict.fromkeys(expanded))


def _intent_metadata_filter(intent: str) -> Optional[Dict[str, Any]]:
    """질문 의도별 청크 메타데이터 필터.

    splitter.py에서 생성하는 contains_code / contains_complexity / chunk_type과 연결된다.
    기존 Chroma에 해당 필드가 없더라도 plain search와 병합되므로 Recall 손실을 최소화한다.
    """
    if intent == "implementation":
        return {"contains_code": True}
    if intent == "complexity":
        return {"contains_complexity": True}
    if intent == "concept_explanation":
        return {"chunk_type": {"$in": ["concept", "explanation"]}}
    return None


def _combine_chroma_filters(*filters: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    valid = [f for f in filters if f]
    if not valid:
        return None
    if len(valid) == 1:
        return valid[0]
    return {"$and": valid}


def build_chroma_filter(plan: RetrievalPlan, min_score: float = 2.8, max_keys: int = 2) -> Optional[Dict[str, Any]]:
    """알고리즘 후보 + 질문 의도 기반 metadata filter를 생성한다.

    - algorithm_key: 알고리즘 후보 문서로 검색 범위 축소
    - contains_code: 구현 코드 질문일 때 코드 chunk 우선
    - contains_complexity: 시간복잡도 질문일 때 복잡도 chunk 우선
    - chunk_type: 개념 설명 질문일 때 개념 chunk 우선
    """
    keys = [c["algorithm_key"] for c in plan.algorithm_candidates if float(c.get("score", 0.0)) >= min_score]
    algorithm_filter = None
    if keys:
        filter_keys = expand_filter_keys(keys[:max_keys])
        algorithm_filter = {"algorithm_key": {"$in": filter_keys}}

    intent_filter = _intent_metadata_filter(plan.intent)
    return _combine_chroma_filters(algorithm_filter, intent_filter)


def expand_query_with_candidates(query: str, candidates: List[Dict[str, Any]], max_terms: int = 8) -> str:
    """룰 결과를 hard filter가 아니라 query expansion에도 사용한다."""
    terms: List[str] = []
    for candidate in candidates[:4]:
        key = candidate["algorithm_key"]
        terms.append(key)
        terms.extend(ALGORITHM_ALIASES.get(key, [])[:2])

    dedup_terms = []
    seen = set()
    for term in terms:
        key = compact_query(term)
        if key and key not in seen and key not in compact_query(query):
            seen.add(key)
            dedup_terms.append(term)
        if len(dedup_terms) >= max_terms:
            break

    if not dedup_terms:
        return query
    return f"{query} {' '.join(dedup_terms)}"


def rewrite_query(
    query: str,
    candidates: Optional[List[Dict[str, Any]]] = None,
    rewrite_model: Optional[RewriteModel] = None,
) -> str:
    """Fallback query rewrite.

    - rewrite_model이 전달되면 LLM/sLLM rewrite 모델을 사용할 수 있다.
    - 모델이 없으면 비용 없는 deterministic rewrite로 동작한다.
    """
    candidates = candidates or []

    if rewrite_model is not None:
        try:
            rewritten = rewrite_model(query, candidates)
            if isinstance(rewritten, str) and rewritten.strip():
                return rewritten.strip()
        except Exception as e:
            print(f"[WARN] rewrite_model failed. fallback to deterministic rewrite. reason={e}")

    # 모델이 없을 때의 fallback: 후보가 있으면 후보 중심 query expansion
    if candidates:
        return expand_query_with_candidates(query, candidates, max_terms=10)

    # 후보가 없을 때는 대표적인 사용자 표현을 알고리즘 학습 검색어로 보강한다.
    rewrite_hints: List[str] = []
    if contains_any(query, ["먼저", "급한", "중요", "우선", "일마다", "작업"]):
        rewrite_hints.extend(["priority_queue", "heap", "우선순위 큐"])
    if contains_any(query, ["반", "범위", "조건", "최소", "최대", "정렬된"]):
        rewrite_hints.extend(["binary_search", "이분탐색", "파라메트릭 서치"])
    if contains_any(query, ["연결", "합치", "집합", "사이클"]):
        rewrite_hints.extend(["union_find", "서로소 집합"])
    if contains_any(query, ["최단", "거리", "경로", "가중치"]):
        rewrite_hints.extend(["bfs", "dijkstra", "bellman_ford", "최단경로"])
    if contains_any(query, ["경우의 수", "반복", "점화", "최대 이익", "최소 비용"]):
        rewrite_hints.extend(["dp", "동적 계획법", "점화식"])

    if not rewrite_hints:
        rewrite_hints = ["알고리즘", "자료구조", "개념", "문제 풀이"]

    return f"{query} {' '.join(dict.fromkeys(rewrite_hints))}"


def safe_similarity_search(
    vector_store,
    query: str,
    k: int,
    filter_used: Optional[Dict[str, Any]],
    use_mmr: bool = True,
    fetch_k: Optional[int] = None,
):
    """ChromaDB 검색 함수.

    MMR을 우선 사용하고, MMR/filter 오류 시 similarity_search로 안전하게 fallback한다.
    """
    fetch_k = fetch_k or max(k * 3, 20)

    if use_mmr and hasattr(vector_store, "max_marginal_relevance_search"):
        try:
            if filter_used:
                return vector_store.max_marginal_relevance_search(query, k=k, fetch_k=fetch_k, filter=filter_used)
            return vector_store.max_marginal_relevance_search(query, k=k, fetch_k=fetch_k)
        except Exception as e:
            print(f"[WARN] MMR search failed. fallback to similarity_search. reason={e}")

    if filter_used:
        try:
            return vector_store.similarity_search(query, k=k, filter=filter_used)
        except Exception as e:
            print(f"[WARN] metadata filter failed. fallback to plain vector search. reason={e}")

    return vector_store.similarity_search(query, k=k)


def _doc_identity(doc: Any) -> Tuple[str, str, str]:
    meta = getattr(doc, "metadata", {}) or {}
    return (
        str(meta.get("doc_id") or meta.get("source_file") or meta.get("filename") or ""),
        str(meta.get("section_title") or ""),
        str(getattr(doc, "page_content", ""))[:120],
    )


def merge_docs(*doc_lists: List[Any]) -> List[Any]:
    """filtered search + plain search + fallback search 결과를 순서 보존 방식으로 병합한다."""
    merged: List[Any] = []
    seen = set()
    for docs in doc_lists:
        for doc in docs or []:
            identity = _doc_identity(doc)
            if identity in seen:
                continue
            seen.add(identity)
            merged.append(doc)
    return merged


def vote_documents(docs, final_k: int = 5, candidate_priorities: Optional[Dict[str, float]] = None):
    """청크 검색 결과를 doc_id 기준으로 집계하여 문서 단위 안정성을 높인다.

    candidate_priorities가 주어지면 Retrieval Planner가 예측한 algorithm_key를
    랭킹에 반영한다. 이 보정은 Recall@5에는 영향을 거의 주지 않으면서,
    정답 문서가 Top-5 안에 있으나 1등으로 올라오지 못하는 Hard Query의
    Hit@1을 개선하기 위한 soft re-ranking 단계다.
    """
    if not docs:
        return []

    candidate_priorities = candidate_priorities or {}

    doc_scores = Counter()
    doc_first_rank = {}
    doc_representative = {}
    doc_chunks = defaultdict(list)

    for rank, doc in enumerate(docs, 1):
        meta = doc.metadata
        doc_id = meta.get("doc_id") or meta.get("source_file") or meta.get("filename") or f"unknown_{rank}"
        algorithm_key = str(meta.get("algorithm_key") or "")

        rank_score = 1.0 / rank
        doc_scores[doc_id] += 1.0 + rank_score

        # Planner 후보와 실제 검색 결과가 일치하면 문서 점수를 보정한다.
        # 예: stack 문서가 Top-5에는 있지만 priority_queue가 1등으로 올라오는 경우를 완화한다.
        if algorithm_key in candidate_priorities:
            doc_scores[doc_id] += 4.0 + min(candidate_priorities[algorithm_key], 4.0)

        # 정렬 계열은 세부 정렬 문서가 일반 sort/greedy 의도보다 과도하게 올라오는 경우가 있어
        # 후보가 sort일 때만 약한 보정을 부여한다.
        if "sort" in candidate_priorities and algorithm_key in {"merge_sort", "quick_sort", "selection_sort", "bubble_sort", "insertion_sort"}:
            doc_scores[doc_id] += 1.0

        if doc_id not in doc_first_rank:
            doc_first_rank[doc_id] = rank
            doc_representative[doc_id] = doc

        doc_chunks[doc_id].append(doc)

    ranked_doc_ids = sorted(doc_scores.keys(), key=lambda d: (-doc_scores[d], doc_first_rank[d]))

    results = []
    for doc_id in ranked_doc_ids[:final_k]:
        rep = doc_representative[doc_id]
        results.append(
            {
                "doc_id": doc_id,
                "score": round(doc_scores[doc_id], 4),
                "first_rank": doc_first_rank[doc_id],
                "representative_doc": rep,
                "chunks": doc_chunks[doc_id],
                "chunk_count": len(doc_chunks[doc_id]),
            }
        )

    return results


def detect_retrieval_failures(
    query: str,
    plan: RetrievalPlan,
    raw_docs: List[Any],
    voted_docs: List[Dict[str, Any]],
    min_voted_docs: int = 2,
) -> List[str]:
    """기존 룰베이스/검색이 실패했다고 판단하는 조건을 명시한다."""
    reasons: List[str] = []

    if not plan.algorithm_candidates:
        reasons.append("rule_no_candidates")

    if len(normalize_query(query)) <= 5:
        reasons.append("ambiguous_short_query")

    if not raw_docs:
        reasons.append("no_raw_docs")

    if len(voted_docs) < min_voted_docs:
        reasons.append("too_few_voted_docs")

    # 후보는 있는데 상위 결과의 algorithm_key가 후보와 전혀 안 맞으면 라우팅 불안정으로 본다.
    candidate_keys = {c["algorithm_key"] for c in plan.algorithm_candidates}
    if candidate_keys and voted_docs:
        top_keys = {
            item["representative_doc"].metadata.get("algorithm_key", "")
            for item in voted_docs[: min(3, len(voted_docs))]
        }
        expanded_candidate_keys = set(expand_filter_keys(list(candidate_keys)))
        if top_keys and top_keys.isdisjoint(expanded_candidate_keys):
            reasons.append("candidate_result_mismatch")

    return reasons


def _rank_raw_docs_as_voted(docs: List[Any], final_k: int = 5) -> List[Dict[str, Any]]:
    """Ablation용: Document Voting을 끈 경우에도 반환 스키마를 유지한다."""
    results: List[Dict[str, Any]] = []
    for rank, doc in enumerate((docs or [])[:final_k], 1):
        meta = getattr(doc, "metadata", {}) or {}
        doc_id = meta.get("doc_id") or meta.get("source_file") or meta.get("filename") or f"raw_{rank}"
        results.append({
            "doc_id": doc_id,
            "score": round(1.0 / rank, 4),
            "first_rank": rank,
            "representative_doc": doc,
            "chunks": [doc],
            "chunk_count": 1,
        })
    return results



def candidate_priority_map(plan: RetrievalPlan) -> Dict[str, float]:
    """Document Voting에서 사용할 후보 알고리즘 우선순위 맵을 만든다."""
    priorities: Dict[str, float] = {}
    for candidate in plan.algorithm_candidates:
        key = candidate.get("algorithm_key")
        if not key:
            continue
        priorities[str(key)] = max(priorities.get(str(key), 0.0), float(candidate.get("score", 1.0)))
    return priorities

def retrieve_v1(
    query: str,
    search_k: int = 20,
    final_k: int = 5,
    use_mmr: bool = True,
    enable_fallback_rewrite: bool = False,
    rewrite_model: Optional[RewriteModel] = None,
    use_alias_mapping: bool = True,
    use_metadata_filter: bool = True,
    use_document_voting: bool = True,
) -> Dict[str, Any]:
    """Retrieval V1.1 main entrypoint.

    반환 key는 기존 V1과 호환된다.
    추가 key: expanded_query, rewrite_query, fallback_used, failure_reasons, search_trace
    """
    vector_store = get_vector_store()
    plan = build_retrieval_plan(query)
    if not use_alias_mapping:
        plan.algorithm_candidates = []

    if not plan.need_rag:
        return {
            "query": query,
            "plan": plan,
            "filter_used": None,
            "raw_docs": [],
            "voted_docs": [],
            "expanded_query": None,
            "rewrite_query": None,
            "fallback_used": False,
            "failure_reasons": [],
            "search_trace": [],
        }

    # 룰 결과를 query expansion에 우선 활용한다.
    expanded_query = expand_query_with_candidates(query, plan.algorithm_candidates) if use_alias_mapping else query
    plan.expanded_query = expanded_query

    # 고신뢰 후보만 metadata filter에 사용한다. 단, plain search와 반드시 병합하여 Recall 손실을 줄인다.
    filter_used = build_chroma_filter(plan) if use_metadata_filter else None
    plan.filter_used = filter_used

    filtered_docs: List[Any] = []
    if filter_used:
        filtered_docs = safe_similarity_search(
            vector_store,
            query=expanded_query,
            k=max(final_k, search_k // 2),
            filter_used=filter_used,
            use_mmr=use_mmr,
        )

    plain_docs = safe_similarity_search(
        vector_store,
        query=expanded_query,
        k=search_k,
        filter_used=None,
        use_mmr=use_mmr,
    )

    raw_docs = merge_docs(filtered_docs, plain_docs)
    candidate_priorities = candidate_priority_map(plan) if use_alias_mapping else {}
    voted_docs = vote_documents(raw_docs, final_k=final_k, candidate_priorities=candidate_priorities) if use_document_voting else _rank_raw_docs_as_voted(raw_docs, final_k=final_k)
    failure_reasons = detect_retrieval_failures(query, plan, raw_docs, voted_docs)

    fallback_docs: List[Any] = []
    rewritten_query: Optional[str] = None
    fallback_used = False
    
    if enable_fallback_rewrite and failure_reasons:
        fallback_used = True
        rewritten_query = rewrite_query(query, plan.algorithm_candidates, rewrite_model=rewrite_model)
        plan.rewrite_query = rewritten_query
        plan.fallback_used = True

        # Fallback은 filter 없이 수행한다. 룰이 틀렸을 때 정답 문서를 제거하지 않기 위함이다.
        fallback_docs = safe_similarity_search(
            vector_store,
            query=rewritten_query,
            k=search_k,
            filter_used=None,
            use_mmr=use_mmr,
        )
        raw_docs = merge_docs(raw_docs, fallback_docs)
        voted_docs = vote_documents(raw_docs, final_k=final_k, candidate_priorities=candidate_priorities) if use_document_voting else _rank_raw_docs_as_voted(raw_docs, final_k=final_k)

    plan.failure_reasons = failure_reasons

    return {
        "query": query,
        "plan": plan,
        "filter_used": filter_used,
        "raw_docs": raw_docs,
        "voted_docs": voted_docs,
        "expanded_query": expanded_query,
        "rewrite_query": rewritten_query,
        "fallback_used": fallback_used,
        "failure_reasons": failure_reasons,
        "search_trace": [
            {"stage": "filtered", "query": expanded_query, "filter": filter_used, "count": len(filtered_docs)},
            {"stage": "plain", "query": expanded_query, "filter": None, "count": len(plain_docs)},
            {"stage": "fallback_rewrite", "query": rewritten_query, "filter": None, "count": len(fallback_docs)} if fallback_used else None,
        ],
        "ablation_flags": {
            "use_alias_mapping": use_alias_mapping,
            "use_metadata_filter": use_metadata_filter,
            "use_document_voting": use_document_voting,
            "use_mmr": use_mmr,
        },
    }


def print_retrieval_result(result: Dict[str, Any]):
    plan = result["plan"]

    print("\n" + "=" * 90)
    print(f"Query: {result['query']}")
    print("-" * 90)
    print(f"need_rag        : {plan.need_rag}")
    print(f"intent          : {plan.intent}")
    print(f"target_level    : {plan.target_level}")
    print(f"candidates      : {plan.algorithm_candidates}")
    print(f"filter_used     : {result['filter_used']}")
    print(f"expanded_query  : {result.get('expanded_query')}")
    print(f"fallback_used   : {result.get('fallback_used')}")
    print(f"failure_reasons : {result.get('failure_reasons')}")
    print(f"rewrite_query   : {result.get('rewrite_query')}")

    trace = [item for item in result.get("search_trace", []) if item]
    if trace:
        print("\nSearch Trace")
        print("-" * 90)
        for item in trace:
            print(f"{item['stage']:16} | count={item['count']:>2} | filter={item['filter']} | query={item['query']}")

    print("\nDocument Voting Results")
    print("-" * 90)

    if not result["voted_docs"]:
        print("No results")
        return

    for i, item in enumerate(result["voted_docs"], 1):
        doc = item["representative_doc"]
        meta = doc.metadata
        preview = " ".join(doc.page_content.split())[:180]

        print(f"\n[{i}] score={item['score']} | chunks={item['chunk_count']} | first_rank={item['first_rank']}")
        print(f"algorithm     : {meta.get('algorithm')} ({meta.get('algorithm_key')})")
        print(f"source_file   : {meta.get('source_file')}")
        print(f"section_title : {meta.get('section_title')}")
        print(f"preview       : {preview}")
