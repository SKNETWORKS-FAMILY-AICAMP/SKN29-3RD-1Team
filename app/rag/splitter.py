import re
import json as json_lib
import hashlib
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Any

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


# ============================================================
# 0. 기본 설정
# ============================================================
DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 180
MAX_SEMANTIC_CHUNK_SIZE = 1600
MIN_CHUNK_SIZE = 80

# 문서 길이/구조에 따라 청크 크기를 자동 조정한다.
# - 짧은 문서: 작은 청크로 핵심만 보존
# - 중간 문서: 기본값 사용
# - 긴 문서/complete guide: 큰 청크로 문맥 보존
# - 코드가 많은 문서: overlap을 조금 늘려 코드 주변 설명 손실 방지
SHORT_DOC_THRESHOLD = 2500
MEDIUM_DOC_THRESHOLD = 7000
LONG_DOC_THRESHOLD = 14000


# ============================================================
# 1. 구조 자동 감지
# ============================================================
def detect_format(text: str) -> str:
    if "# 본문" in text and "# 메타데이터" in text:
        return "A"
    if re.search(r"^##\s+\d+\.", text, re.MULTILINE):
        return "B"
    return "C"


# ============================================================
# 2. 메타데이터 JSON 추출
# ============================================================
def extract_metadata_json(text: str) -> Dict[str, Any]:
    """Markdown 내부의 메타데이터 JSON 블록 추출"""
    patterns = [
        r"#+\s*메타데이터\s*```json\s*(\{.*?\})\s*```",
        r"##\s*메타데이터\s*```json\s*(\{.*?\})\s*```",
        r"#\s*메타데이터\s*```json\s*(\{.*?\})\s*```",
    ]
    for pattern in patterns:
        m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if not m:
            continue
        try:
            return json_lib.loads(m.group(1))
        except Exception:
            return {}
    return {}


def _meta_value(meta: Dict[str, Any], keys: List[str], default: str = "") -> str:
    """여러 후보 키 중 첫 값을 문자열로 반환"""
    for key in keys:
        value = meta.get(key)
        if value is None or value == "":
            continue
        if isinstance(value, (list, tuple, set)):
            return ", ".join(map(str, value))
        return str(value)
    return default


def _meta_json_string(value: Any) -> str:
    """
    Chroma metadata 안정성을 위해 list/dict는 JSON 문자열로 저장
    Chroma는 list metadata를 거부하는 경우가 있음.
    """
    if value is None or value == "":
        return ""
    if isinstance(value, str):
        return value
    return json_lib.dumps(value, ensure_ascii=False)


# ============================================================
# 3. 본문 추출
# ============================================================
def extract_body(text: str, fmt: str) -> str:
    """제목/링크/메타데이터를 제거하고 실제 본문만 반환"""
    if fmt == "A":
        m = re.search(r"#\s*본문\s*\n(.*?)(?=\n#\s*메타데이터|\Z)", text, re.DOTALL)
        return m.group(1).strip() if m else text.strip()

    if fmt == "B":
        # ## 1. 부터 메타데이터 전까지. 메타데이터가 없어도 끝까지 추출.
        m = re.search(r"(##\s+1\..*?)(?=\n##\s*메타데이터|\Z)", text, re.DOTALL)
        return m.group(1).strip() if m else text.strip()

    # C: 일반 문서. 중간 위치의 제목/링크/메타데이터도 제거되도록 MULTILINE 사용.
    body = re.sub(r"\n?#+\s*메타데이터\s*```json\s*\{.*?\}\s*```", "", text, flags=re.DOTALL | re.IGNORECASE)
    body = re.sub(r"^#\s*제목\s*\n.*?(?=\n#|\n##|\Z)", "", body, flags=re.MULTILINE | re.DOTALL)
    body = re.sub(r"^#\s*링크\s*\n.*?(?=\n#|\n##|\Z)", "", body, flags=re.MULTILINE | re.DOTALL)
    return body.strip()


# ============================================================
# 4. 전처리
# ============================================================
def clean_text(text: str) -> str:
    code_blocks: List[str] = []

    def _save_code(m):
        code_blocks.append(m.group(0))
        return f"__CODE_BLOCK_{len(code_blocks) - 1}__"

    text = re.sub(r"```[\s\S]*?```", _save_code, text)
    text = re.sub(r"<IMAGE>(.*?)</IMAGE>", r"[그림: \1]", text, flags=re.DOTALL)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\*(.*?)\*", r"\1", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    for i, block in enumerate(code_blocks):
        text = text.replace(f"__CODE_BLOCK_{i}__", block)

    return text.strip()


# ============================================================
# 5. Graph DB용 메타데이터 후보 추출
# ============================================================
def extract_graph_metadata(text: str, doc_meta: Dict[str, Any]) -> Dict[str, str]:
    algorithm = _meta_value(doc_meta, ["algorithm", "title", "name"])
    category = _meta_value(doc_meta, ["category", "type"])
    difficulty = _meta_value(doc_meta, ["difficulty", "level", "target_level"])

    related = doc_meta.get("related_topics") or doc_meta.get("related") or doc_meta.get("keywords") or []
    prerequisites = doc_meta.get("prerequisites") or doc_meta.get("prerequisite") or []

    # 메타데이터가 부족한 문서는 본문 키워드로 관계 후보 보강
    keyword_pool = [
        "DFS", "BFS", "DP", "LCS", "편집거리", "이분탐색", "Lower Bound", "Upper Bound",
        "Parametric Search", "힙", "우선순위 큐", "완전이진트리", "백트래킹", "그래프",
        "정렬", "그리디", "스택", "큐", "해시", "트리", "다익스트라", "벨만포드",
    ]
    inferred = [kw for kw in keyword_pool if kw.lower() in text.lower()]

    if isinstance(related, str):
        related_list = [x.strip() for x in re.split(r"[,/|]", related) if x.strip()]
    else:
        related_list = list(related) if isinstance(related, (list, tuple, set)) else []

    for kw in inferred:
        if kw not in related_list and kw != algorithm:
            related_list.append(kw)

    if isinstance(prerequisites, str):
        prerequisite_list = [x.strip() for x in re.split(r"[,/|]", prerequisites) if x.strip()]
    else:
        prerequisite_list = list(prerequisites) if isinstance(prerequisites, (list, tuple, set)) else []

    return {
        "algorithm": algorithm,
        "category": category,
        "difficulty": difficulty,
        "related_topics": _meta_json_string(related_list[:12]),
        "prerequisites": _meta_json_string(prerequisite_list[:12]),
    }


# ============================================================
# 5-1. 파일명/경로 기반 메타데이터 보강
# ============================================================
def infer_metadata_from_path(metadata: Dict[str, Any], text: str = "") -> Dict[str, str]:
    """
    # 메타데이터 JSON이 비어 있는 문서 보정용.

    중요 원칙:
    - algorithm은 파일명/경로/명시 metadata 중심으로만 추론한다.
    - 본문 키워드는 related_topics 보강에는 사용할 수 있지만 algorithm 결정에는 사용하지 않는다.
    - 예: PriorityQueue 문서 안에 "다익스트라"가 있어도 algorithm은 다익스트라가 아니라 우선순위 큐여야 한다.
    """
    path_raw = " ".join([
        str(metadata.get("filename", "")),
        str(metadata.get("source_file", "")),
        str(metadata.get("source", "")),
        str(metadata.get("source_path", "")),
        str(metadata.get("category", "")),
    ]).lower()
    compact = _safe_lower(path_raw)

    # 더 구체적인 파일명/경로 규칙을 먼저 둔다.
    rules = [
        {
            "keys": ["priorityqueue", "priority_queue", "우선순위큐", "우선순위 큐"],
            "algorithm": "우선순위 큐",
            "category": metadata.get("category", "자료구조") or "자료구조",
            "difficulty": metadata.get("difficulty", metadata.get("target_level", "beginner")) or "beginner",
            "related_topics": ["힙", "큐", "다익스트라"],
            "prerequisites": ["queue", "heap"],
        },
        {
            "keys": ["heapq", "heap_complete", "heap", "힙"],
            "algorithm": "힙",
            "category": metadata.get("category", "자료구조") or "자료구조",
            "difficulty": metadata.get("difficulty", metadata.get("target_level", "mid")) or "mid",
            "related_topics": ["우선순위 큐", "완전이진트리", "정렬", "다익스트라"],
            "prerequisites": ["tree"],
        },
        {
            "keys": ["queue", "큐_generated", "큐.md", "queue_ai"],
            "algorithm": "큐",
            "category": metadata.get("category", "자료구조") or "자료구조",
            "difficulty": metadata.get("difficulty", metadata.get("target_level", "beginner")) or "beginner",
            "related_topics": ["BFS", "Deque", "스택"],
            "prerequisites": [],
        },
        {
            "keys": ["deque", "덱"],
            "algorithm": "덱",
            "category": metadata.get("category", "자료구조") or "자료구조",
            "difficulty": metadata.get("difficulty", metadata.get("target_level", "beginner")) or "beginner",
            "related_topics": ["큐", "스택"],
            "prerequisites": ["queue"],
        },
        {
            "keys": ["stack", "스택"],
            "algorithm": "스택",
            "category": metadata.get("category", "자료구조") or "자료구조",
            "difficulty": metadata.get("difficulty", metadata.get("target_level", "beginner")) or "beginner",
            "related_topics": ["DFS", "큐"],
            "prerequisites": [],
        },
        {
            "keys": ["병합정렬", "병합_정렬", "merge_sort", "mergesort"],
            "algorithm": "병합 정렬",
            "category": "정렬",
            "difficulty": metadata.get("difficulty", metadata.get("target_level", "beginner")) or "beginner",
            "related_topics": ["정렬", "분할정복"],
            "prerequisites": ["recursion"],
        },
        {
            "keys": ["선택정렬", "선택_정렬", "selection_sort", "selectionsort"],
            "algorithm": "선택 정렬",
            "category": "정렬",
            "difficulty": metadata.get("difficulty", metadata.get("target_level", "beginner")) or "beginner",
            "related_topics": ["정렬"],
            "prerequisites": [],
        },
        {
            "keys": ["버블정렬", "버블_정렬", "bubble_sort", "bubblesort"],
            "algorithm": "버블 정렬",
            "category": "정렬",
            "difficulty": metadata.get("difficulty", metadata.get("target_level", "beginner")) or "beginner",
            "related_topics": ["정렬"],
            "prerequisites": [],
        },
        {
            "keys": ["삽입정렬", "삽입_정렬", "insertion_sort", "insertionsort"],
            "algorithm": "삽입 정렬",
            "category": "정렬",
            "difficulty": metadata.get("difficulty", metadata.get("target_level", "beginner")) or "beginner",
            "related_topics": ["정렬"],
            "prerequisites": [],
        },
        {
            "keys": ["퀵정렬", "퀵_정렬", "quick_sort", "quicksort"],
            "algorithm": "퀵 정렬",
            "category": "정렬",
            "difficulty": metadata.get("difficulty", metadata.get("target_level", "beginner")) or "beginner",
            "related_topics": ["정렬", "분할정복"],
            "prerequisites": ["recursion"],
        },
        {
            "keys": ["정렬", "sort", "sorting"],
            "algorithm": "정렬",
            "category": "정렬",
            "difficulty": metadata.get("difficulty", metadata.get("target_level", "beginner")) or "beginner",
            "related_topics": ["정렬"],
            "prerequisites": [],
        },
        {
            "keys": ["dijkstra", "다익스트라"],
            "algorithm": "다익스트라",
            "category": "graph",
            "difficulty": "advanced",
            "related_topics": ["최단경로", "우선순위 큐", "벨만포드", "플로이드워셜"],
            "prerequisites": ["graph", "priority_queue"],
        },
        {
            "keys": ["bellman", "벨만", "벨만포드", "벨만-포드"],
            "algorithm": "벨만포드",
            "category": "graph",
            "difficulty": "advanced",
            "related_topics": ["최단경로", "음수 간선", "음수 사이클", "다익스트라", "플로이드워셜"],
            "prerequisites": ["graph", "shortest_path"],
        },
        {
            "keys": ["floyd", "플로이드", "워셜", "warshall"],
            "algorithm": "플로이드워셜",
            "category": "graph",
            "difficulty": "advanced",
            "related_topics": ["최단경로", "다익스트라", "벨만포드", "DP"],
            "prerequisites": ["graph", "dynamic_programming"],
        },
        {
            "keys": ["bitmask", "비트마스킹", "비트 마스킹"],
            "algorithm": "비트마스킹",
            "category": "algorithm_technique",
            "difficulty": "advanced",
            "related_topics": ["부분집합", "상태 압축", "DP", "조합"],
            "prerequisites": ["binary", "set"],
        },
        {
            "keys": ["topological", "위상정렬", "위상 정렬"],
            "algorithm": "위상정렬",
            "category": "graph",
            "difficulty": "advanced",
            "related_topics": ["DAG", "진입차수", "큐", "그래프"],
            "prerequisites": ["graph", "queue"],
        },
        {
            "keys": ["segment", "세그먼트", "segment tree"],
            "algorithm": "세그먼트 트리",
            "category": "data_structure",
            "difficulty": "advanced",
            "related_topics": ["구간 쿼리", "트리", "펜윅 트리"],
            "prerequisites": ["tree", "recursion"],
        },
        {
            "keys": ["trie", "트라이"],
            "algorithm": "트라이",
            "category": "data_structure",
            "difficulty": "advanced",
            "related_topics": ["문자열", "트리", "prefix"],
            "prerequisites": ["tree", "string"],
        },
    ]

    for rule in rules:
        if any(_safe_lower(k) in compact for k in rule["keys"]):
            return {
                "algorithm": rule["algorithm"],
                "category": rule["category"],
                "difficulty": rule["difficulty"],
                "related_topics": _meta_json_string(rule["related_topics"]),
                "prerequisites": _meta_json_string(rule["prerequisites"]),
                "source_type": "curated" if "고급" in path_raw else str(metadata.get("source_type") or "generated"),
                "content_type": "explanation",
                "style": "hard" if "고급" in path_raw else "easy, code, theory",
                "quality_score": "8.5" if "고급" in path_raw else "8.0",
                "verified": "true",
                "use_for_rag": "true",
                "retrieval_priority": "1" if "고급" in path_raw else "2",
            }

    return {}

# ============================================================
# 5-2. 메타데이터 표준화 / 기본값 보강
# ============================================================
def _safe_lower(value: Any) -> str:
    return str(value or "").lower().replace(" ", "").replace("_", "")


def canonicalize_algorithm(value: str, filename: str = "", source: str = "", text: str = "") -> Dict[str, str]:
    """
    검색/필터링 안정성을 위해 알고리즘 표기를 표준화한다.

    오분류 방지 원칙:
    - value가 있으면 value를 최우선으로 본다.
    - filename/source는 보조 신호로만 사용한다.
    - 본문 text는 algorithm 결정에 사용하지 않는다.
      본문에 관련 알고리즘명이 등장한다고 원본문서의 algorithm을 바꾸면 안 된다.
    """
    value_compact = _safe_lower(value)
    path_compact = _safe_lower(" ".join([filename or "", source or ""]))
    compact = value_compact or path_compact

    # 구체적인 자료구조/정렬/알고리즘을 broad rule보다 먼저 둔다.
    rules = [
        ("priority_queue", "우선순위 큐", ["priorityqueue", "priority_queue", "우선순위큐", "우선순위 큐"]),
        ("heap", "힙", ["heapq", "heap", "힙"]),
        ("deque", "덱", ["deque", "덱"]),
        ("queue", "큐", ["queue", "큐"]),
        ("stack", "스택", ["stack", "스택"]),
        ("merge_sort", "병합 정렬", ["mergesort", "merge_sort", "병합정렬", "병합_정렬", "병합 정렬"]),
        ("selection_sort", "선택 정렬", ["selectionsort", "selection_sort", "선택정렬", "선택_정렬", "선택 정렬"]),
        ("bubble_sort", "버블 정렬", ["bubblesort", "bubble_sort", "버블정렬", "버블_정렬", "버블 정렬"]),
        ("insertion_sort", "삽입 정렬", ["insertionsort", "insertion_sort", "삽입정렬", "삽입_정렬", "삽입 정렬"]),
        ("quick_sort", "퀵 정렬", ["quicksort", "quick_sort", "퀵정렬", "퀵_정렬", "퀵 정렬"]),
        ("sort", "정렬", ["sort", "sorting", "정렬"]),
        ("dijkstra", "다익스트라", ["dijkstra", "다익스트라"]),
        ("bellman_ford", "벨만포드", ["bellman", "벨만포드", "벨만-포드"]),
        ("floyd_warshall", "플로이드워셜", ["floyd", "warshall", "플로이드", "워셜"]),
        ("bfs", "BFS", ["bfs", "너비우선탐색"]),
        ("dfs", "DFS", ["dfs", "깊이우선탐색"]),
        ("dp", "DP", ["dynamicprogramming", "동적계획", "dpcomplete", "dp"]),
        ("binary_search", "이분탐색", ["binarysearch", "이분탐색", "lowerbound", "upperbound", "parametricsearch"]),
        ("backtracking", "백트래킹", ["backtracking", "백트래킹"]),
        ("bruteforce", "완전탐색", ["bruteforce", "완전탐색", "브루트포스"]),
        ("greedy", "그리디", ["greedy", "그리디", "탐욕"]),
        ("hash", "해시", ["hashmap", "hashset", "hash", "해시"]),
        ("trie", "트라이", ["trie", "트라이"]),
        ("segment_tree", "세그먼트 트리", ["segmenttree", "세그먼트트리", "세그먼트 트리"]),
        ("topological_sort", "위상정렬", ["topological", "위상정렬", "위상 정렬"]),
        ("union_find", "유니온 파인드", ["unionfind", "유니온파인드", "disjointset"]),
        ("mst", "최소 신장 트리", ["kruskal", "prim", "크루스칼", "프림", "최소신장트리", "mst"]),
        ("bitmask", "비트마스킹", ["bitmask", "비트마스킹", "비트 마스킹"]),
    ]

    for key, display, aliases in rules:
        if any(_safe_lower(alias) in compact for alias in aliases):
            return {"algorithm": display, "algorithm_key": key, "display_name": display}

    cleaned = str(value or "").strip()
    return {"algorithm": cleaned, "algorithm_key": _safe_lower(cleaned), "display_name": cleaned}

def _json_list(value: Any) -> List[str]:
    if not value:
        return []
    if isinstance(value, str):
        try:
            loaded = json_lib.loads(value)
            if isinstance(loaded, list):
                return [str(x) for x in loaded if str(x).strip()]
        except Exception:
            pass
        return [x.strip() for x in re.split(r"[,/|]", value) if x.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(x) for x in value if str(x).strip()]
    return [str(value)]


def merge_json_lists(*values: Any, limit: int = 16) -> str:
    merged: List[str] = []
    seen = set()
    for value in values:
        for item in _json_list(value):
            norm = item.lower().replace(" ", "")
            if norm and norm not in seen:
                seen.add(norm)
                merged.append(item)
            if len(merged) >= limit:
                return _meta_json_string(merged)
    return _meta_json_string(merged)


def infer_doc_quality_defaults(meta: Dict[str, Any]) -> Dict[str, str]:
    """비어 있는 품질 메타데이터를 일관된 기본값으로 채운다."""
    source_type = str(meta.get("source_type") or "generated").lower()
    if source_type in {"official", "curated", "docs"}:
        return {"quality_score": "9.0", "verified": "true", "use_for_rag": "true", "retrieval_priority": "1"}
    if source_type == "blog":
        return {"quality_score": "7.5", "verified": "true", "use_for_rag": "true", "retrieval_priority": "2"}
    return {"quality_score": "8.0", "verified": "true", "use_for_rag": "true", "retrieval_priority": "2"}


def infer_chunk_content_type(section_title: str, chunk_type: str, text: str, doc_content_type: str = "") -> str:
    """청크 단위 content_type을 보강한다."""
    if doc_content_type:
        return str(doc_content_type)
    raw = f"{section_title}\n{text[:500]}".lower()
    if chunk_type == "problem" or "문제" in raw or "프로그래머스" in raw:
        return "problem"
    if "```" in text or "구현" in raw or "코드" in raw or "소스코드" in raw:
        return "code"
    if "시간복잡도" in raw or "공간복잡도" in raw or "o(" in raw:
        return "complexity"
    if "주의" in raw or "실수" in raw or "헷갈" in raw:
        return "pitfall"
    if "언제 사용" in raw or "사용하면 좋은" in raw or "어떤 조건" in raw:
        return "usage"
    return "explanation"


def infer_chunk_style(section_title: str, text: str, doc_style: str = "") -> str:
    if doc_style:
        return str(doc_style)
    raw = f"{section_title}\n{text[:700]}".lower()
    styles = []
    if any(x in raw for x in ["쉽게 말", "비유", "생각하면", "예를 들어", "떠올려"]):
        styles.append("easy")
        styles.append("analogy")
    if "```" in text or "코드" in raw or "구현" in raw:
        styles.append("code")
    if any(x in raw for x in ["정의", "원리", "개념", "시간복잡도"]):
        styles.append("theory")
    if not styles:
        styles.append("easy")
    return ", ".join(dict.fromkeys(styles))


# ============================================================
# 5-3. 보고서 대응 메타데이터 보강 유틸
# ============================================================
ALGORITHM_ALIASES_FOR_METADATA: Dict[str, List[str]] = {
    "dfs": ["깊이우선탐색", "depth first search", "스택 탐색"],
    "bfs": ["너비우선탐색", "breadth first search", "큐 탐색", "가중치 없는 최단거리"],
    "binary_search": ["이분탐색", "이진탐색", "lower bound", "upper bound", "parametric search"],
    "heap": ["힙", "heapq", "priority heap", "최솟값", "최댓값"],
    "priority_queue": ["우선순위 큐", "priority queue", "heapq", "중요도", "급한 순서"],
    "dp": ["동적계획법", "dynamic programming", "메모이제이션", "점화식", "LCS", "LIS"],
    "dijkstra": ["다익스트라", "dijkstra", "양수 가중치 최단경로", "single source shortest path"],
    "bellman_ford": ["벨만포드", "bellman ford", "음수 간선", "음수 사이클"],
    "floyd_warshall": ["플로이드 워셜", "floyd warshall", "모든 쌍 최단경로"],
    "union_find": ["유니온 파인드", "서로소 집합", "disjoint set", "연결 여부", "집합 합치기"],
    "segment_tree": ["세그먼트 트리", "segment tree", "구간 합", "구간 쿼리", "range query", "RMQ"],
    "trie": ["트라이", "trie", "prefix tree", "접두사", "자동완성"],
    "mst": ["최소 신장 트리", "minimum spanning tree", "kruskal", "prim", "크루스칼", "프림"],
    "topological_sort": ["위상정렬", "topological sort", "DAG", "선후관계"],
    "backtracking": ["백트래킹", "backtracking", "가지치기", "정답 후보"],
    "greedy": ["그리디", "greedy", "탐욕", "현재 최선"],
    "sort": ["정렬", "sorting", "오름차순", "내림차순"],
}


def infer_aliases_for_metadata(algorithm_key: str, doc_meta: Dict[str, Any]) -> str:
    """표준 algorithm_key에 연결되는 동의어를 metadata 문자열(JSON list)로 저장한다."""
    explicit_aliases = doc_meta.get("aliases") or doc_meta.get("alias") or doc_meta.get("synonyms")
    return merge_json_lists(explicit_aliases, ALGORITHM_ALIASES_FOR_METADATA.get(str(algorithm_key or ""), []), limit=20)


def contains_code_block(text: str) -> bool:
    """청크가 구현 코드 또는 코드 블록을 포함하는지 판정한다."""
    raw = text or ""
    lower = raw.lower()
    if "```" in raw:
        return True
    code_markers = ["def ", "class ", "import ", "return ", "for (", "while (", "public static", "#include", "int main", "print("]
    return any(marker in lower for marker in code_markers)


def contains_complexity_text(section_title: str, text: str) -> bool:
    """청크가 시간/공간복잡도 설명을 포함하는지 판정한다."""
    raw = f"{section_title}\n{text}".lower()
    if any(x in raw for x in ["시간복잡도", "시간 복잡도", "공간복잡도", "공간 복잡도", "복잡도", "big-o", "big o", "빅오"]):
        return True
    return bool(re.search(r"\bo\s*\(\s*(1|log\s*n|n|n\s*log\s*n|n\^?2|n2)\s*\)", raw))


def normalize_chunk_type(section_title: str, chunk_type: str, text: str) -> str:
    """concept/code/complexity/pitfall/problem/usage 등 검색 의도와 직접 연결되는 chunk_type 생성."""
    raw = f"{section_title}\n{text[:700]}".lower()
    if chunk_type == "problem" or "문제" in raw or "프로그래머스" in raw or "백준" in raw:
        return "problem"
    if contains_code_block(text) or "구현" in raw or "코드" in raw or "소스코드" in raw:
        return "code"
    if contains_complexity_text(section_title, text):
        return "complexity"
    if any(x in raw for x in ["주의", "실수", "헷갈", "예외"]):
        return "pitfall"
    if any(x in raw for x in ["언제 사용", "사용하면 좋은", "어떤 경우", "판단 기준"]):
        return "usage"
    return chunk_type or "concept"


# ============================================================
# 6. 청킹 전략 3가지 + Adaptive Chunking
# ============================================================

def choose_chunk_config(text: str, doc_format: str = "A", metadata: Optional[Dict[str, Any]] = None) -> Dict[str, int]:
    """
    문서 길이와 구조를 보고 청크 크기/overlap을 결정한다.

    판단 기준:
    - 2,500자 미만: 짧은 개념 문서가 많으므로 800/120
    - 2,500~7,000자: 일반 알고리즘 문서 기준 1,200/180
    - 7,000~14,000자: complete guide급 문맥 보존을 위해 1,400/200
    - 14,000자 이상: 긴 문제풀이/가이드 문서 기준 1,600/220
    - 코드블록이 많은 문서: 코드와 설명이 분리되지 않도록 overlap +40
    - 구조 없는 C 문서: semantic 기준점이 없으므로 1,000~1,200 범위로 보수적 분할
    """
    metadata = metadata or {}
    text_len = len(text or "")
    code_block_count = len(re.findall(r"```", text or "")) // 2
    filename = str(metadata.get("filename", "")).lower()
    source = str(metadata.get("source", "")).lower()
    is_complete = "complete" in filename or "complete" in source or "guide" in filename

    if text_len < SHORT_DOC_THRESHOLD:
        chunk_size, overlap = 800, 120
    elif text_len < MEDIUM_DOC_THRESHOLD:
        chunk_size, overlap = 1200, 180
    elif text_len < LONG_DOC_THRESHOLD:
        chunk_size, overlap = 1400, 200
    else:
        chunk_size, overlap = 1600, 220

    if is_complete and text_len >= MEDIUM_DOC_THRESHOLD:
        chunk_size = max(chunk_size, 1400)
        overlap = max(overlap, 200)

    if code_block_count >= 3:
        overlap += 40

    if doc_format == "C":
        chunk_size = min(chunk_size, 1200)
        overlap = min(overlap, 180)

    # 안정 범위 clamp
    chunk_size = max(700, min(chunk_size, 1700))
    overlap = max(100, min(overlap, 260))
    max_semantic_size = max(chunk_size, min(chunk_size + 250, 1800))

    return {
        "chunk_size": chunk_size,
        "overlap": overlap,
        "max_semantic_size": max_semantic_size,
        "text_len": text_len,
        "code_block_count": code_block_count,
    }

def chunk_fixed(text: str, size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> List[Dict[str, Any]]:
    """전략 1: Fixed-size Chunking"""
    chunks, start = [], 0
    step = max(size - overlap, 1)
    while start < len(text):
        chunk_text = text[start:start + size].strip()
        if len(chunk_text) >= MIN_CHUNK_SIZE:
            chunks.append({
                "text": chunk_text,
                "title": f"fixed_{len(chunks)}",
                "type": "concept",
                "strategy": "fixed_size",
            })
        start += step
    return chunks


def _split_text_preserving_code_blocks(text: str, max_size: int, overlap: int) -> List[str]:
    """
    코드 블록을 절대 내부에서 자르지 않는 Recursive 분할기.
    분할 우선순위: 섹션 경계 -> 문단 경계 -> 리스트/문장 경계 -> 코드 블록 외부.
    코드 블록 자체가 max_size를 초과해도 하나의 chunk로 유지한다.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_size,
        chunk_overlap=overlap,
        separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n- ", "\n* ", "\n", ". ", " ", ""],
    )
    parts = re.split(r"(```[\s\S]*?```)", text or "")
    chunks: List[str] = []
    buffer = ""

    def flush_buffer():
        nonlocal buffer
        if buffer.strip():
            chunks.extend([c.strip() for c in splitter.split_text(buffer) if c.strip()])
        buffer = ""

    for part in parts:
        if not part:
            continue
        if part.startswith("```") and part.endswith("```"):
            # 코드 블록 앞 설명은 먼저 분할하고, 코드 블록은 통째로 보존한다.
            flush_buffer()
            chunks.append(part.strip())
            continue
        buffer += part
    flush_buffer()

    # 코드 블록이 아닌 인접 짧은 chunk는 max_size 이내에서 병합한다.
    merged: List[str] = []
    for chunk in chunks:
        is_code = chunk.startswith("```") and chunk.endswith("```")
        if not merged or is_code or merged[-1].startswith("```"):
            merged.append(chunk)
            continue
        candidate = f"{merged[-1]}\n\n{chunk}".strip()
        if len(candidate) <= max_size:
            merged[-1] = candidate
        else:
            merged.append(chunk)
    return merged


def chunk_recursive(
    text: str,
    max_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
    title: str = "recursive",
    chunk_type: str = "concept",
) -> List[Dict[str, Any]]:
    """전략 2: Recursive Chunking. 코드블록 내부 분할 금지."""
    raw_chunks = _split_text_preserving_code_blocks(text, max_size=max_size, overlap=overlap)
    return [
        {
            "text": c.strip(),
            "title": title if len(raw_chunks) == 1 else f"{title}_{i}",
            "type": chunk_type,
            "strategy": "recursive_code_safe",
            "sub_chunk_index": i,
        }
        for i, c in enumerate(raw_chunks)
        if len(c.strip()) >= MIN_CHUNK_SIZE
    ]

def _semantic_sections(text: str, doc_format: str = "A") -> List[Dict[str, Any]]:
    """헤더 기준으로 의미 단위 섹션 생성"""
    chunks: List[Dict[str, Any]] = []
    cur = {"title": "intro", "lines": [], "type": "concept"}

    for line in text.split("\n"):
        # 문제 단위는 통째로 보존
        if re.match(r"^###\s*문제\s*\d+", line):
            if cur["lines"]:
                chunks.append(cur)
            cur = {"title": line.lstrip("#").strip(), "lines": [line], "type": "problem"}
            continue

        # ## 헤더 기준 semantic section 분리
        if re.match(r"^##\s+", line):
            skip = ["메타데이터"]
            if any(s in line for s in skip):
                if cur["lines"]:
                    chunks.append(cur)
                cur = {"title": "", "lines": [], "type": "concept"}
                continue

            if cur["lines"]:
                chunks.append(cur)

            title = line.lstrip("#").strip()
            title = re.sub(r"^\d+\.\s*", "", title)
            cur = {"title": title, "lines": [line], "type": "concept"}
            continue

        cur["lines"].append(line)

    if cur["lines"]:
        chunks.append(cur)

    cleaned_sections = []
    for c in chunks:
        joined = "\n".join(c["lines"]).strip()
        if len(joined) >= MIN_CHUNK_SIZE:
            cleaned_sections.append({"text": joined, "title": c["title"], "type": c["type"]})
    return cleaned_sections


def chunk_semantic(
    text: str,
    doc_format: str = "A",
    max_size: int = MAX_SEMANTIC_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Dict[str, Any]]:
    """
    전략 3: Semantic Chunking
    - ## / ### 문제 단위로 먼저 자름
    - 긴 semantic section은 Recursive로 한 번 더 자름
    - 따라서 최종 전략은 Adaptive Semantic + Recursive
    """
    sections = _semantic_sections(text, doc_format)
    final_chunks: List[Dict[str, Any]] = []

    for section in sections:
        section_text = section["text"]
        title = section["title"]
        chunk_type = section["type"]

        if len(section_text) <= max_size:
            final_chunks.append({
                "text": section_text,
                "title": title,
                "type": chunk_type,
                "strategy": "semantic",
                "sub_chunk_index": 0,
            })
            continue

        # 너무 긴 의미 섹션은 recursive로 재분할
        sub_chunks = chunk_recursive(
            section_text,
            max_size=max_size,
            overlap=overlap,
            title=title,
            chunk_type=chunk_type,
        )
        for sub in sub_chunks:
            sub["strategy"] = "adaptive_semantic_recursive"
            sub["parent_section_title"] = title
        final_chunks.extend(sub_chunks)

    return final_chunks


# ============================================================
# 7. 전략 평가
# ============================================================
def evaluate_strategy(chunks: List[Dict[str, Any]], eval_cases: Optional[List[Dict[str, List[str]]]] = None) -> Dict[str, Any]:
    texts = [c["text"] for c in chunks]
    sizes = [len(t) for t in texts]

    if not texts:
        return {
            "count": 0,
            "avg_size": 0,
            "max_size": 0,
            "code_integrity": 0.0,
            "context_completeness": 0.0,
            "size_quality": 0.0,
            "retrieval": 0.0,
            "total": 0.0,
        }

    broken_code = sum(1 for t in texts if t.count("```") % 2 != 0)
    code_pct = round((1 - broken_code / max(len(texts), 1)) * 100, 1)

    marker_groups = [
        ["핵심 개념", "풀이 전략"],
        ["소스코드", "시간 복잡도"],
        ["주의", "실수"],
        ["언제", "사용"],
    ]
    ctx_hits, ctx_total = 0, 0
    for group in marker_groups:
        for t in texts:
            found = [m for m in group if m.lower() in t.lower()]
            if found:
                ctx_total += 1
                if len(found) == len(group):
                    ctx_hits += 1
                break
    ctx_pct = round((ctx_hits / max(ctx_total, 1)) * 100, 1)

    # 600~1600자 범위를 안정적인 RAG 청크 크기로 간주
    good_size = sum(1 for s in sizes if 300 <= s <= MAX_SEMANTIC_CHUNK_SIZE)
    oversized = sum(1 for s in sizes if s > MAX_SEMANTIC_CHUNK_SIZE)
    size_pct = round((good_size / len(sizes)) * 100 - (oversized / len(sizes)) * 20, 1)
    size_pct = max(0.0, min(100.0, size_pct))

    if eval_cases is None:
        eval_cases = [
            {"query_kw": ["시간 복잡도"], "answer_kw": ["O(", "복잡도"]},
            {"query_kw": ["코드", "구현"], "answer_kw": ["def ", "class ", "public ", "return", "import "]},
            {"query_kw": ["실수", "주의"], "answer_kw": ["실수", "주의", "헷갈", "예외"]},
            {"query_kw": ["언제", "사용"], "answer_kw": ["경우", "사용", "패턴", "문제"]},
            {"query_kw": ["개념", "원리"], "answer_kw": ["개념", "원리", "동작", "구조"]},
        ]

    hits = 0
    for case in eval_cases:
        best_idx, best_score = 0, -1
        for i, text in enumerate(texts):
            score = sum(1 for kw in case["query_kw"] if kw.lower() in text.lower())
            if score > best_score:
                best_score, best_idx = score, i
        if best_score > 0 and any(kw.lower() in texts[best_idx].lower() for kw in case["answer_kw"]):
            hits += 1
    retrieval_pct = round(hits / max(len(eval_cases), 1) * 100, 1)

    total = (
        code_pct * 0.20
        + ctx_pct * 0.25
        + size_pct * 0.25
        + retrieval_pct * 0.30
    )

    return {
        "count": len(chunks),
        "avg_size": round(mean(sizes), 1),
        "max_size": max(sizes),
        "min_size": min(sizes),
        "code_integrity": code_pct,
        "context_completeness": ctx_pct,
        "size_quality": round(size_pct, 1),
        "retrieval": retrieval_pct,
        "total": round(total, 1),
    }


def compare_strategies(text: str, fmt: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, Any]]:
    """Fixed / Recursive / Semantic / Adaptive 전략 비교 출력"""
    config = choose_chunk_config(text, fmt, metadata)
    fixed = chunk_fixed(text, size=config["chunk_size"], overlap=config["overlap"])
    recursive = chunk_recursive(text, max_size=config["chunk_size"], overlap=config["overlap"])
    semantic_only = _semantic_sections(text, fmt)
    semantic = [
        {**c, "strategy": "semantic", "sub_chunk_index": 0}
        for c in semantic_only
    ]
    adaptive = chunk_semantic(text, fmt, max_size=config["max_semantic_size"], overlap=config["overlap"])

    results = {
        "fixed_size": evaluate_strategy(fixed),
        "recursive": evaluate_strategy(recursive),
        "semantic_only": evaluate_strategy(semantic),
        "adaptive": evaluate_strategy(adaptive),
    }

    print(f"    [동적 청킹 설정] text_len={config['text_len']} | chunk_size={config['chunk_size']} | overlap={config['overlap']} | max_semantic={config['max_semantic_size']} | code_blocks={config['code_block_count']}")
    print(f"    {'전략':<15} {'청크':>4} {'평균':>6} {'최대':>6} {'코드':>7} {'문맥':>7} {'크기':>7} {'검색':>7} {'종합':>6}")
    print(f"    {'─' * 80}")
    best_name = max(results, key=lambda k: results[k]["total"])
    for name, ev in results.items():
        marker = " *" if name == best_name else ""
        print(
            f"    {name:<15} {ev['count']:>3}개 {ev['avg_size']:>6} {ev['max_size']:>6} "
            f"{ev['code_integrity']:>6.1f}% {ev['context_completeness']:>6.1f}% "
            f"{ev['size_quality']:>6.1f}% {ev['retrieval']:>6.1f}% {ev['total']:>5.1f}{marker}"
        )
    return results


# ============================================================
# 8. 기존 호환 — PDF / 일반 텍스트용
# ============================================================
def split_documents(documents: List[Document]) -> List[Document]:
    """PDF / 일반 텍스트용. 문서 길이에 따라 Recursive 설정을 동적으로 적용한다."""
    results: List[Document] = []
    for doc in documents:
        config = choose_chunk_config(doc.page_content, "C", doc.metadata)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=config["chunk_size"],
            chunk_overlap=config["overlap"],
            separators=["\n```", "\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
        )
        results.extend(splitter.split_documents([doc]))
    return results



# ============================================================
# 8-2. 원본문서/청크 추적 ID 생성
# ============================================================
def _slugify_id(value: Any, fallback: str = "doc") -> str:
    """파일명/경로를 Chroma metadata와 평가셋에서 쓰기 좋은 안정 ID로 변환"""
    raw = str(value or "").strip().replace("\\", "/")
    if not raw:
        raw = fallback
    stem = Path(raw).stem or raw
    stem = re.sub(r"[^0-9A-Za-z가-힣_\-]+", "_", stem)
    stem = re.sub(r"_+", "_", stem).strip("_")
    return stem or fallback


def build_document_trace_metadata(metadata: Dict[str, Any], text: str = "") -> Dict[str, str]:
    """
    원본문서 추적성(traceability)을 위한 문서 단위 ID 생성.
    - original_doc_id/doc_id: 원본문서 기준 안정 ID
    - source_file: 파일명
    - source_path: 상대 경로 또는 원본 경로
    - source_hash: 경로+본문 기반 짧은 해시. 같은 파일명 충돌 방지용
    """
    source_path = str(metadata.get("source") or metadata.get("source_path") or "")
    filename = str(metadata.get("filename") or metadata.get("source_file") or Path(source_path).name or "document.md")
    doc_base = _slugify_id(metadata.get("doc_id") or metadata.get("original_doc_id") or filename or source_path)
    hash_input = f"{source_path}|{filename}|{text[:1000]}".encode("utf-8", errors="ignore")
    source_hash = hashlib.md5(hash_input).hexdigest()[:10]
    original_doc_id = f"{doc_base}_{source_hash}"
    return {
        "original_doc_id": original_doc_id,
        "doc_id": original_doc_id,
        "source_file": filename,
        "source_path": source_path,
        "source_hash": source_hash,
    }

# ============================================================
# 9. 메인 — md 파일 청킹
# ============================================================
def split_md_smart(
    text: str,
    metadata: Optional[Dict[str, Any]] = None,
    show_comparison: bool = False,
    max_semantic_size: int = MAX_SEMANTIC_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Document]:
    """
    Markdown 문서 -> LangChain Document 리스트

    처리 흐름
    1. 문서 구조 감지
    2. 메타데이터 JSON 추출
    3. 본문 추출 및 전처리
    4. A/B 구조: Adaptive Semantic Chunking
    5. C 구조: Recursive Chunking fallback
    6. Vector DB / Graph DB용 metadata 부착
    """
    metadata = dict(metadata or {})
    fmt = detect_format(text)
    doc_meta = extract_metadata_json(text)

    body = extract_body(text, fmt)
    cleaned = clean_text(body)
    chunk_config = choose_chunk_config(cleaned, fmt, metadata)
    graph_meta = extract_graph_metadata(cleaned, doc_meta)
    inferred_meta = infer_metadata_from_path(metadata, cleaned)
    trace_meta = build_document_trace_metadata(metadata, cleaned)

    # algorithm 결정은 명시 metadata 또는 파일명/경로 기반 추론만 사용한다.
    # 본문 키워드는 related_topics에는 쓰지만 algorithm override에는 쓰지 않는다.
    raw_algorithm = graph_meta.get("algorithm") or inferred_meta.get("algorithm") or metadata.get("algorithm", "")
    canonical = canonicalize_algorithm(
        raw_algorithm,
        filename=str(metadata.get("filename", "")),
        source=str(metadata.get("source", "")),
        text="",
    )

    base_source_type = _meta_value(doc_meta, ["source_type"], inferred_meta.get("source_type", metadata.get("source_type", "generated")))
    quality_defaults = infer_doc_quality_defaults({"source_type": base_source_type})

    base_meta = {
        **metadata,
        **trace_meta,
        "doc_format": fmt,

        # 핵심 검색/필터링 메타데이터
        "algorithm": canonical["algorithm"],
        "algorithm_key": canonical["algorithm_key"],
        "display_name": canonical["display_name"],
        "aliases": infer_aliases_for_metadata(canonical["algorithm_key"], doc_meta),
        "category": graph_meta.get("category") or inferred_meta.get("category") or metadata.get("category", ""),
        "difficulty": graph_meta.get("difficulty") or inferred_meta.get("difficulty") or metadata.get("difficulty", ""),
        "target_level": _meta_value(doc_meta, ["target_level", "difficulty", "level"], inferred_meta.get("difficulty", metadata.get("target_level", ""))),
        "language": _meta_value(doc_meta, ["language"], metadata.get("language", "")),

        # 데이터 품질/출처 메타데이터
        "source_type": base_source_type,
        "content_type": _meta_value(doc_meta, ["content_type", "type"], inferred_meta.get("content_type", "")),
        "style": _meta_value(doc_meta, ["style"], inferred_meta.get("style", "")),
        "quality_score": _meta_value(doc_meta, ["quality_score"], inferred_meta.get("quality_score", quality_defaults["quality_score"])),
        "verified": _meta_value(doc_meta, ["verified"], inferred_meta.get("verified", quality_defaults["verified"])),
        "use_for_rag": _meta_value(doc_meta, ["use_for_rag"], inferred_meta.get("use_for_rag", quality_defaults["use_for_rag"])),
        "retrieval_priority": _meta_value(doc_meta, ["retrieval_priority"], inferred_meta.get("retrieval_priority", quality_defaults["retrieval_priority"])),

        # GraphRAG 확장용 메타데이터
        "related_topics": merge_json_lists(graph_meta.get("related_topics"), inferred_meta.get("related_topics")),
        "prerequisites": merge_json_lists(graph_meta.get("prerequisites"), inferred_meta.get("prerequisites")),

        # 청킹 판단 근거 메타데이터
        "chunk_size_used": chunk_config["chunk_size"],
        "chunk_overlap_used": chunk_config["overlap"],
        "max_semantic_chunk_size_used": chunk_config["max_semantic_size"],
        "source_text_len": chunk_config["text_len"],
        "source_code_block_count": chunk_config["code_block_count"],
    }

    if show_comparison:
        compare_strategies(cleaned, fmt, metadata)

    if fmt == "C":
        raw_chunks = chunk_recursive(
            cleaned,
            max_size=chunk_config["chunk_size"],
            overlap=chunk_config["overlap"],
            title="recursive_fallback",
            chunk_type="concept",
        )
    else:
        raw_chunks = chunk_semantic(
            cleaned,
            doc_format=fmt,
            max_size=chunk_config["max_semantic_size"],
            overlap=chunk_config["overlap"],
        )

    documents: List[Document] = []
    total = len(raw_chunks)
    for idx, chunk in enumerate(raw_chunks):
        text_chunk = chunk["text"].strip()
        if not text_chunk:
            continue

        section_title = chunk.get("title", "")
        raw_chunk_type = chunk.get("type", "concept")
        chunk_type = normalize_chunk_type(section_title, raw_chunk_type, text_chunk)
        original_doc_id = base_meta.get("original_doc_id", "doc")
        chunk_uid = f"{original_doc_id}_chunk_{idx + 1:04d}"
        chunk_meta = {
            **base_meta,
            # 추적성 메타데이터
            "chunk_id": chunk_uid,
            "chunk_index": idx,
            "chunk_no": idx + 1,
            "chunk_count": total,
            "stable_chunk_key": chunk_uid,

            # 청크 설명 메타데이터
            "chunk_type": chunk_type,
            "section_title": section_title,
            "section_order": idx + 1,
            "parent_section_title": chunk.get("parent_section_title", section_title),
            "strategy": chunk.get("strategy", "adaptive_semantic_recursive" if fmt != "C" else "recursive"),
            "char_count": len(text_chunk),
            "sub_chunk_index": chunk.get("sub_chunk_index", 0),
            "content_type": infer_chunk_content_type(section_title, chunk_type, text_chunk, base_meta.get("content_type", "")),
            "contains_code": contains_code_block(text_chunk),
            "contains_complexity": contains_complexity_text(section_title, text_chunk),
            "style": infer_chunk_style(section_title, text_chunk, base_meta.get("style", "")),
            "for_vector_db": True,
            "for_graph_db": True,
        }
        documents.append(Document(page_content=text_chunk, metadata=chunk_meta))

    return documents
