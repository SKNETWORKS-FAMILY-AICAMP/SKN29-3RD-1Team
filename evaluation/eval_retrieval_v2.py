from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv

# OpenAI / httpx 로그가 평가 결과를 가리지 않도록 숨긴다.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

load_dotenv()

from app.rag.vector_store import get_vector_store
from app.rag.retrieval_v1 import retrieve_v1
from app.rag.retrieval_v2 import retrieve_v2


BASIC_DATASET_PATH = "evaluation/retrieval_dataset_expanded_50.json"
HARD_DATASET_PATH = "evaluation/retrieval_hard_dataset_expanded_50.json"

# V2가 해결해야 하는 문법/자연어/복합 의도 케이스.
# 기존 basic/hard 평가셋에 없을 수 있으므로 V2 고도화 효과 확인용으로 함께 둔다.
V2_SPECIAL_CASES = [
    {
        "query": "조건문을 사용하여 두 수를 비교하는 방법에 대해 더 공부할 수 있는 자료는 무엇인가요?",
        "expected": ["conditional_statement", "조건문", "java조건문", "c/cpp조건문", "파이썬조건문"],
        "type": "grammar",
    },
    {
        "query": "if문과 비교 연산자로 값의 크기를 판단하는 법을 알고 싶어",
        "expected": ["conditional_statement", "조건문", "java조건문", "c/cpp조건문", "파이썬조건문"],
        "type": "grammar",
    },
    {
        "query": "두 값이 같은지 다른지 확인하는 조건 분기는 어떻게 작성하나요?",
        "expected": ["conditional_statement", "조건문", "java조건문", "c/cpp조건문", "파이썬조건문"],
        "type": "grammar",
    },
    {
        "query": "파이썬에서 원소 개수를 세는 라이브러리를 더 공부하고 싶어",
        "expected": ["파이썬collections", "hash"],
        "type": "library",
    },
    {
        "query": "순열과 조합을 파이썬에서 쉽게 만드는 방법이 궁금해",
        "expected": ["파이썬itertools", "순열,조합", "backtracking"],
        "type": "library",
    },
    {
        "query": "정답 후보를 만들다가 조건에 안 맞으면 바로 버리는 풀이가 궁금해",
        "expected": ["backtracking"],
        "type": "natural_language",
    },
    {
        "query": "같은 그룹인지 확인하고 두 그룹을 합치는 자료구조를 공부하고 싶어",
        "expected": ["union_find"],
        "type": "natural_language",
    },
    {
        "query": "자동완성처럼 접두사로 문자열을 빠르게 찾는 구조는 무엇인가요?",
        "expected": ["trie"],
        "type": "natural_language",
    },
    {
        "query": "구간 합을 여러 번 구하고 중간 값도 바뀌는 문제를 공부하고 싶어",
        "expected": ["segment_tree"],
        "type": "natural_language",
    },
    {
        "query": "가장 급한 작업을 먼저 처리하는 자료구조는 무엇을 공부해야 하나요?",
        "expected": ["priority_queue", "heap"],
        "type": "natural_language",
    },
]

# 평가 시 같은 개념으로 인정할 key 묶음.
EQUIVALENT_KEYS = {
    "conditional_statement": {"conditional_statement", "조건문", "if_statement", "python_condition", "파이썬조건문", "java조건문", "c/cpp조건문", "c조건문", "cpp조건문"},
    "priority_queue": {"priority_queue", "heap"},
    "heap": {"heap", "priority_queue"},
    "파이썬collections": {"파이썬collections", "hash", "deque"},
    "파이썬itertools": {"파이썬itertools", "순열,조합", "backtracking"},
    "순열,조합": {"순열,조합", "backtracking", "파이썬itertools"},
}


def expand_expected_keys(expected: List[str]) -> set[str]:
    expanded: set[str] = set()
    for key in expected:
        key = str(key)
        expanded.add(key)
        expanded.update(EQUIVALENT_KEYS.get(key, set()))
    return expanded


def normalize_case(case: Dict[str, Any], default_type: str) -> Dict[str, Any]:
    question = case.get("question") or case.get("query")
    expected = case.get("expected_algorithm_keys") or case.get("expected")

    if not question:
        raise ValueError(f"평가 케이스에 question/query가 없습니다: {case}")
    if not expected:
        raise ValueError(f"평가 케이스에 expected_algorithm_keys/expected가 없습니다: {case}")

    if isinstance(expected, str):
        expected = [expected]

    return {
        "question": question,
        "expected_algorithm_keys": [str(x) for x in expected],
        "type": case.get("type", default_type),
    }


def load_eval_cases(path: str, default_type: str) -> List[Dict[str, Any]]:
    dataset_path = Path(path)
    if not dataset_path.exists():
        return []

    with dataset_path.open("r", encoding="utf-8") as f:
        raw_cases = json.load(f)

    if not isinstance(raw_cases, list):
        raise ValueError(f"{path} JSON은 list 형태여야 합니다.")

    return [normalize_case(case, default_type) for case in raw_cases]


def get_keys_from_plain_docs(docs: List[Any]) -> List[str]:
    return [str((doc.metadata or {}).get("algorithm_key", "")) for doc in docs]


def get_keys_from_voted_docs(voted_docs: List[Dict[str, Any]]) -> List[str]:
    return [
        str((item["representative_doc"].metadata or {}).get("algorithm_key", ""))
        for item in voted_docs
    ]


def hit_at_k(result_keys: List[str], expected_keys: set[str], k: int) -> bool:
    return any(key in expected_keys for key in result_keys[:k])


def reciprocal_rank(result_keys: List[str], expected_keys: set[str]) -> float:
    for i, key in enumerate(result_keys, 1):
        if key in expected_keys:
            return 1.0 / i
    return 0.0


def evaluate_plain(cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    db = get_vector_store()
    rows = []
    hit1 = 0
    hit5 = 0
    mrr_sum = 0.0

    for case in cases:
        docs = db.similarity_search(case["question"], k=5)
        keys = get_keys_from_plain_docs(docs)
        expected = expand_expected_keys(case["expected_algorithm_keys"])

        h1 = hit_at_k(keys, expected, 1)
        h5 = hit_at_k(keys, expected, 5)
        rr = reciprocal_rank(keys, expected)

        hit1 += int(h1)
        hit5 += int(h5)
        mrr_sum += rr
        rows.append({
            "question": case["question"],
            "expected": sorted(expected),
            "keys": keys,
            "hit1": h1,
            "recall5": h5,
            "rr": rr,
        })

    n = max(len(cases), 1)
    return {"hit1": hit1 / n, "recall5": hit5 / n, "mrr": mrr_sum / n, "rows": rows}


def evaluate_v1(cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    rows = []
    hit1 = 0
    hit5 = 0
    mrr_sum = 0.0

    for case in cases:
        result = retrieve_v1(case["question"], search_k=20, final_k=5)
        keys = get_keys_from_voted_docs(result.get("voted_docs") or [])
        expected = expand_expected_keys(case["expected_algorithm_keys"])

        h1 = hit_at_k(keys, expected, 1)
        h5 = hit_at_k(keys, expected, 5)
        rr = reciprocal_rank(keys, expected)

        hit1 += int(h1)
        hit5 += int(h5)
        mrr_sum += rr
        rows.append({
            "question": case["question"],
            "expected": sorted(expected),
            "keys": keys,
            "hit1": h1,
            "recall5": h5,
            "rr": rr,
        })

    n = max(len(cases), 1)
    return {"hit1": hit1 / n, "recall5": hit5 / n, "mrr": mrr_sum / n, "rows": rows}


def evaluate_v2(cases: List[Dict[str, Any]], use_llm: bool = True, use_hyde: bool = True) -> Dict[str, Any]:
    rows = []
    hit1 = 0
    hit5 = 0
    mrr_sum = 0.0
    full_doc_success = 0

    for case in cases:
        result = retrieve_v2(
            case["question"],
            search_k=20,
            final_k=5,
            use_llm_rewrite=use_llm,
            use_hyde=use_hyde,
            return_full_documents=True,
        )
        keys = get_keys_from_voted_docs(result.get("voted_docs") or [])
        expected = expand_expected_keys(case["expected_algorithm_keys"])

        h1 = hit_at_k(keys, expected, 1)
        h5 = hit_at_k(keys, expected, 5)
        rr = reciprocal_rank(keys, expected)
        has_full_doc = bool(result.get("full_documents"))

        hit1 += int(h1)
        hit5 += int(h5)
        mrr_sum += rr
        full_doc_success += int(has_full_doc)

        rows.append({
            "question": case["question"],
            "expected": sorted(expected),
            "keys": keys,
            "hit1": h1,
            "recall5": h5,
            "rr": rr,
            "full_documents": len(result.get("full_documents") or []),
            "search_query": result.get("search_query"),
            "hyde_used": bool(result.get("hyde_document")),
            "rewrite_error": (result.get("rewrite_result") or {}).get("error"),
        })

    n = max(len(cases), 1)
    return {
        "hit1": hit1 / n,
        "recall5": hit5 / n,
        "mrr": mrr_sum / n,
        "full_doc_rate": full_doc_success / n,
        "rows": rows,
    }


def print_metric_block(title: str, plain: Dict[str, Any] | None, v1: Dict[str, Any], v2: Dict[str, Any]) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)

    if plain is not None:
        print("Plain Vector Search")
        print(f"Hit@1   : {plain['hit1']:.3f}")
        print(f"Recall@5: {plain['recall5']:.3f}")
        print(f"MRR     : {plain['mrr']:.3f}")
        print()

    print("Retrieval V1")
    print(f"Hit@1   : {v1['hit1']:.3f}")
    print(f"Recall@5: {v1['recall5']:.3f}")
    print(f"MRR     : {v1['mrr']:.3f}")
    print()

    print("Retrieval V2")
    print(f"Hit@1        : {v2['hit1']:.3f}")
    print(f"Recall@5     : {v2['recall5']:.3f}")
    print(f"MRR          : {v2['mrr']:.3f}")
    print(f"FullDoc Rate : {v2['full_doc_rate']:.3f}")
    print()

    print("V1 -> V2 Improvement")
    print(f"Hit@1   : {v1['hit1']:.3f} -> {v2['hit1']:.3f} ({v2['hit1'] - v1['hit1']:+.3f})")
    print(f"Recall@5: {v1['recall5']:.3f} -> {v2['recall5']:.3f} ({v2['recall5'] - v1['recall5']:+.3f})")
    print(f"MRR     : {v1['mrr']:.3f} -> {v2['mrr']:.3f} ({v2['mrr'] - v1['mrr']:+.3f})")


def print_failures(label: str, result: Dict[str, Any], limit: int = 5) -> None:
    failures = [row for row in result["rows"] if not row["recall5"]]
    if not failures:
        print(f"\n{label}: Recall@5 실패 케이스 없음")
        return

    print(f"\n{label}: Recall@5 실패 케이스 {len(failures)}개")
    for row in failures[:limit]:
        print("-" * 100)
        print(f"Q        : {row['question']}")
        print(f"Expected : {row['expected']}")
        print(f"Pred     : {row['keys']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Retrieval V1 vs V2")
    parser.add_argument("--basic", default=BASIC_DATASET_PATH)
    parser.add_argument("--hard", default=HARD_DATASET_PATH)
    parser.add_argument("--no-plain", action="store_true", help="Plain vector baseline 생략")
    parser.add_argument("--no-llm", action="store_true", help="V2 LLM rewrite 비활성화")
    parser.add_argument("--no-hyde", action="store_true", help="V2 HyDE 비활성화")
    parser.add_argument("--show-failures", action="store_true", help="실패 케이스 출력")
    args = parser.parse_args()

    use_llm = not args.no_llm
    use_hyde = not args.no_hyde

    basic_cases = load_eval_cases(args.basic, "basic")
    hard_cases = load_eval_cases(args.hard, "hard")
    special_cases = [normalize_case(case, "v2_special") for case in V2_SPECIAL_CASES]

    print("\n" + "=" * 100)
    print("Retrieval V1 vs V2 Evaluation")
    print("=" * 100)
    print(f"V2 LLM Rewrite : {use_llm}")
    print(f"V2 HyDE        : {use_hyde}")

    if basic_cases:
        plain_basic = None if args.no_plain else evaluate_plain(basic_cases)
        v1_basic = evaluate_v1(basic_cases)
        v2_basic = evaluate_v2(basic_cases, use_llm=use_llm, use_hyde=use_hyde)
        print_metric_block(f"Basic Evaluation ({len(basic_cases)} cases)", plain_basic, v1_basic, v2_basic)
        if args.show_failures:
            print_failures("V2 Basic", v2_basic)

    if hard_cases:
        plain_hard = None if args.no_plain else evaluate_plain(hard_cases)
        v1_hard = evaluate_v1(hard_cases)
        v2_hard = evaluate_v2(hard_cases, use_llm=use_llm, use_hyde=use_hyde)
        print_metric_block(f"Hard Evaluation ({len(hard_cases)} cases)", plain_hard, v1_hard, v2_hard)
        if args.show_failures:
            print_failures("V2 Hard", v2_hard)

    plain_special = None if args.no_plain else evaluate_plain(special_cases)
    v1_special = evaluate_v1(special_cases)
    v2_special = evaluate_v2(special_cases, use_llm=use_llm, use_hyde=use_hyde)
    print_metric_block(f"V2 Special Evaluation ({len(special_cases)} cases)", plain_special, v1_special, v2_special)
    if args.show_failures:
        print_failures("V2 Special", v2_special)

    print("\n" + "=" * 100)
    print("FINAL SUMMARY")
    print("=" * 100)
    print("V2 검증 포인트:")
    print("- LLM Query Rewrite와 HyDE가 실제 검색 쿼리와 가상 문서를 생성하는지 확인")
    print("- 문법/자연어 질의에서 V1 대비 개선되는지 확인")
    print("- full_documents 반환률이 1.000에 가까운지 확인")
    print("- Basic/Hard에서 V1 성능을 크게 해치지 않는지 확인")
    print("=" * 100)


if __name__ == "__main__":
    main()
