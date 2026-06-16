from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

load_dotenv()

from app.rag.vector_store import get_vector_store
from app.rag.retrieval_v1 import retrieve_v1


DEFAULT_DATASET_PATH = "evaluation/retrieval_dataset_expanded_50.json"


def normalize_case(case: Dict[str, Any]) -> Dict[str, Any]:
    """Support both old and new evaluation dataset formats.

    Old format:
        {"question": "...", "expected_algorithm_keys": ["..."]}

    New format:
        {"query": "...", "expected": ["..."], "type": "basic"}
    """

    question = case.get("question") or case.get("query")
    expected = case.get("expected_algorithm_keys") or case.get("expected")

    if not question:
        raise ValueError(f"평가 케이스에 question/query가 없습니다: {case}")
    if not expected:
        raise ValueError(f"평가 케이스에 expected_algorithm_keys/expected가 없습니다: {case}")

    return {
        "question": question,
        "expected_algorithm_keys": expected,
        "type": case.get("type", "basic"),
    }


def load_eval_cases(dataset_path: str = DEFAULT_DATASET_PATH) -> List[Dict[str, Any]]:
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(
            f"평가셋 파일을 찾을 수 없습니다: {dataset_path}\n"
            f"현재 위치에서 실행 중인지 확인하세요. 예: python eval_retrieval_v1.py"
        )

    with path.open("r", encoding="utf-8") as f:
        raw_cases = json.load(f)

    if not isinstance(raw_cases, list):
        raise ValueError("평가셋 JSON은 list 형태여야 합니다.")

    return [normalize_case(case) for case in raw_cases]


DATASET_PATH = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATASET_PATH
EVAL_CASES = load_eval_cases(DATASET_PATH)


def get_algorithm_keys_from_plain_docs(docs):
    return [doc.metadata.get("algorithm_key", "") for doc in docs]


def get_algorithm_keys_from_voted_docs(voted_docs):
    return [item["representative_doc"].metadata.get("algorithm_key", "") for item in voted_docs]


def hit_at_k(result_keys, expected_keys, k):
    return any(key in expected_keys for key in result_keys[:k])


def reciprocal_rank(result_keys, expected_keys):
    for i, key in enumerate(result_keys, 1):
        if key in expected_keys:
            return 1.0 / i
    return 0.0


def evaluate_plain_vector():
    db = get_vector_store()
    hit1 = 0
    hit5 = 0
    mrr_sum = 0.0
    rows = []

    for case in EVAL_CASES:
        docs = db.similarity_search(case["question"], k=5)
        keys = get_algorithm_keys_from_plain_docs(docs)
        expected = case["expected_algorithm_keys"]

        h1 = hit_at_k(keys, expected, 1)
        h5 = hit_at_k(keys, expected, 5)
        rr = reciprocal_rank(keys, expected)

        hit1 += int(h1)
        hit5 += int(h5)
        mrr_sum += rr
        rows.append((case["question"], expected, keys, h1, h5, rr))

    n = len(EVAL_CASES)
    return {
        "name": "Plain Vector Search",
        "hit1": hit1 / n,
        "recall5": hit5 / n,
        "mrr": mrr_sum / n,
        "rows": rows,
    }


def evaluate_retrieval_v1():
    hit1 = 0
    hit5 = 0
    mrr_sum = 0.0
    rows = []

    for case in EVAL_CASES:
        result = retrieve_v1(case["question"], search_k=20, final_k=5)
        keys = get_algorithm_keys_from_voted_docs(result["voted_docs"])
        expected = case["expected_algorithm_keys"]

        h1 = hit_at_k(keys, expected, 1)
        h5 = hit_at_k(keys, expected, 5)
        rr = reciprocal_rank(keys, expected)

        hit1 += int(h1)
        hit5 += int(h5)
        mrr_sum += rr
        rows.append((case["question"], expected, keys, h1, h5, rr))

    n = len(EVAL_CASES)
    return {
        "name": "Retrieval V1 Metadata Filter + Voting",
        "hit1": hit1 / n,
        "recall5": hit5 / n,
        "mrr": mrr_sum / n,
        "rows": rows,
    }


def print_report(result):
    print("\n" + "=" * 100)
    print(result["name"])
    print("=" * 100)
    print(f"Hit@1   : {result['hit1']:.3f}")
    print(f"Recall@5: {result['recall5']:.3f}")
    print(f"MRR     : {result['mrr']:.3f}")

    print("\nCase Details")
    print("-" * 100)
    for i, (q, expected, keys, h1, h5, rr) in enumerate(result["rows"], 1):
        print(f"\n[{i}] {q}")
        print(f"  expected : {expected}")
        print(f"  results  : {keys}")
        print(f"  Hit@1={h1} | Hit@5={h5} | RR={rr:.3f}")


def print_failed_cases(result):
    failed = [row for row in result["rows"] if not row[3]]
    print("\n" + "=" * 100)
    print(f"Hit@1 실패 케이스: {len(failed)}개")
    print("=" * 100)

    if not failed:
        print("없음")
        return

    for q, expected, keys, h1, h5, rr in failed:
        print(f"\n질문     : {q}")
        print(f"expected : {expected}")
        print(f"results  : {keys}")
        print(f"Recall@5 : {h5} | RR={rr:.3f}")


if __name__ == "__main__":
    print("\n" + "=" * 100)
    print("Retrieval V1 Evaluation")
    print("=" * 100)
    print(f"Dataset : {DATASET_PATH}")
    print(f"Cases   : {len(EVAL_CASES)}")

    plain = evaluate_plain_vector()
    v1 = evaluate_retrieval_v1()

    print_report(plain)
    print_report(v1)

    print("\n" + "=" * 100)
    print("Comparison")
    print("=" * 100)
    print(f"Hit@1    : {plain['hit1']:.3f} -> {v1['hit1']:.3f}")
    print(f"Recall@5 : {plain['recall5']:.3f} -> {v1['recall5']:.3f}")
    print(f"MRR      : {plain['mrr']:.3f} -> {v1['mrr']:.3f}")

    print_failed_cases(v1)
