"""
TF-IDF Baseline 평가 스크립트

목적
- 평가표의 '희소 표현(BoW, TF-IDF)' 항목 대응
- 현재 Retrieval V1 성능에는 영향을 주지 않는 비교용 baseline
- ChromaDB에 저장된 문서/메타데이터를 읽어서 TF-IDF 검색 성능을 측정한다.

실행
    python evaluation/tfidf_baseline.py
    python evaluation/tfidf_baseline.py evaluation/retrieval_dataset_expanded_50.json
    python evaluation/tfidf_baseline.py evaluation/retrieval_hard_dataset_expanded_50.json
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

try:
    import chromadb
except Exception as exc:  # pragma: no cover
    raise RuntimeError("chromadb가 필요합니다. pip install chromadb 후 실행하세요.") from exc

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.rag.konlpy_preprocessing import preprocess_for_tfidf


DEFAULT_CHROMA_PATH = "chroma_db"
DEFAULT_DATASET_PATH = "evaluation/retrieval_dataset_expanded_50.json"


def normalize_case(case: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "question": case.get("question") or case.get("query") or "",
        "expected_algorithm_keys": case.get("expected_algorithm_keys") or case.get("expected") or [],
    }


def load_dataset(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [normalize_case(c) for c in raw]


def get_largest_collection(client: Any):
    collections = client.list_collections()
    if not collections:
        raise ValueError("ChromaDB collection이 없습니다. 먼저 run_ingest.py를 실행하세요.")

    infos = []
    for col in collections:
        name = col.name if hasattr(col, "name") else str(col)
        collection = client.get_collection(name)
        infos.append((name, collection.count()))

    infos.sort(key=lambda x: x[1], reverse=True)
    return client.get_collection(infos[0][0]), infos[0]


def load_chroma_documents(chroma_path: str = DEFAULT_CHROMA_PATH) -> Tuple[List[str], List[Dict[str, Any]]]:
    client = chromadb.PersistentClient(path=chroma_path)
    collection, (collection_name, count) = get_largest_collection(client)
    print(f"Collection: {collection_name} ({count} chunks)")

    results = collection.get(include=["documents", "metadatas"])
    documents = results.get("documents") or []
    metadatas = results.get("metadatas") or []

    if not documents:
        raise ValueError("ChromaDB에서 documents를 읽지 못했습니다.")

    return documents, metadatas


def metadata_algorithm_key(metadata: Dict[str, Any]) -> str:
    value = metadata.get("algorithm_key") or metadata.get("algorithm") or ""
    return str(value).strip()


def evaluate_tfidf(
    documents: List[str],
    metadatas: List[Dict[str, Any]],
    dataset: List[Dict[str, Any]],
    top_k: int = 5,
) -> Dict[str, float]:
    processed_docs = [preprocess_for_tfidf(doc) for doc in documents]

    vectorizer = TfidfVectorizer(
        tokenizer=str.split,
        preprocessor=None,
        token_pattern=None,
        lowercase=False,
        min_df=1,
        ngram_range=(1, 2),
    )
    doc_matrix = vectorizer.fit_transform(processed_docs)

    hit1 = 0
    recall5 = 0
    reciprocal_sum = 0.0

    print("\n" + "=" * 100)
    print("TF-IDF Baseline Evaluation")
    print("=" * 100)
    print(f"Cases: {len(dataset)}")

    for idx, case in enumerate(dataset, start=1):
        question = case["question"]
        expected = set(case["expected_algorithm_keys"])

        query_vec = vectorizer.transform([preprocess_for_tfidf(question)])
        scores = cosine_similarity(query_vec, doc_matrix).ravel()
        top_indices = scores.argsort()[::-1][:top_k]

        result_keys = [metadata_algorithm_key(metadatas[i]) for i in top_indices]

        is_hit1 = bool(result_keys and result_keys[0] in expected)
        is_hit5 = any(key in expected for key in result_keys)
        rr = 0.0
        for rank, key in enumerate(result_keys, start=1):
            if key in expected:
                rr = 1.0 / rank
                break

        hit1 += int(is_hit1)
        recall5 += int(is_hit5)
        reciprocal_sum += rr

        print(f"\n[{idx}] {question}")
        print(f"  expected : {list(expected)}")
        print(f"  results  : {result_keys}")
        print(f"  Hit@1={is_hit1} | Hit@5={is_hit5} | RR={rr:.3f}")

    n = len(dataset) or 1
    metrics = {
        "Hit@1": hit1 / n,
        "Recall@5": recall5 / n,
        "MRR": reciprocal_sum / n,
    }

    print("\n" + "=" * 100)
    print("TF-IDF Baseline Result")
    print("=" * 100)
    print(f"Hit@1   : {metrics['Hit@1']:.3f}")
    print(f"Recall@5: {metrics['Recall@5']:.3f}")
    print(f"MRR     : {metrics['MRR']:.3f}")

    return metrics


def main() -> None:
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATASET_PATH
    if not Path(dataset_path).exists():
        raise FileNotFoundError(f"평가셋을 찾을 수 없습니다: {dataset_path}")

    dataset = load_dataset(dataset_path)
    documents, metadatas = load_chroma_documents(DEFAULT_CHROMA_PATH)
    evaluate_tfidf(documents, metadatas, dataset)


if __name__ == "__main__":
    main()
