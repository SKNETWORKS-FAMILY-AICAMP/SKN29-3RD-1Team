from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from app.rag.vector_store import get_vector_store
from app.rag.retrieval_v1 import (
    RetrievalPlan,
    build_retrieval_plan,
    safe_similarity_search,
    merge_docs,
    vote_documents,
    candidate_priority_map,
)
from app.rag.prompts.retrieval_rewrite_prompt import rewrite_query_with_llm


# V2 전용 key 확장.
# V1의 FILTER_KEY_EXPANSIONS가 최신이 아니어도 V2는 이 매핑을 사용한다.
V2_FILTER_KEY_EXPANSIONS: Dict[str, List[str]] = {
    "conditional_statement": [
        "conditional_statement",
        "조건문",
        "if_statement",
        "python_condition",
        "파이썬조건문",
        "java조건문",
        "c/cpp조건문",
        "c조건문",
        "cpp조건문",
    ],
    "time_complexity": [
        "time_complexity",
        "c/cpp시간복잡도",
        "파이썬시간복잡도",
        "java시간복잡도",
    ],
}


def expand_filter_keys_v2(keys: List[str]) -> List[str]:
    expanded: List[str] = []
    for key in keys:
        expanded.extend(V2_FILTER_KEY_EXPANSIONS.get(key, [key]))
    return list(dict.fromkeys(expanded))


def build_chroma_filter_v2(plan: RetrievalPlan, min_score: float = 2.8, max_keys: int = 3) -> Optional[Dict[str, Any]]:
    keys = [
        c["algorithm_key"]
        for c in plan.algorithm_candidates
        if float(c.get("score", 0.0)) >= min_score
    ]

    if not keys:
        return None

    filter_keys = expand_filter_keys_v2(keys[:max_keys])
    return {"algorithm_key": {"$in": filter_keys}}


def get_default_rewrite_llm() -> Optional[Any]:
    try:
        from app.llm.openai_model import get_openai_model
        return get_openai_model()
    except Exception:
        return None


def _get_chroma_collection(vector_store: Any):
    if hasattr(vector_store, "_collection"):
        return vector_store._collection
    if hasattr(vector_store, "collection"):
        return vector_store.collection
    return None


def _doc_identity(meta: Dict[str, Any]) -> Tuple[str, str]:
    doc_id = str(meta.get("doc_id") or "").strip()
    source_file = str(meta.get("source_file") or meta.get("filename") or "").strip()
    return doc_id, source_file


def _safe_int(value: Any, default: int = 10**9) -> int:
    try:
        if value is None or value == "":
            return default
        return int(value)
    except Exception:
        return default


def _chunk_sort_key(chunk: Dict[str, Any]) -> Tuple[int, int, str]:
    meta = chunk.get("metadata", {}) or {}
    return (
        _safe_int(meta.get("chunk_index")),
        _safe_int(meta.get("section_order")),
        str(meta.get("section_title") or ""),
    )


def get_full_document_by_metadata(
    vector_store: Any,
    metadata: Dict[str, Any],
    max_chunks: int = 500,
) -> Optional[Dict[str, Any]]:
    """대표 chunk metadata 기준으로 원본 문서 전체를 복원한다.

    source_file 기준을 우선한다.
    기존 chunk doc_id가 section 단위로 달라도 파일 전체 복원이 가능하다.
    """
    collection = _get_chroma_collection(vector_store)
    if collection is None:
        return None

    doc_id, source_file = _doc_identity(metadata)
    if not doc_id and not source_file:
        return None

    where_candidates = []
    if source_file:
        where_candidates.append({"source_file": source_file})
        where_candidates.append({"filename": source_file})
    if doc_id:
        where_candidates.append({"doc_id": doc_id})

    data = None
    for where in where_candidates:
        try:
            candidate = collection.get(
                where=where,
                include=["documents", "metadatas"],
                limit=max_chunks,
            )
            if candidate and candidate.get("documents"):
                data = candidate
                break
        except Exception:
            continue

    if not data:
        return None

    documents = data.get("documents") or []
    metadatas = data.get("metadatas") or []

    chunks: List[Dict[str, Any]] = []
    for idx, (content, meta) in enumerate(zip(documents, metadatas)):
        meta = meta or {}
        chunks.append(
            {
                "chunk_index": meta.get("chunk_index", idx),
                "section_order": meta.get("section_order"),
                "section_title": meta.get("section_title"),
                "algorithm_key": meta.get("algorithm_key"),
                "content": content,
                "metadata": meta,
            }
        )

    chunks.sort(key=_chunk_sort_key)

    return {
        "doc_id": source_file or doc_id,
        "source_file": source_file,
        "chunk_count": len(chunks),
        "full_text": "\n\n".join(str(chunk.get("content") or "") for chunk in chunks).strip(),
        "chunks": chunks,
    }


def build_full_documents(
    vector_store: Any,
    voted_docs: List[Dict[str, Any]],
    max_documents: int = 5,
) -> List[Dict[str, Any]]:
    """voted_docs의 대표 chunk를 원본 문서 전체로 확장한다.

    중복 제거는 source_file 우선.
    """
    full_documents: List[Dict[str, Any]] = []
    seen = set()

    for item in voted_docs:
        doc = item.get("representative_doc")
        if doc is None:
            continue

        meta = getattr(doc, "metadata", {}) or {}
        doc_id, source_file = _doc_identity(meta)
        identity = source_file or doc_id
        if not identity or identity in seen:
            continue

        full_doc = get_full_document_by_metadata(vector_store, meta)
        if not full_doc:
            continue

        full_doc["vote_score"] = item.get("score")
        full_doc["first_rank"] = item.get("first_rank")
        full_doc["representative_section_title"] = meta.get("section_title")

        full_documents.append(full_doc)
        seen.add(identity)

        if len(full_documents) >= max_documents:
            break

    return full_documents


def _merge_algorithm_candidates(
    rule_plan: RetrievalPlan,
    rewrite_result: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}

    for candidate in rule_plan.algorithm_candidates or []:
        key = candidate.get("algorithm_key")
        if not key:
            continue
        merged[str(key)] = {
            "algorithm_key": str(key),
            "score": float(candidate.get("score", 1.0)),
            "reason": candidate.get("reason", "rule"),
        }

    if rewrite_result:
        for idx, key in enumerate(rewrite_result.get("algorithm_candidates") or []):
            if not key:
                continue
            key = str(key)
            score = 3.7 - idx * 0.2
            if key not in merged or score > float(merged[key].get("score", 0.0)):
                merged[key] = {
                    "algorithm_key": key,
                    "score": round(score, 3),
                    "reason": "llm_rewrite",
                }

    return sorted(merged.values(), key=lambda x: x["score"], reverse=True)[:4]


def _filter_docs_by_algorithm_candidates(
    docs: List[Any],
    plan: RetrievalPlan,
    min_score: float = 2.8,
) -> List[Any]:
    candidate_keys = [
        str(c.get("algorithm_key"))
        for c in (plan.algorithm_candidates or [])
        if c.get("algorithm_key") and float(c.get("score", 0.0)) >= min_score
    ]
    if not candidate_keys:
        return docs

    allowed = set(expand_filter_keys_v2(candidate_keys))
    filtered = [
        doc for doc in docs
        if str((getattr(doc, "metadata", {}) or {}).get("algorithm_key", "")) in allowed
    ]

    return filtered if len(filtered) >= 1 else docs


def candidate_priority_map_v2(plan: RetrievalPlan) -> Dict[str, float]:
    """V2 확장 key까지 priority를 부여한다."""
    priorities: Dict[str, float] = {}
    for candidate in plan.algorithm_candidates:
        key = str(candidate.get("algorithm_key") or "")
        if not key:
            continue
        score = float(candidate.get("score", 1.0))
        for expanded_key in expand_filter_keys_v2([key]):
            priorities[expanded_key] = max(priorities.get(expanded_key, 0.0), score)
    return priorities


def retrieve_v2(
    query: str,
    llm: Optional[Any] = None,
    search_k: int = 20,
    final_k: int = 5,
    use_mmr: bool = True,
    use_rule_router: bool = True,
    use_llm_rewrite: bool = True,
    use_hyde: bool = True,
    return_full_documents: bool = True,
) -> Dict[str, Any]:
    """Retrieval V2.

    - Rule Router: 확실한 키워드는 빠르게 처리
    - LLM Query Rewrite: 애매한 질의 보완
    - HyDE: 가상 학습 문서 기반 검색
    - Vector Search + Voting: V1 로직 재사용
    - Full Document Reconstruction: 원본 문서 전체 반환
    """
    vector_store = get_vector_store()

    rule_plan = build_retrieval_plan(query) if use_rule_router else RetrievalPlan(
        need_rag=True,
        intent="concept_explanation",
        algorithm_candidates=[],
        target_level="beginner",
    )

    rewrite_result: Optional[Dict[str, Any]] = None
    if use_llm_rewrite and llm is None:
        llm = get_default_rewrite_llm()

    if use_llm_rewrite and llm is not None:
        try:
            rewrite_result = rewrite_query_with_llm(llm, query)
        except Exception as e:
            rewrite_result = {
                "algorithm_candidates": [],
                "search_query": query,
                "hyde_document": query,
                "error": str(e),
            }

    merged_candidates = _merge_algorithm_candidates(rule_plan, rewrite_result)

    plan = RetrievalPlan(
        need_rag=rule_plan.need_rag,
        intent=rule_plan.intent,
        algorithm_candidates=merged_candidates,
        target_level=rule_plan.target_level,
    )

    search_query = query
    if rewrite_result and rewrite_result.get("search_query"):
        search_query = rewrite_result["search_query"]

    hyde_document = None
    if use_hyde and rewrite_result and rewrite_result.get("hyde_document"):
        hyde_document = rewrite_result["hyde_document"]

    filter_used = build_chroma_filter_v2(plan)
    plan.filter_used = filter_used
    plan.expanded_query = search_query

    filtered_docs: List[Any] = []
    if filter_used:
        filtered_docs = safe_similarity_search(
            vector_store,
            query=search_query,
            k=max(final_k, search_k // 2),
            filter_used=filter_used,
            use_mmr=use_mmr,
        )

    query_docs = safe_similarity_search(
        vector_store,
        query=search_query,
        k=search_k,
        filter_used=None,
        use_mmr=use_mmr,
    )

    hyde_docs: List[Any] = []
    if hyde_document:
        hyde_docs = safe_similarity_search(
            vector_store,
            query=hyde_document,
            k=search_k,
            filter_used=filter_used,
            use_mmr=use_mmr,
        )

    raw_docs = merge_docs(filtered_docs, query_docs, hyde_docs)
    raw_docs = _filter_docs_by_algorithm_candidates(raw_docs, plan)

    priorities = candidate_priority_map_v2(plan)
    voted_docs = vote_documents(
        raw_docs,
        final_k=final_k,
        candidate_priorities=priorities,
    )

    full_documents = build_full_documents(
        vector_store,
        voted_docs,
        max_documents=final_k,
    ) if return_full_documents else []

    return {
        "query": query,
        "version": "retrieval_v2",
        "rule_plan": rule_plan,
        "rewrite_result": rewrite_result,
        "plan": plan,
        "filter_used": filter_used,
        "search_query": search_query,
        "hyde_document": hyde_document,
        "raw_docs": raw_docs,
        "voted_docs": voted_docs,
        "full_documents": full_documents,
        "search_trace": [
            {"stage": "filtered", "query": search_query, "filter": filter_used, "count": len(filtered_docs)},
            {"stage": "query_search", "query": search_query, "filter": None, "count": len(query_docs)},
            {"stage": "hyde_search", "query": hyde_document, "filter": filter_used, "count": len(hyde_docs)} if hyde_document else None,
        ],
        "flags": {
            "use_rule_router": use_rule_router,
            "use_llm_rewrite": use_llm_rewrite,
            "use_hyde": use_hyde,
            "use_mmr": use_mmr,
            "return_full_documents": return_full_documents,
        },
    }


def print_retrieval_v2_result(result: Dict[str, Any]) -> None:
    print("\n" + "=" * 90)
    print(f"Retrieval V2 Query: {result['query']}")
    print("-" * 90)
    print(f"search_query : {result.get('search_query')}")
    print(f"filter_used  : {result.get('filter_used')}")
    print(f"candidates   : {result['plan'].algorithm_candidates}")
    print(f"hyde_document: {str(result.get('hyde_document') or '')[:300]}")

    rewrite_result = result.get("rewrite_result")
    if rewrite_result:
        print(f"rewrite_result: {rewrite_result}")

    print("\nVoted Docs")
    print("-" * 90)
    for i, item in enumerate(result.get("voted_docs") or [], 1):
        doc = item["representative_doc"]
        meta = doc.metadata
        print(
            f"[{i}] score={item['score']} | "
            f"algorithm_key={meta.get('algorithm_key')} | "
            f"source={meta.get('source_file')} | "
            f"section={meta.get('section_title')}"
        )

    print("\nFull Documents")
    print("-" * 90)
    for i, doc in enumerate(result.get("full_documents") or [], 1):
        print(
            f"[{i}] doc_id={doc.get('doc_id')} | "
            f"chunks={doc.get('chunk_count')} | "
            f"source={doc.get('source_file')}"
        )


if __name__ == "__main__":
    result = retrieve_v2(
        query="조건문을 사용하여 두 수를 비교하는 방법에 대해 더 공부할 수 있는 자료는 무엇인가요?",
        use_llm_rewrite=True,
        use_hyde=True,
    )
    print_retrieval_v2_result(result)
