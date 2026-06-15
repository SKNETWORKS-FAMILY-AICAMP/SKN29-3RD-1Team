from dotenv import load_dotenv
load_dotenv()
from collections import Counter, defaultdict
from app.rag.vector_store import get_vector_store


def main():
    db = get_vector_store()

    print("=" * 100)
    print("Algorithm Key Distribution")
    print("=" * 100)

    try:
        data = db.get()
    except Exception as e:
        print(f"DB 조회 실패: {e}")
        return

    metadatas = data.get("metadatas", [])

    chunk_counter = Counter()
    doc_counter = defaultdict(set)

    for meta in metadatas:
        if not meta:
            continue

        algorithm_key = meta.get("algorithm_key", "UNKNOWN")

        chunk_counter[algorithm_key] += 1

        source_file = meta.get("source_file", "")
        if source_file:
            doc_counter[algorithm_key].add(source_file)

    print(f"\n총 청크 수 : {len(metadatas)}")
    print(f"총 알고리즘 수 : {len(chunk_counter)}\n")

    print(f"{'algorithm_key':30} {'chunks':10} {'docs':10}")
    print("-" * 100)

    for algorithm_key, chunk_count in chunk_counter.most_common():
        doc_count = len(doc_counter[algorithm_key])

        print(
            f"{algorithm_key:30}"
            f"{chunk_count:<10}"
            f"{doc_count:<10}"
        )

    print("\n" + "=" * 100)
    print("중요 알고리즘 확인")
    print("=" * 100)

    targets = [
        "dijkstra",
        "priority_queue",
        "heap",
        "binary_search",
        "dfs",
        "bfs",
        "dp",
        "backtracking",
        "greedy",
        "stack",
        "queue",
    ]

    for target in targets:
        print(
            f"{target:20}"
            f"chunks={chunk_counter[target]:<5}"
            f"docs={len(doc_counter[target])}"
        )


if __name__ == "__main__":
    main()