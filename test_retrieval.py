from dotenv import load_dotenv
load_dotenv()

from app.rag.vector_store import get_vector_store

db = get_vector_store()

queries = [
    "힙은 언제 사용하나요?",
    "우선순위 큐 구현 방법 알려줘",
    "DFS와 BFS 차이는?",
    "이분탐색은 어떤 조건에서 쓰나요?",
    "DP는 언제 사용하나요?",
    "백트래킹은 DFS와 뭐가 다른가요?",
    "다익스트라 알고리즘은 언제 쓰나요?",
    "시간복잡도 O(log n)은 무슨 뜻인가요?",
]

for q in queries:
    print("\n" + "=" * 80)
    print("Q:", q)

    docs = db.similarity_search(q, k=5)

    for i, doc in enumerate(docs, 1):
        print(f"\n[{i}]")
        print("본문:", doc.page_content[:250].replace("\n", " "))
        print("메타데이터:", doc.metadata)