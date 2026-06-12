
import os
import sys
import json
from langchain_core.documents import Document
from dotenv import load_dotenv
load_dotenv()

# 프로젝트 경로 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.rag.splitter import split_md_smart
from app.rag.vector_store import get_vector_store


def run(folder_path):
    # 1. md 파일 재귀 수집
    md_files = []
    for root, dirs, files in os.walk(folder_path):
        for f in sorted(files):
            if f.endswith(".md") and "(1)" not in f:
                md_files.append(os.path.join(root, f))

    # complete/generated 우선
    priority = [f for f in md_files if "complete" in f.lower() or "generated" in f.lower()]
    targets = priority if priority else md_files

    if not targets:
        print(f"md 파일 없음: {folder_path}")
        return

    print(f"폴더: {folder_path}")
    print(f"전체 md: {len(md_files)}개 | 청킹 대상: {len(targets)}개\n")

    # 2. 각 파일 청킹 + VectorDB 저장
    vector_store = get_vector_store()
    total_chunks = 0

    for filepath in targets:
        with open(filepath, encoding="utf-8") as f:
            text = f.read()

        metadata = {
            "source": os.path.relpath(filepath, folder_path),
            "filename": os.path.basename(filepath),
        }

        # splitter.py의 split_md_smart 사용 (3전략 비교 → 최적 선택)
        chunks = split_md_smart(text, metadata)

        if chunks:
            vector_store.add_documents(chunks)
            total_chunks += len(chunks)
            print(f"  {os.path.basename(filepath):<50} -> {len(chunks)}개 청크")

    print(f"\n완료: {total_chunks}개 청크 -> chroma_db/ 저장됨")
    print(f"\n검색 테스트:")
    print(f"  from app.rag.retriever import get_retriever")
    print(f"  retriever = get_retriever()")
    print(f'  results = retriever.invoke("힙에서 삽입은 어떻게 동작해?")')
    print(f"  print(results)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('사용법: python run_ingest.py "C:\\skn29\\3차 프로젝트\\자료"')
        sys.exit(1)

    run(sys.argv[1])
