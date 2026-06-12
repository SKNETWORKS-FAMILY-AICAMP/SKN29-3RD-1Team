import os
import json
from langchain_core.documents import Document
 
from app.rag.loader import load_pdf
from app.rag.splitter import split_documents, split_md_smart
from app.rag.vector_store import get_vector_store


def ingest_pdf(file_path: str):
    docs = load_pdf(file_path)
    chunks = split_documents(docs)
    vector_store = get_vector_store()
    vector_store.add_documents(chunks)
    print(f"saved {len(chunks)} chunks")


# 추가: md 파일 저장 (splitter.py의 semantic 청킹 사용)
def ingest_md(file_path: str):
    with open(file_path, encoding="utf-8") as f:
        text = f.read()
 
    metadata = {
        "source": os.path.basename(file_path),
    }
 
    chunks = split_md_smart(text, metadata)  # splitter.py 거침
 
    vector_store = get_vector_store()
    vector_store.add_documents(chunks)
    print(f"saved {len(chunks)} chunks from {os.path.basename(file_path)}")
    return {"message": "saved", "chunks": len(chunks)}

# 추가: 폴더 전체 md 일괄 저장
def ingest_md_folder(folder_path: str):
    md_files = []
    for root, dirs, files in os.walk(folder_path):
        for f in sorted(files):
            if f.endswith(".md") and "(1)" not in f:
                md_files.append(os.path.join(root, f))
 
    # complete/generated 우선
    priority = [f for f in md_files if "complete" in f.lower() or "generated" in f.lower()]
    targets = priority if priority else md_files
 
    total = 0
    for fp in targets:
        result = ingest_md(fp)
        total += result["chunks"]
 
    print(f"total: {total} chunks from {len(targets)} files")
    return {"message": "saved", "files": len(targets), "chunks": total}

# 추가: 청킹 완료된 JSON 저장 (chunking_pipeline.py 결과물)
def ingest_from_json(json_path: str):
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
 
    documents = []
    for text, meta in zip(data["documents"], data["metadatas"]):
        documents.append(Document(page_content=text, metadata=meta))
 
    vector_store = get_vector_store()
    vector_store.add_documents(documents)
    print(f"saved {len(documents)} chunks from json")
    return {"message": "saved", "chunks": len(documents)}