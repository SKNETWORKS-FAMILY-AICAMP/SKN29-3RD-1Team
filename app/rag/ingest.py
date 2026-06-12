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


def ingest_md(file_path: str):
    with open(file_path, encoding="utf-8") as f:
        text = f.read()
    metadata = {"source": os.path.basename(file_path)}
    chunks = split_md_smart(text, metadata)
    vector_store = get_vector_store()
    vector_store.add_documents(chunks)
    print(f"saved {len(chunks)} chunks from {os.path.basename(file_path)}")
    return {"message": "saved", "chunks": len(chunks)}


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