from app.rag.loader import load_pdf
from app.rag.splitter import split_documents
from app.rag.vector_store import get_vector_store


def ingest_pdf(file_path: str):
    docs = load_pdf(file_path)
    chunks = split_documents(docs)
    vector_store = get_vector_store()
    vector_store.add_documents(chunks)
    print(f"saved {len(chunks)} chunks")