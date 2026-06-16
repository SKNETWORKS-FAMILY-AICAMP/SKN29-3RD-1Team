from langchain_chroma import Chroma

from app.config import CHROMA_PERSIST_DIR
from app.rag.embeddings import get_embeddings


def get_vector_store():
    return Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=get_embeddings(),
    )
