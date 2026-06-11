from langchain_chroma import Chroma

from app.rag.embeddings import get_embeddings


PERSIST_DIR = "./chroma_db"


def get_vector_store():

    return Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=get_embeddings(),
    )