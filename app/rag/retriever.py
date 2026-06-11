from app.rag.vector_store import get_vector_store


def get_retriever():

    vector_store = get_vector_store()

    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 5
        }
    )