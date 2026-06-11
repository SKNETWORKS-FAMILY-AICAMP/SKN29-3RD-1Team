from fastapi import APIRouter
from pydantic import BaseModel

from langchain_core.documents import Document

from app.rag.vector_store import get_vector_store


router = APIRouter(
    prefix="/rag",
    tags=["rag"]
)


class SaveTextRequest(BaseModel):
    text: str


@router.post("/documents")
def save_document(request: SaveTextRequest):

    document = Document(
        page_content=request.text,
        metadata={
            "source": "api"
        }
    )

    vector_store = get_vector_store()

    vector_store.add_documents([document])

    return {
        "message": "saved",
        "length": len(request.text)
    }