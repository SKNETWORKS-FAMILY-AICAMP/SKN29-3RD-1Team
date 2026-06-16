import app.config  # noqa: F401
from fastapi import FastAPI
from app.api import health_router, rag_router, solve_router


app = FastAPI(
    title="SKN29 Algorithm RAG LLM API",
    description="RAG 검색 자료를 활용해 알고리즘 문제 풀이를 생성하는 LLM API",
    version="1.0.0",
)

app.include_router(health_router.router)
app.include_router(rag_router.router)
app.include_router(solve_router.router)
