import app.config
from fastapi import FastAPI
from app.api import health_router
from app.api import rag_router


app = FastAPI()
app.include_router(health_router.router)
app.include_router(rag_router.router)