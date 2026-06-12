# health check router
from fastapi import APIRouter
from app.rag.retriever import get_retriever
from app.llm.openai_model import get_openai_model
from app.llm.qwen_model import get_qwen_model
from app.agent.solve_agent import solve_graph


router = APIRouter(prefix="/health", tags=["health"])

@router.get("")
def health_check():
    return {"status": "healthy"}

@router.get("/rag")
def rag_health_check():
    try:
        retriever = get_retriever()
        docs = retriever.invoke("test query")
        return {
            "status": "healthy", 
            "retrieved_docs": len(docs),
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


@router.get("/chat")
def chat(request: str):
    response = get_openai_model().invoke(request)

    return {
        "answer": response.content
    }

@router.get("/qwen")
def qwen_chat(request: str):
    llm = get_qwen_model()

    print("question:", request)
    response = llm.invoke(request)
    print("Qwen response:", response)

    return {
        "answer": response
    }

@router.get("/solve")
def solve_problem(request: str):
    response = solve_graph.invoke({"problem": request})
    return {
        "answer": response["final_answer"]
    }