from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.agent.solve_agent import solve_graph


router = APIRouter(prefix="/solve", tags=["solve"])


class SolveRequest(BaseModel):
    problem: str = Field(..., description="알고리즘 문제 설명")


class SolveResponse(BaseModel):
    answer: str


@router.post("", response_model=SolveResponse)
def solve_problem(request: SolveRequest):
    result = solve_graph.invoke({"problem": request.problem})
    return {"answer": result["final_answer"]}
