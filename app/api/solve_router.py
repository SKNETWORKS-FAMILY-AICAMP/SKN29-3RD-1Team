from fastapi import APIRouter
from pydantic import BaseModel
from app.agent.solve_agent import solve_graph
from app.utils import pretty_print, dump_agent_state, safe_file_name
from app.config import get_llm_setting_string

router = APIRouter(
    prefix="/preview",
    tags=['preview']
)


class SolveRequest(BaseModel):
    problem: str

class ExecutionTrace(BaseModel):
    code: str
    result: str
    success: bool

class SolveResponse(BaseModel):
    problem: str
    think_process: str
    execution_traces: list[ExecutionTrace]
    code: str
    concept_docs: str # \n\n로 구분된 문서들 덩어리 문자열
    final_answer: str


@router.post("/solve", response_model=SolveResponse)
def save_document(request: SolveRequest):
    state = solve_graph.invoke({"problem": request.problem})
    dump_agent_state("solve_agent", state, safe_file_name("solve_" + get_llm_setting_string()))

    execution_traces = []
    for et in state['execution_traces']:
        
        execution_traces.append(
            ExecutionTrace(
                code=et['code'],
                result=et['result'],
                success=(et['result'] == state['expected_output']),
            )
        )

    if state['code_passed']:
        response = SolveResponse(
            problem=state['problem'],
            think_process=state['think_process'],
            execution_traces=execution_traces,
            code=state['code'],
            concept_docs=state['concept_docs'],
            final_answer=state['final_answer'],
        )
    else:
        response = SolveResponse(
            problem=state['problem'],
            think_process=state['think_process'],
            execution_traces=execution_traces,
            code=state['code'],
            concept_docs="",
            final_answer="문제 풀이에 실패하였습니다.",
        )
    
    pretty_print(response)
    return response
