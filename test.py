from app.agent.solve_agent import retrieve_node
from app.utils import pretty_print

test_state = {
    "algorithm_query": "DFS 재귀 구현 방법"
}

result = retrieve_node(test_state)

print("===== 결과 =====")
pretty_print(result)

