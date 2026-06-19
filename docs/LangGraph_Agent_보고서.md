# LangGraph Agent 설계 보고서

## 담당자
- 김진욱
- 양정현

---

## 목차

1. [개요](#1-개요)
2. [전체 그래프 구조](#2-전체-그래프-구조)
3. [State 설계](#3-state-설계)
4. [노드 설명](#4-노드-설명)
5. [엣지 및 라우팅 로직](#5-엣지-및-라우팅-로직)
6. [사용 모델](#6-사용-모델)
7. [프롬프트 설계](#7-프롬프트-설계)
8. [실행 샌드박스](#8-실행-샌드박스)
9. [실행 흐름 예시](#9-실행-흐름-예시)
10. [설계 결정 이유](#10-설계-결정-이유)
11. [한계 및 향후 개선](#11-한계-및-향후-개선)

---

## 1. 개요

본 파트는 코딩테스트 문제를 입력받아 최종 풀이 설명을 생성하는 **LangGraph 기반 멀티스텝 에이전트**를 담당한다.

핵심 목표:

- 문제를 받아 알고리즘 풀이 전략을 수립한다.
- 파이썬 코드를 생성하고 실제로 실행하여 정답 여부를 검증한다.
- 오류 발생 시 자동으로 디버깅을 수행한다.
- RAG 시스템에서 관련 알고리즘 개념 문서를 검색하여 설명에 활용한다.
- 최종적으로 사용자에게 접근법, 알고리즘 설명, 코드 설명, 시간복잡도를 포함한 완성된 답변을 제공한다.

담당 파일:

```text
app/agent/solve_agent.py
app/llm/prompts.py
app/llm/openai_model.py
app/llm/qwen_model.py
app/mcp/tools.py
```

API 진입점:

```text
POST /preview/solve
```

---

## 2. 전체 그래프 구조

```text
START
  ↓
[think]              ← 문제 분석 및 풀이 전략 수립
  ↓
[generate_code]      ← 파이썬 코드 생성
  ↓
[execute]            ← 코드 실행 및 출력 검증
  ↓
  ├── 성공 ──────→ [algorithm_query]   ← RAG 검색 쿼리 생성
  │                     ↓
  │               [retrieve]           ← ChromaDB에서 관련 문서 검색
  │                     ↓
  │               [final_answer]       ← 최종 답변 생성
  │                     ↓
  │                    END
  │
  └── 실패 ──────→ [debug]             ← 오류 원인 분석 및 코드 수정
                        ↓
                    retry_count < 3 → [execute]  (재시도)
                    retry_count >= 3 → END        (포기)
```

---

## 3. State 설계

그래프 전체에서 공유되는 상태 객체이다.

```python
class SolveState(TypedDict):
    problem: str            # 입력 문제
    think_process: str      # 사고 과정
    code: str               # 생성된 파이썬 코드
    test_input: str         # 테스트 입력값
    expected_output: str    # 기대 출력값
    execution_result: str   # 실제 실행 결과
    retry_count: int        # 디버깅 재시도 횟수
    code_passed: bool       # 코드 통과 여부
    execution_traces: list[ExecutionTrace]  # 디버그 코드 결과 기록
    algorithm_query: str    # RAG 검색용 쿼리
    concept_docs: str       # 검색된 알고리즘 개념 문서
    final_answer: str       # 최종 답변
```

각 노드는 상태의 일부 필드만 업데이트하고 나머지는 그대로 유지된다. LangGraph는 노드 반환값을 기존 상태에 머지(merge)하는 방식으로 동작한다.

---

## 4. 노드 설명

### 4-1. think_node

역할: 문제를 분석하고 풀이 전략을 수립한다.

사용 모델: `get_qwen_model("think")` 또는 `OpenAI GPT-4o-mini`

입력 상태: `problem`

출력 상태: `think_process`

프롬프트 출력 형식:

```text
1. 문제 요약
2. 핵심 아이디어
3. 사용할 알고리즘
4. 시간복잡도
5. 구현 포인트
```

---

### 4-2. generate_code_node

역할: think_node의 사고 과정을 참고하여 실행 가능한 파이썬 코드를 생성한다.

사용 모델: `get_openai_model()` (GPT-4o-mini, temperature=0)

입력 상태: `problem`, `think_process`

출력 상태: `code`, `test_input`, `expected_output`

LLM 응답 형식 (JSON 강제):

```json
{
  "code": "python code",
  "test_input": "sample input",
  "expected_output": "sample output"
}
```

JSON 파싱은 `app/utils.py`의 `extract_json()` 함수로 처리한다.

---

### 4-3. execute_node

역할: 생성된 코드를 실제로 실행하여 출력값을 확인한다.

사용 도구: `app/mcp/tools.py`의 `execute_python()`

입력 상태: `code`, `test_input`

출력 상태: `execution_result`

실행 방식: `exec()`를 사용한 로컬 인메모리 샌드박스이며, stdin을 `io.StringIO`로 주입한다.

오류 발생 시 `execution_result`에 `ERROR: <오류메시지>` 형태로 저장된다.

---

### 4-4. debug_node

역할: 실행 결과가 기대값과 다를 때 오류 원인을 분석하고 코드를 수정한다.

사용 모델: `get_openai_model()`

입력 상태: `problem`, `code`, `test_input`, `expected_output`, `execution_result`

출력 상태: `code` (수정된 코드), `retry_count` (+1 증가)

LLM 응답 형식 (JSON 강제):

```json
{
  "reason": "원인 설명",
  "fixed_code": "수정된 코드"
}
```

---

### 4-5. algorithm_query_node

역할: 문제에서 관련 알고리즘 키워드를 추출하여 RAG 검색 쿼리를 생성한다.

사용 모델: `get_openai_model()`

입력 상태: `problem`

출력 상태: `algorithm_query`

출력 예시:

```text
투포인터
이분탐색
```

---

### 4-6. retrieve_node

역할: algorithm_query를 사용해 ChromaDB에서 관련 알고리즘 개념 문서를 검색한다.

사용 모듈: `app/rag/retriever.py`의 `get_retriever()`

검색 설정: similarity search, Top-5

입력 상태: `algorithm_query`

출력 상태: `concept_docs` (검색된 문서들을 `\n\n`으로 연결한 문자열)

---

### 4-7. final_answer_node

역할: 문제, 사고 과정, 검색된 개념 문서, 정답 코드를 종합하여 최종 답변을 생성한다.

사용 모델: `get_openai_model()`

입력 상태: `problem`, `think_process`, `code`, `concept_docs`

출력 상태: `final_answer`

출력 형식:

```text
1. 문제 접근법
2. 핵심 알고리즘 설명
3. 코드 설명
4. 시간복잡도
5. 최종 정답 코드
```

---

## 5. 엣지 및 라우팅 로직

### 고정 엣지

```text
START → think → generate_code → execute
algorithm_query → retrieve → final_answer → END
```

### 조건부 엣지 1: execution_router

`execute` 노드 이후 실행 결과를 판단한다.

```python
def execution_router(state):
    if state["execution_result"].strip() == state["expected_output"].strip():
        return "success"
    return "retry"
```

| 결과 | 다음 노드 |
|------|-----------|
| `"success"` | `algorithm_query` |
| `"retry"` | `debug` |

### 조건부 엣지 2: retry_router

`debug` 노드 이후 재시도 횟수를 판단한다.

```python
def retry_router(state):
    if state["retry_count"] >= 3:
        return "failed"
    return "execute"
```

| 결과 | 다음 노드 |
|------|-----------|
| `"execute"` | `execute` (재시도) |
| `"failed"` | `END` (포기) |

최대 디버깅 시도 횟수: **3회**

3회 초과 시 `final_answer` 생성 없이 그래프가 종료된다.

---

## 6. 사용 모델

| 노드 | 함수 | 실제 모델 | 비고 |
|------|------|-----------|------|
| think | `get_qwen_model("think")` | Qwen2.5-3B | ENABLE_LOCAL_MODEL=True 필요 |
| generate_code | `get_openai_model()` | GPT-4o-mini | temperature=0 |
| debug | `get_openai_model()` | GPT-4o-mini | temperature=0 |
| algorithm_query | `get_openai_model()` | GPT-4o-mini | temperature=0 |
| final_answer | `get_openai_model()` | GPT-4o-mini | temperature=0 |

`.env`의 `ENABLE_LOCAL_MODEL=True` 설정 시 로컬 Qwen 호출이 가능하다.

[로라 어댑터 G-Drive]()

어댑터 다운로드 후 어댑터 경로 매핑 수정
```python
# app/llm/qwen_model.py
ADAPTER_MAPPING = {
    "think": "adapters/qwen3-0.6B-thinking", # 실제 어댑터 경로로 수정
    "coding":   "adapters/qwen3-0.6B-coding"
}
```

---

## 7. 프롬프트 설계

### THINK_PROMPT

문제 분석 단계. 구조화된 사고 과정을 강제하여 이후 코드 생성 품질을 높인다.

```
1. 문제 요약
2. 핵심 아이디어
3. 사용할 알고리즘
4. 시간복잡도
5. 구현 포인트
```

### CODE_GENERATION_PROMPT

코드 생성 단계. JSON 형식을 강제하여 파싱 실패 가능성을 줄인다. 테스트 입력과 기대 출력을 함께 생성하여 자동 검증이 가능하도록 설계했다.

### DEBUG_PROMPT

디버깅 단계. 문제, 코드, 입력, 기대 출력, 실제 출력을 모두 제공하여 LLM이 오류 원인을 정확히 파악할 수 있도록 한다.

### QUERY_PROMPT

RAG 쿼리 생성 단계. 알고리즘 키워드 1~3개만 출력하도록 제한하여 검색 노이즈를 줄인다.

### FINAL_PROMPT

최종 답변 단계. 검색된 개념 문서(`concept_docs`)를 컨텍스트로 포함하여 RAG 기반 설명이 가능하도록 한다.

---

## 8. 실행 샌드박스

코드 실행은 별도 프로세스 없이 Python 런타임 내부에서 수행된다.

```python
@mcp.tool()
def execute_python(
    code: str,
    stdin: str = ""
) -> str:
    buffer = io.StringIO()
    old_stdin = sys.stdin

    try:
        sys.stdin = io.StringIO(stdin)

        exec_globals = {
            "__name__": "__main__"
        }

        with redirect_stdout(buffer):
            exec(code, exec_globals)

        return buffer.getvalue()

    except Exception as e:
        return f"ERROR: {e}"

    finally:
        sys.stdin = old_stdin
```

특징:

- `exec(code, exec_globals)`: 생성된 Python 코드 실행
- `__name__="__main__`": 일반 스크립트와 유사한 실행 환경 제공
- `redirect_stdout`: `print()` 출력 캡처
- `io.StringIO(stdin)`: 표준 입력 주입 (`input()` 지원)
- 오류 발생 시 ERROR: <메시지> 반환

현재는 Agent가 생성한 알고리즘 문제 풀이 코드의 동작을 검증하기 위한 용도로 사용한다.

한계:

- 실행 시간 제한 없음
- 메모리 제한 없음
- 파일 시스템 및 네트워크 접근 제한 없음
- 신뢰할 수 없는 코드 실행 환경으로 사용하기 어려움

현재 호출 방식

현재는 MCP 프로토콜을 거치지 않고 execute_python을 직접 임포트하여 호출한다.
```
from app.mcp.tools import execute_python

result = execute_python(
    code=state["code"],
    stdin=state["test_input"]
)
```

tools.py에는 @mcp.tool() 데코레이터가 선언되어 있으나, FastMCP 서버는 아직 별도 프로세스로 기동되지 않는다.

MCP 서버 연동 (추가 예정)
```
TODO

MCP 서버 기동 방식 및 포트 설정
에이전트의 MCP Client 연동 구조
추가 MCP Tool 목록
LangGraph 노드와 MCP Tool 연결 방식

※ 4차 개발 단계에서 Docker 기반 보안 격리 및 보다 유연한 테스트케이스 입력 구조를 도입할 예정이다.
```
---

## 9. 실행 흐름 예시

입력 문제:

```text
정수 N개가 주어질 때 가장 큰 수를 출력하라.
```

실행 순서:

```text
1. think_node
   → "배열을 순회하며 최댓값을 찾는다. max() 함수 사용. O(N)"

2. generate_code_node
   → code: "n = int(input())\narr = list(map(int, input().split()))\nprint(max(arr))"
   → test_input: "5\n3 1 4 1 5"
   → expected_output: "5"

3. execute_node
   → execution_result: "5"

4. execution_router
   → "5" == "5" → "success"

5. algorithm_query_node
   → "정렬, 최댓값"

6. retrieve_node
   → ChromaDB에서 정렬/최댓값 관련 문서 5개 검색

7. final_answer_node
   → 접근법, 알고리즘 설명, 코드 설명, O(N) 시간복잡도, 최종 코드 포함 답변 생성
```

---

## 10. 설계 결정 이유

### LangGraph를 선택한 이유

단순 LLM 체인이 아니라 조건부 분기(성공/실패)와 반복(디버깅 재시도)이 필요했다. LangGraph는 StateGraph를 통해 이러한 사이클 구조를 명시적으로 표현할 수 있다.

### 코드를 실제 실행하는 이유

LLM이 생성한 코드가 항상 정확하지 않다. 실행 결과를 직접 검증하면 잘못된 코드를 자동으로 감지하고 디버깅 루프를 통해 수정할 수 있다.

### 최대 재시도 3회로 제한한 이유

동일한 오류에 대해 LLM이 같은 방식의 수정을 반복할 수 있다. 3회 초과 시 강제 종료하여 무한 루프와 불필요한 API 비용을 방지한다.

### RAG를 코드 검증 이후에 수행하는 이유

코드가 정확히 동작하는 것이 확인된 이후에 개념 설명을 보강한다. 코드가 틀린 상태에서 개념 설명을 생성하면 일관성이 없는 답변이 만들어질 수 있다.

### JSON 응답 형식을 강제하는 이유

`generate_code_node`와 `debug_node`에서 LLM 응답을 파싱하여 구조화된 필드를 추출해야 한다. 자연어 응답은 파싱 실패 가능성이 높기 때문에 JSON 형식을 프롬프트에 명시적으로 요구한다.

---

## 11. 한계 및 향후 개선

### 현재 한계

| 항목 | 내용 |
|------|------|
| 실행 샌드박스 | 무한루프, 대용량 메모리 사용 방어 없음 |
| 재시도 실패 시 | final_answer 없이 END 종료 |
| 단일 테스트 케이스 | 하나의 테스트 입력만으로 정답 여부 판단 |

### 향후 개선 방향

1. **타임아웃 및 메모리 제한** 추가하여 샌드박스 안정성 확보
2. **멀티 테스트 케이스 검증**: 단일 케이스가 아닌 여러 케이스로 코드 정확도 향상
3. **재시도 실패 시 부분 답변 생성**: 코드 검증 실패에도 think_process 기반 답변 반환
5. **스트리밍 응답**: 최종 답변을 청크 단위로 스트리밍하여 응답 체감 속도 개선
