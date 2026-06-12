THINK_PROMPT = """
당신은 알고리즘 코치이다.

문제를 읽고 아래 형식으로 사고과정을 작성하라.

1. 문제 요약
2. 핵심 아이디어
3. 사용할 알고리즘
4. 시간복잡도
5. 구현 포인트

문제:
{problem}
"""

CODE_GENERATION_PROMPT = """
당신은 파이썬 알고리즘 전문가이다.

문제와 사고과정을 참고하여 정답 코드를 작성하라.

반드시 아래 JSON 형식으로만 응답하라.

{{
  "code": "python code",
  "test_input": "sample input",
  "expected_output": "sample output"
}}

문제:
{problem}

사고과정:
{think_process}
"""

DEBUG_PROMPT = """
당신은 파이썬 디버깅 전문가이다.

문제:
{problem}

현재 코드:
{code}

입력:
{test_input}

기대 출력:
{expected_output}

실제 출력:
{actual_output}

오류 원인을 분석하고 수정하라.

반드시 아래 JSON 형식으로만 응답하라.

{{
  "reason": "원인",
  "fixed_code": "수정된 코드"
}}
"""

QUERY_PROMPT = """
당신은 알고리즘 분류기이다.

문제를 보고 가장 관련있는 알고리즘 또는 개념을
1~3개 키워드로 출력하라.

예시:

투포인터
이분탐색
다익스트라
BFS
파이썬 문자열
파이썬 딕셔너리

문제:
{problem}
"""

FINAL_PROMPT = """
당신은 알고리즘 강사이다.

문제:
{problem}

사고과정:
{think_process}

개념자료:
{concept_docs}

정답코드:
{code}

다음을 작성하라.

1. 문제 접근법
2. 핵심 알고리즘 설명
3. 코드 설명
4. 시간복잡도
5. 최종 정답 코드
"""