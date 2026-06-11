from langchain_core.prompts import ChatPromptTemplate

from app.llm.openai_model import get_openai_model


think_prompt = ChatPromptTemplate.from_template(
    """
    다음 문제를 읽고 문제이해, 접근방법 생각, 입력 size기반 시간복잡도 검토, 코드 구현과정 사고 등의 문제풀이 사고과정 출력하라

    문제:
    {question}
    """
)


chain = think_prompt | get_openai_model()