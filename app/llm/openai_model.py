from langchain_openai import ChatOpenAI

from app.config import OPENAI_MODEL, OPENAI_TEMPERATURE


def get_openai_model(model: str | None = None, temperature: float | None = None):
    """OpenAI Chat LLM 객체를 생성한다.

    OPENAI_API_KEY는 .env 또는 시스템 환경변수에 있어야 한다.
    """

    return ChatOpenAI(
        model=model or OPENAI_MODEL,
        temperature=OPENAI_TEMPERATURE if temperature is None else temperature,
    )
