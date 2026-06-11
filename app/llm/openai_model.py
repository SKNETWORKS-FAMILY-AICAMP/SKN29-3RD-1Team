from langchain_openai import ChatOpenAI


def get_openai_model():

    return ChatOpenAI(
        model="gpt-5.5",
        temperature=0
    )