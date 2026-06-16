from langchain_huggingface import HuggingFacePipeline
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from app.config import QWEN_BASE_MODEL, USE_LOCAL_QWEN
from app.llm.openai_model import get_openai_model


ADAPTER_MAPPING = {
    "thinking": "adapters/qwen3-0.6B-thinking",
    "coding": "adapters/qwen3-0.6B-coding",
    "query": "adapters/qwen3-0.6B-thinking",
}


def get_qwen_model(name: str | None = None):
    """로컬 Qwen LoRA 모델 또는 OpenAI 대체 모델을 반환한다.

    기본값은 OpenAI 대체 모델이다. 로컬 Qwen을 쓰려면 .env에
    USE_LOCAL_QWEN=true 를 설정한다.
    """

    if not USE_LOCAL_QWEN:
        return get_openai_model()

    tokenizer = AutoTokenizer.from_pretrained(QWEN_BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        QWEN_BASE_MODEL,
        device_map="auto",
    )

    adapter_path = ADAPTER_MAPPING.get(name or "")
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        do_sample=False,
    )

    return HuggingFacePipeline(pipeline=pipe)
