from langchain_huggingface import HuggingFacePipeline
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from app.llm.openai_model import get_openai_model
from app.config import ENABLE_LOCAL_MODEL
from langchain_huggingface import HuggingFacePipeline

ADAPTER_MAPPING = {
    "thinking": "adapters/qwen3-0.6B-thinking", # 실제 경로로 수정
    # "coding": "adapters/qwen3-0.6B-coding",
    # "query": "adapters/qwen3-0.6B-thinking",
}


def get_qwen_model(name: str = None):
    if not ENABLE_LOCAL_MODEL:
        return get_openai_model()

    model_name = "Qwen/Qwen3-0.6B"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
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
