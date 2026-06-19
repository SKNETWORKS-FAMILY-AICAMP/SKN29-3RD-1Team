from langchain_huggingface import HuggingFacePipeline
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from app.llm.openai_model import get_openai_model
from app.config import ENABLE_LOCAL_MODEL, QWEN_MODEL_NAME
from langchain_huggingface import HuggingFacePipeline

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

ADAPTER_MAPPING = {
    "think": BASE_DIR / "app" / "scripts" / "outputs" / "split_adapters" / "qwen2.5-3b-thinking-lora",
    "coding": BASE_DIR / "app" / "scripts" / "outputs" / "split_adapters" / "qwen2.5-3b-code-lora",
    "query": BASE_DIR / "app" / "scripts" / "outputs" / "split_adapters" / "qwen2.5-3b-thinking-lora",
}


def get_qwen_model(name: str = None):
    if not ENABLE_LOCAL_MODEL:
        return get_openai_model()
    
    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL_NAME,
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
