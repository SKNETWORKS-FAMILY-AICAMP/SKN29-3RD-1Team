from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from peft import PeftModel
from transformers import pipeline
from app.llm.openai_model import get_openai_model
from app.config import ENABLE_LOCAL_MODEL
from langchain_huggingface import HuggingFacePipeline

ADAPTER_MAPPING = {
    "thinking": "adapters/qwen3-0.6B-thinking",
    "coding": "adapters/qwen3-0.6B-coding"
}


def get_qwen_model(name: str = None):
    if not ENABLE_LOCAL_MODEL:
        return get_openai_model()
    
    model_name = "Qwen/Qwen3-0.6B"

    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto"
    )

    if name and name in ADAPTER_MAPPING:
        model = PeftModel.from_pretrained(
            model,
            ADAPTER_MAPPING[name]
        )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=32
    )

    return HuggingFacePipeline(
        pipeline=pipe
    )



