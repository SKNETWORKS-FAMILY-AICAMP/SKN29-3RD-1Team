from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from peft import PeftModel
from transformers import pipeline

from langchain_huggingface import HuggingFacePipeline


def get_qwen_model(adapter_path: str = None):

    model_name = "Qwen/Qwen3-0.6B"

    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto"
    )

    if adapter_path:
        model = PeftModel.from_pretrained(
            model,
            adapter_path
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

