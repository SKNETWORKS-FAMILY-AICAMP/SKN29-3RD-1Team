from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class BaseModelConfig:
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    cuda_device: int = 0
    require_cuda: bool = True
    trust_remote_code: bool = True
    low_cpu_mem_usage: bool = True
    attn_implementation: Optional[str] = None


def select_device(config: BaseModelConfig) -> torch.device:
    if not torch.cuda.is_available():
        if config.require_cuda:
            raise RuntimeError(
                "CUDA is not available. Install a CUDA-enabled PyTorch build "
                "or run with --no-cuda for CPU-only validation."
            )
        print("[warning] CUDA is not available. Using CPU.")
        return torch.device("cpu")

    torch.cuda.set_device(config.cuda_device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = torch.device(f"cuda:{config.cuda_device}")
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA device count:", torch.cuda.device_count())
    print("GPU:", torch.cuda.get_device_name(config.cuda_device))
    print("Selected device:", device)
    return device


def load_tokenizer(config: BaseModelConfig):
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=config.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_base_model(config: BaseModelConfig, device: torch.device):
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    kwargs = {
        "torch_dtype": dtype,
        "low_cpu_mem_usage": config.low_cpu_mem_usage,
        "trust_remote_code": config.trust_remote_code,
    }
    if config.attn_implementation:
        kwargs["attn_implementation"] = config.attn_implementation

    model = AutoModelForCausalLM.from_pretrained(config.model_name, **kwargs)
    model.to(device)
    model.config.use_cache = False
    return model


def load_base_model_and_tokenizer(config: BaseModelConfig, device: torch.device):
    tokenizer = load_tokenizer(config)
    model = load_base_model(config, device)
    print("Base model device:", next(model.parameters()).device)
    return model, tokenizer
