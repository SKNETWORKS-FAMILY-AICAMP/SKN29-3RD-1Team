from __future__ import annotations

import argparse
import csv
import json
import re
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

SCRIPT_DIR = Path(__file__).resolve().parent
APP_DIR = SCRIPT_DIR.parent

DEFAULT_DATASET = APP_DIR / "data" / "opencode_reasoning_test_1000.jsonl"
DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "outputs" / "base_eval_split"
DEFAULT_ADAPTER = SCRIPT_DIR / "outputs" / "qwen2.5-3b-split-lora"
DEFAULT_SPLIT_ADAPTER_ROOT = SCRIPT_DIR / "outputs" / "split_adapters_tuning_2"     # default경로 "split_adapters"
DEFAULT_KEYWORD_ADAPTER = DEFAULT_SPLIT_ADAPTER_ROOT / "qwen2.5-3b-keyword-lora"
DEFAULT_THINKING_ADAPTER = DEFAULT_SPLIT_ADAPTER_ROOT / "qwen2.5-3b-thinking-lora"
DEFAULT_CODE_ADAPTER = DEFAULT_SPLIT_ADAPTER_ROOT / "qwen2.5-3b-code-lora"
DEFAULT_LORA_OUTPUT_ROOT = SCRIPT_DIR / "outputs" / "lora_eval_split_tuning_2"

ALLOWED_VOCAB = f'''{
    "implementation", "python_basic", "parsing", "type_conversion",
    "condition", "loop", "iteration", "range", "math", "modulo",
    "operator", "string", "string_processing", "formatting", "regex",
    "function_definition", "lambda_function", "recursion",
    "recursive_function", "base_case", "recursive_call",
    "data_structure", "list", "tuple", "dictionary", "set",
    "array", "indexing", "slicing", "list_comprehension",
    "key_value", "hashing", "hashmap", "counter", "frequency",
    "counting", "collections", "itertools", "deque", "queue",
    "stack", "hash", "heap", "heapq", "priority_queue",
    "sort", "sorting", "binary_search", "lower_bound", "upper_bound",
    "parametric_search", "two_pointers", "sliding_window",
    "brute_force", "permutation", "combination", "combinatorics",
    "backtracking", "visited", "simulation", "grid",
    "greedy", "optimization",
    "graph", "adjacency_list", "adjacency_matrix", "bfs", "dfs",
    "connected_component", "multi_source_bfs", "tree",
    "shortest_path", "dijkstra", "floyd_warshall", "bellman_ford",
    "union_find", "topological_sort", "mst",
    "minimum_spanning_tree", "kruskal", "prim",
    "dynamic_programming", "memoization", "tabulation",
    "recurrence_relation", "prefix_sum", "bitmask", "state_dp",
    "tree_dp",
    "tree_traversal", "binary_tree", "bst", "lca",
    "segment_tree", "lazy_propagation", "fenwick_tree",
    "binary_indexed_tree", "range_query", "point_update",
    "kmp", "trie", "rabin_karp",
    "coordinates", "geometry", "distance",
    "time_complexity"
}'''

SAFE_KEYWORDS = "python_basic | data_structure | parsing | implementation | base_case | condition"


def read_dataset(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    rows.append(obj if isinstance(obj, dict) else {"input": obj})
        return rows

    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [x if isinstance(x, dict) else {"input": x} for x in data]
    for key in ("data", "samples", "items", "problems"):
        if isinstance(data.get(key), list):
            return [x if isinstance(x, dict) else {"input": x} for x in data[key]]
    return [data]


def first_text(sample: Dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = sample.get(key)
        if value:
            return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return ""


def build_problem_text(sample: Dict[str, Any]) -> str:
    if isinstance(sample.get("messages"), list):
        user_parts = [
            str(message.get("content", "")).strip()
            for message in sample["messages"]
            if message.get("role") in {"user", "human"} and message.get("content")
        ]
        if user_parts:
            return "\n\n".join(user_parts)

    parts = [
        first_text(sample, "instruction", "prompt", "query"),
        first_text(sample, "title", "name"),
        first_text(sample, "algorithm", "algorithm_tag", "tag"),
        first_text(sample, "input", "problem", "question", "content", "text"),
    ]
    return "\n\n".join(part for part in parts if part).strip() or json.dumps(sample, ensure_ascii=False)


class BaseModelRunner:
    def __init__(self, model_name: str, max_new_tokens: int, device_map: str = "auto") -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise RuntimeError("Install torch and transformers before running evaluation.") from e

        self.torch = torch
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
        )
        self.model.eval()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(
        self,
        system_prompt: str,
        problem_text: str,
        do_sample: bool,
        temperature: float,
        top_p: float,
        max_input_length: int = 2048,
    ) -> tuple[str, int, bool, float]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": problem_text},
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        start = time.time()
        with self.torch.inference_mode():
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_length)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            kwargs = {
                "max_new_tokens": self.max_new_tokens,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "do_sample": do_sample,
            }
            if do_sample:
                kwargs.update({"temperature": temperature, "top_p": top_p})

            output_ids = self.model.generate(**inputs, **kwargs)
            generated_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
            text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            token_count = int(generated_ids.shape[-1])
            truncated = token_count >= self.max_new_tokens
            if self.tokenizer.eos_token_id is not None and token_count:
                truncated = truncated and int(generated_ids[-1]) != self.tokenizer.eos_token_id
        return text, token_count, truncated, round(time.time() - start, 4)


class LoraModelRunner:
    def __init__(self, model_name: str, adapter_path: str, max_new_tokens: int, device_map: str = "auto") -> None:
        try:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise RuntimeError("Install torch, transformers, and peft before running LoRA evaluation.") from e

        self.torch = torch
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
        )
        self.model = PeftModel.from_pretrained(base_model, adapter_path)
        self.model.eval()

    def generate(
        self,
        system_prompt: str,
        problem_text: str,
        do_sample: bool,
        temperature: float,
        top_p: float,
        max_input_length: int = 2048,
    ) -> tuple[str, int, bool, float]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": problem_text},
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        start = time.time()
        with self.torch.inference_mode():
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_length)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            kwargs = {
                "max_new_tokens": self.max_new_tokens,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "do_sample": do_sample,
            }
            if do_sample:
                kwargs.update({"temperature": temperature, "top_p": top_p})

            output_ids = self.model.generate(**inputs, **kwargs)
            generated_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
            text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            token_count = int(generated_ids.shape[-1])
            truncated = token_count >= self.max_new_tokens
            if self.tokenizer.eos_token_id is not None and token_count:
                truncated = truncated and int(generated_ids[-1]) != self.tokenizer.eos_token_id
        return text, token_count, truncated, round(time.time() - start, 4)


def add_common_args(parser: argparse.ArgumentParser, output_dir: Path, max_new_tokens: int) -> None:
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", default=str(output_dir))
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=max_new_tokens)
    parser.add_argument("--max-generation-seconds", type=float, default=20.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--device-map", default="auto")


def add_lora_args(
    parser: argparse.ArgumentParser,
    output_dir: Path,
    max_new_tokens: int,
    adapter_path: Path = DEFAULT_ADAPTER,
) -> None:
    add_common_args(parser, output_dir, max_new_tokens)
    parser.add_argument("--adapter-path", default=str(adapter_path))


def rate(count: int, total: int) -> float:
    return round(count / total * 100, 2) if total else 0.0


def rows_to_dicts(rows: Iterable[Any]) -> List[Dict[str, Any]]:
    out = []
    for row in rows:
        out.append(asdict(row) if is_dataclass(row) else dict(row))
    return out


def save_outputs(output_dir: Path, summary_name: str, details_name: str, summary: Dict[str, Any], rows: List[Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    dict_rows = rows_to_dicts(rows)
    (output_dir / summary_name).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / details_name.replace(".csv", ".json")).write_text(
        json.dumps(dict_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if dict_rows:
        with (output_dir / details_name).open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(dict_rows[0].keys()))
            writer.writeheader()
            writer.writerows(dict_rows)


def keyword_list(response: str) -> List[str]:
    match = re.search(r"##\s*RAG Keywords\s*\n(.*?)(?=\n##|\Z)", response or "", re.I | re.S)
    if not match:
        return []
    return [x.strip() for x in match.group(1).strip().splitlines()[0].split("|") if x.strip()]


def is_english_only(text: str) -> bool:
    return not re.search(r"[\uac00-\ud7a3\u4e00-\u9fff]", text or "")


def one_python_code_block(response: str) -> List[str]:
    return re.findall(r"```python\s*\n(.*?)\n```", response or "", re.I | re.S)
