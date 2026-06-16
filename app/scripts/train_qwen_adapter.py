from __future__ import annotations

import argparse
import importlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
APP_DIR = SCRIPT_DIR.parent

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"

DEFAULT_TRAIN_FILE = APP_DIR / "data" / "opencode_reasoning_train_4000.jsonl"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs" / "qwen2.5-3b-split-lora"

ALLOWED_VOCAB = {
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
    "greedy", "optimization", "graph", "adjacency_list",
    "adjacency_matrix", "bfs", "dfs", "connected_component",
    "multi_source_bfs", "tree", "shortest_path", "dijkstra",
    "floyd_warshall", "bellman_ford", "union_find", "topological_sort",
    "mst", "minimum_spanning_tree", "kruskal", "prim",
    "dynamic_programming", "memoization", "tabulation",
    "recurrence_relation", "prefix_sum", "bitmask", "state_dp",
    "tree_dp", "tree_traversal", "binary_tree", "bst", "lca",
    "segment_tree", "lazy_propagation", "fenwick_tree",
    "binary_indexed_tree", "range_query", "point_update",
    "kmp", "trie", "rabin_karp", "coordinates", "geometry",
    "distance", "time_complexity",
}

SAFE_KEYWORDS = "python_basic | data_structure | parsing | implementation | base_case | condition"

KEYWORD_SYSTEM_PROMPT = f"""
Goal:
Return only the RAG Keywords section for the coding problem.
Choose exactly 6 reusable solving-concept keywords.
Every keyword must be copied exactly from the allowed vocabulary.

Required format:
## RAG Keywords
keyword1 | keyword2 | keyword3 | keyword4 | keyword5 | keyword6

Format rules:
Your first line must always be exactly:
## RAG Keywords
Never omit the first line.
Your second line must contain exactly 6 keywords.
Your second line must contain exactly 5 separators: " | ".
Stop immediately after the second line.

Allowed vocabulary:
{ALLOWED_VOCAB}

Selection rules:
Choose keywords based on the solution approach, not the problem story.
Choose algorithms, data structures, Python concepts, or implementation techniques.
Use only lowercase snake_case keywords copied from the allowed vocabulary.
Do not create similar words outside the allowed vocabulary.
If a concept is not in the allowed vocabulary, choose the closest allowed keyword.
Do not use section-label words such as input_output, core_concept, solution_strategy, implementation_plan, or verification_points.
Do not use problem-specific words, object names, variable names, story words, or output symbols.
Do not write explanations, analysis, bullets, commas, code, Korean, Chinese, or extra text.

Replacement rules:
Use distance instead of sum_of_distances.
Use geometry or math instead of area_calculation or circle_area.
Use greedy or optimization instead of car_selection.
Use math instead of multiplication.
Use implementation instead of variable_assignment.
Use optimization instead of maximize.
Use string_processing instead of palindrome or substring.
Use modulo instead of modular_arithmetic.
Use graph instead of graph_theory.
Use tree instead of tree_data_structure.

Priority:
algorithm > data_structure > implementation > loop > condition > math

Fallback:
If you are not sure, use exactly this second line:
{SAFE_KEYWORDS}

Example:
## RAG Keywords
implementation | grid | loop | condition | modulo | string_processing
""".strip()

THINKING_SYSTEM_PROMPT = """
Write only the Thinking Process section.

Goals:
Each answer must be one beginner-friendly English sentence.
You must follow the output pattern only use the string '## Thinking Process', '1. Problem Understanding', '2. Input/Output Analysis', '3. Core Concept', '4. Solving Strategy', '5. Implementation Plan'.

Hard bans:
Never write the string '1. **Problem Understanding**', '2. **Input/Output Analysis**', '3. **Core Concept**', '4. **Solving Strategy**', '5. **Implementation Plan**'.
Never write any label string starting with 6 in output pattern.
Do not add 'Final Answer'.
Do not use Korean, Chinese.
Do not use bullets, Markdown decoration, code, examples, or extra sections.
Do not write "## Verification Points".

Correct output pattern:
## Thinking Process
1. Problem Understanding: Easy keyword short English sentence.
2. Input/Output Analysis: Easy keyword short English sentence.
3. Core Concept: Easy keyword short English sentence.
4. Solving Strategy: Easy keyword short English sentence.
5. Implementation Plan: Easy keyword short English sentence.
""".strip()

REQUIRED_LABELS = [
    "## Thinking Process",
    "1. Problem Understanding:",
    "2. Input/Output Analysis:",
    "3. Core Concept:",
    "4. Solving Strategy:",
    "5. Implementation Plan:",
]

CODE_SYSTEM_PROMPT = """
Goals:
Output EXACTLY 2 lines. Non-negotiable.
Line 1: ## Final Answer
Line 2: One complete Python code block with working solution

Instruction:
Write the complete, executable Python solution for the coding problem.
Do not add anything else.

Output Format (MANDATORY):
## Final Answer
```python
complete_python_solution_here
```

ABSOLUTE Rules:
1. Exactly 2 lines total. No exceptions.
2. Line 1 is always: ## Final Answer
3. Line 2 is EXACTLY one Python code block (triple backticks with python tag).
4. The code block contains the COMPLETE working solution.
5. STOP immediately after the closing backticks ```.
6. Write nothing after line 2.

What Code Must Include:
- All necessary imports
- All function definitions
- Input/output handling (input(), print())
- Complete solution logic
- No placeholder comments like "# TODO", "# write code here"
- No incomplete functions
- No sample test calls

What Code Must NOT Include:
- RAG Keywords section
- Thinking Process section
- Explanation outside code block
- Multiple code blocks
- Extra sections or headers
- Comments explaining logic
- Test cases or examples

Forbidden:
- Do NOT write: "# write final code here"
- Do NOT write: "# Example usage" 
- Do NOT use placeholders
- Do NOT call functions with test values
- Do NOT add text after line 2
- Do NOT write anything except the 2 lines

If code is incomplete or unsure, still output the 2-line format with best attempt.
After line 2, STOP immediately. Write nothing.
""".strip()

@dataclass
class TrainConfig:
    base_model_name: str = DEFAULT_BASE_MODEL
    train_file: str = str(DEFAULT_TRAIN_FILE)
    output_dir: str = str(DEFAULT_OUTPUT_DIR)
    max_length: int = 1024
    limit: int = 0
    task_mix: str = "keyword,code"
    include_full: bool = False
    num_train_epochs: float = 1.0
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 1e-4
    logging_steps: int = 10
    save_steps: int = 500
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    cuda_device: int = 0
    require_cuda: bool = True
    device_map: str = "auto"
    max_gpu_memory: str = "9.8GiB"
    max_cpu_memory: str = "30GiB"
    offload_folder: str = str(SCRIPT_DIR / "outputs" / "offload")

def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train Qwen2.5 LoRA adapter with content-focused keyword/code tasks.")
    parser.add_argument("--base-model-name", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--train-file", default=str(DEFAULT_TRAIN_FILE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=0, help="Use 0 for all rows.")
    parser.add_argument("--task-mix", default="keyword,code", help="Comma-separated tasks. Recommended: keyword,code. Optional: thinking,full.")
    parser.add_argument("--include-full", action="store_true", help="Also train on assembled full answers.")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--device-map", default="auto", help="Use auto for CPU/GPU offload, or none to place the full model on one device.")
    parser.add_argument("--max-gpu-memory", default="", help="Optional GPU memory cap such as 7GiB. Empty means no explicit cap.")
    parser.add_argument("--max-cpu-memory", default="24GiB", help="CPU memory cap used with device_map auto.")
    parser.add_argument("--offload-folder", default=str(SCRIPT_DIR / "outputs" / "offload"))
    args = parser.parse_args()

    return TrainConfig(
        base_model_name=args.base_model_name,
        train_file=str(Path(args.train_file).expanduser()),
        output_dir=str(Path(args.output_dir).expanduser()),
        max_length=args.max_length,
        limit=args.limit,
        task_mix=args.task_mix,
        include_full=args.include_full,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        cuda_device=args.cuda_device,
        require_cuda=not args.no_cuda,
        device_map=args.device_map,
        max_gpu_memory=args.max_gpu_memory,
        max_cpu_memory=args.max_cpu_memory,
        offload_folder=str(Path(args.offload_folder).expanduser()),
    )

def verify_training_dependencies() -> None:
    required = {
        "torch": "torch",
        "transformers": "transformers",
        "datasets": "datasets",
        "peft": "peft",
        "accelerate": "accelerate",
    }
    missing = []
    for package_name, module_name in required.items():
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError:
            missing.append(package_name)
    if missing:
        raise RuntimeError("Missing training packages: " + ", ".join(missing))


def first_text(example: Dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = example.get(key)
        if value:
            return str(value).strip()
    return ""


def problem_text(example: Dict[str, Any]) -> str:
    return first_text(example, "input", "problem", "question", "prompt", "content", "text")


def raw_output(example: Dict[str, Any]) -> str:
    return first_text(example, "output", "response", "answer", "completion")


def extract_code(output: str) -> str:
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)\n```", output or "", re.I | re.S)
    if blocks:
        code = blocks[-1].strip()
    else:
        code = ""
    if "# Example usage" in code or "complete_python_solution" in code:
        return ""
    return code


def clean_inline_text(text: str) -> str:
    text = re.sub(r"[*`#]+", "", text or "")
    text = re.sub(r"^\s*[-?\s]*", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.rstrip(".") + "." if text else ""


def extract_rag_keywords(output: str) -> List[str]:
    match = re.search(r"##\s*RAG Keywords\s*\n([^\n]+)", output or "", re.I)
    if not match:
        return []
    raw_items = [item.strip().lower() for item in match.group(1).split("|")]
    keywords: List[str] = []
    for item in raw_items:
        item = re.sub(r"[^a-z0-9_]", "", item.replace(" ", "_"))
        add_keyword(keywords, item)
    return keywords[:6]


def extract_section_thinking(output: str) -> str:
    match = re.search(
        r"##\s*Thinking Process\s*(.*?)(?=\n##\s*Final Answer|\n##\s*RAG Keywords|\n```|\Z)",
        output or "",
        re.I | re.S,
    )
    if not match:
        return ""

    body = match.group(1)
    labels = [
        "Problem Understanding",
        "Input/Output Analysis",
        "Core Concept",
        "Solving Strategy",
        "Implementation Plan",
    ]
    lines = ["## Thinking Process"]
    for index, label in enumerate(labels, start=1):
        pattern = rf"{index}\.\s*\**{re.escape(label)}\**\s*:?\s*(.*?)(?=\n\s*{index + 1}\.\s*\**|\n\s*##|\Z)"
        item_match = re.search(pattern, body, re.I | re.S)
        if not item_match:
            return ""
        sentence = clean_inline_text(item_match.group(1).splitlines()[0])
        if not sentence:
            return ""
        lines.append(f"{index}. {label}: {sentence}")
    return "\n".join(lines)


def extract_think_text(output: str) -> str:
    match = re.search(r"<think>\s*(.*?)\s*</think>", output or "", re.I | re.S)
    if match:
        return match.group(1).strip()
    return ""


def split_sentences(text: str) -> List[str]:
    text = re.sub(r"```.*?```", " ", text or "", flags=re.S)
    text = re.sub(r"\s+", " ", text).strip()
    pieces = re.split(r"(?<=[.!?])\s+", text)
    sentences: List[str] = []
    for piece in pieces:
        sentence = clean_inline_text(piece)
        words = re.findall(r"[A-Za-z0-9']+", sentence)
        lower = sentence.lower()
        if not 7 <= len(words) <= 28:
            continue
        if lower.startswith(("okay", "hmm", "wait", "let me", "let's")):
            continue
        if "example" in lower or "sample" in lower:
            continue
        sentences.append(sentence)
    return sentences


def pick_sentence(sentences: Sequence[str], keywords: Sequence[str], fallback: str) -> str:
    for sentence in sentences:
        lower = sentence.lower()
        if any(keyword in lower for keyword in keywords):
            return sentence
    return fallback


def fallback_problem_sentence(example: Dict[str, Any], keywords: Sequence[str]) -> str:
    first_line = re.split(r"\n\s*Input\s*\n|\n\s*Constraints\s*\n", problem_text(example), flags=re.I)[0]
    first_line = clean_inline_text(first_line)
    words = re.findall(r"[A-Za-z0-9']+", first_line)
    if 7 <= len(words) <= 28:
        return first_line
    core = keywords[0] if keywords else "implementation"
    return f"The task should be solved using {core.replace('_', ' ')} to produce the required result."


def thinking_from_reasoning(example: Dict[str, Any], output: str, keywords: Sequence[str]) -> str:
    reasoning = extract_think_text(output)
    if not reasoning:
        return ""

    sentences = split_sentences(reasoning)
    core = keywords[0] if keywords else "implementation"
    problem = pick_sentence(
        sentences,
        ("task", "problem", "need to", "have to", "goal", "determine", "compute", "draw", "find"),
        fallback_problem_sentence(example, keywords),
    )
    io = pick_sentence(
        sentences,
        ("input", "output", "read", "print", "given"),
        "Read the given values from standard input and print only the required output.",
    )
    concept = pick_sentence(
        sentences,
        ("key", "concept", "main", "approach", "use", core.replace("_", " ")),
        concept_sentence(core),
    )
    strategy = pick_sentence(
        sentences,
        ("so", "therefore", "approach", "check", "iterate", "sort", "traverse", "count", "compute"),
        strategy_sentence(keywords),
    )
    plan = pick_sentence(
        sentences,
        ("code", "implement", "loop", "read", "then", "finally", "print"),
        "Parse the input, apply the selected method, and print the computed answer.",
    )

    return "\n".join(
        [
            "## Thinking Process",
            f"1. Problem Understanding: {problem}",
            f"2. Input/Output Analysis: {io}",
            f"3. Core Concept: {concept}",
            f"4. Solving Strategy: {strategy}",
            f"5. Implementation Plan: {plan}",
        ]
    )


def add_keyword(out: List[str], keyword: str) -> None:
    if keyword in ALLOWED_VOCAB and keyword not in out:
        out.append(keyword)


def has_any(text: str, patterns: Sequence[str]) -> bool:
    for pattern in patterns:
        if re.search(r"(?<![a-z0-9_])" + re.escape(pattern) + r"(?![a-z0-9_])", text):
            return True
    return False


def choose_keywords(example: Dict[str, Any], code: str) -> List[str]:
    text = (problem_text(example) + "\n" + code).lower()
    keywords: List[str] = []

    if has_any(text, ("graph", "node", "nodes", "edge", "edges", "adjacency", "tree")):
        add_keyword(keywords, "graph")
        if has_any(text, ("tree",)):
            add_keyword(keywords, "tree")
        add_keyword(keywords, "adjacency_list")
    if has_any(text, ("bfs", "breadth", "deque")):
        add_keyword(keywords, "bfs")
        add_keyword(keywords, "queue")
    if has_any(text, ("dfs", "depth", "recursive dfs")):
        add_keyword(keywords, "dfs")
        add_keyword(keywords, "visited")
    if has_any(text, ("shortest path", "dijkstra")):
        add_keyword(keywords, "shortest_path")
        add_keyword(keywords, "dijkstra")
    if has_any(text, ("union find", "disjoint", "dsu")):
        add_keyword(keywords, "union_find")
    if has_any(text, ("dynamic programming", "dp", "memo")):
        add_keyword(keywords, "dynamic_programming")
        add_keyword(keywords, "memoization")
    if has_any(text, ("binary search", "bisect", "lower_bound", "upper_bound")):
        add_keyword(keywords, "binary_search")
    if has_any(text, ("prefix sum", "cumulative")):
        add_keyword(keywords, "prefix_sum")
    if has_any(text, ("two pointer", "two-pointer")):
        add_keyword(keywords, "two_pointers")
    if has_any(text, ("sliding window", "window")):
        add_keyword(keywords, "sliding_window")
    if has_any(text, ("sort", "sorted")):
        add_keyword(keywords, "sorting")
    if has_any(text, ("heap", "priority queue", "heapq")):
        add_keyword(keywords, "heap")
    if has_any(text, ("grid", "matrix", "board", "cell", "row", "column", "chessboard")):
        add_keyword(keywords, "grid")
    if has_any(text, ("string", "substring", "palindrome", "character")):
        add_keyword(keywords, "string_processing")
    if "%" in text or has_any(text, ("mod", "modulo", "divisible", "remainder")):
        add_keyword(keywords, "modulo")
        add_keyword(keywords, "math")
    if has_any(text, ("count", "counts", "frequency")):
        add_keyword(keywords, "counting")
        add_keyword(keywords, "frequency")
    if has_any(text, ("recursion", "recursive")):
        add_keyword(keywords, "recursion")
    if has_any(text, ("greedy", "minimum", "maximum", "minimize", "maximize")):
        add_keyword(keywords, "greedy")
        add_keyword(keywords, "optimization")
    if has_any(text, ("brute force", "try all", "enumerate all")):
        add_keyword(keywords, "brute_force")

    for fallback in ("implementation", "parsing", "loop", "condition", "array", "simulation"):
        add_keyword(keywords, fallback)
        if len(keywords) >= 6:
            break
    return keywords[:6]


def keyword_target(keywords: Sequence[str]) -> str:
    items = list(keywords)[:6]
    if len(items) < 6:
        items = [item.strip() for item in SAFE_KEYWORDS.split("|")]
    return "## RAG Keywords\n" + " | ".join(items[:6])


def concept_sentence(keyword: str) -> str:
    sentences = {
        "grid": "The main concept is grid traversal with row and column checks.",
        "graph": "The main concept is graph representation and visiting connected nodes.",
        "tree": "The main concept is tree traversal without revisiting the parent node.",
        "bfs": "The main concept is breadth first search using a queue.",
        "dfs": "The main concept is depth first search using recursive or stack-based traversal.",
        "sorting": "The main concept is sorting values before applying the required rule.",
        "binary_search": "The main concept is binary search over a sorted range or answer space.",
        "dynamic_programming": "The main concept is dynamic programming with reusable subproblem results.",
        "greedy": "The main concept is making the best local choice at each step.",
        "modulo": "The main concept is using remainders to handle repeated patterns.",
        "string_processing": "The main concept is processing characters and substrings carefully.",
        "prefix_sum": "The main concept is prefix sums for fast range calculation.",
        "two_pointers": "The main concept is moving two indices to scan the data efficiently.",
        "heap": "The main concept is using a heap to repeatedly access the smallest or largest value.",
        "counting": "The main concept is counting occurrences needed for the final answer.",
    }
    return sentences.get(keyword, f"The main concept is {keyword.replace('_', ' ')} with careful implementation.")


def strategy_sentence(keywords: Sequence[str]) -> str:
    keyset = set(keywords)
    if {"graph", "tree"} & keyset:
        return "Build the relationships first, then traverse the nodes to compute the needed values."
    if "grid" in keyset:
        return "Visit each cell or position and decide the result from its coordinates."
    if "sorting" in keyset:
        return "Sort the data first, then scan it to apply the rule efficiently."
    if "dynamic_programming" in keyset:
        return "Define a state, reuse previous results, and build the answer step by step."
    if "binary_search" in keyset:
        return "Check whether a candidate value works, then narrow the search range."
    if "modulo" in keyset:
        return "Use division and remainder patterns to avoid unnecessary repeated work."
    return "Process the input step by step and update the values needed for the answer."


def thinking_target(keywords: Sequence[str]) -> str:
    core = keywords[0] if keywords else "implementation"
    return "\n".join(
        [
            "## Thinking Process",
            f"1. Problem Understanding: The task should be solved using {core.replace('_', ' ')} to produce the required result.",
            "2. Input/Output Analysis: Read the given values from standard input and print only the required output.",
            f"3. Core Concept: {concept_sentence(core)}",
            f"4. Solving Strategy: {strategy_sentence(keywords)}",
            "5. Implementation Plan: Parse the input, apply the selected method, and print the computed answer.",
        ]
    )


def code_target(code: str) -> str:
    return f"## Final Answer\n```python\n{code.strip()}\n```"


def full_target(keywords: Sequence[str], thinking: str, code: str) -> str:
    return "\n\n".join([keyword_target(keywords), thinking, code_target(code)])


def system_prompt_for(task: str) -> str:
    if task == "keyword":
        return KEYWORD_SYSTEM_PROMPT
    if task == "thinking":
        return THINKING_SYSTEM_PROMPT
    if task == "code":
        return CODE_SYSTEM_PROMPT
    if task == "full":
        return FULL_SYSTEM_PROMPT
    raise ValueError(f"Unknown task: {task}")


def user_prompt_for(task: str, example: Dict[str, Any]) -> str:
    if task == "keyword":
        return (
            "Return only RAG Keywords for this coding problem.\n"
            "Do not write Analysis, Problem Understanding, Core Concept, Thinking Process, or Final Answer.\n"
            "Your answer must be exactly two lines.\n\n"
            "Coding Problem:\n"
            f"{problem_text(example)}"
        )

    label = {
        "thinking": "Return only Thinking Process for this coding problem.",
        "code": "Return only the final Python answer for this coding problem.",
        "full": "Return the full tutoring answer for this coding problem.",
    }[task]
    return f"{label}\n\nCoding Problem:\n{problem_text(example)}"


def make_split_examples(example: Dict[str, Any], tasks: Iterable[str], include_full: bool) -> List[Dict[str, str]]:
    output = raw_output(example)
    code = extract_code(output)
    if not code:
        return []

    keywords = extract_rag_keywords(output) or choose_keywords(example, code)
    thinking = extract_section_thinking(output) or thinking_from_reasoning(example, output, keywords) or thinking_target(keywords)
    targets = {
        "keyword": keyword_target(keywords),
        "thinking": thinking,
        "code": code_target(code),
    }
    if include_full:
        targets["full"] = full_target(keywords, thinking, code)

    examples = []
    for task in tasks:
        if task not in targets:
            continue
        examples.append(
            {
                "task": task,
                "system": system_prompt_for(task),
                "user": user_prompt_for(task, example),
                "assistant": targets[task],
            }
        )
    return examples


def build_prompt(tokenizer: Any, row: Dict[str, str]) -> str:
    messages = [
        {"role": "system", "content": row["system"]},
        {"role": "user", "content": row["user"]},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def tokenize_example(row: Dict[str, str], tokenizer: Any, max_length: int) -> Dict[str, List[int]]:
    prompt_ids = tokenizer(build_prompt(tokenizer, row), add_special_tokens=False)["input_ids"]
    response_ids = tokenizer(row["assistant"] + tokenizer.eos_token, add_special_tokens=False)["input_ids"]

    input_ids = prompt_ids + response_ids
    labels = [-100] * len(prompt_ids) + response_ids

    if len(input_ids) > max_length:
        overflow = len(input_ids) - max_length
        input_ids = input_ids[overflow:]
        labels = labels[overflow:]

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


@dataclass
class CausalLMCollator:
    tokenizer: Any

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        import torch

        labels = [feature["labels"] for feature in features]
        batch = self.tokenizer.pad(
            {
                "input_ids": [feature["input_ids"] for feature in features],
                "attention_mask": [feature["attention_mask"] for feature in features],
            },
            padding=True,
            return_tensors="pt",
        )
        max_len = batch["input_ids"].shape[1]
        batch["labels"] = torch.tensor(
            [label + [-100] * (max_len - len(label)) for label in labels],
            dtype=torch.long,
        )
        return batch


def create_lora_model(base_model: Any, config: TrainConfig, device: Any) -> Any:
    from peft import LoraConfig, TaskType, get_peft_model

    if hasattr(base_model, "enable_input_require_grads"):
        base_model.enable_input_require_grads()

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(base_model, lora_config)
    if not hasattr(base_model, "hf_device_map"):
        model.to(device)
    model.print_trainable_parameters()
    return model


def load_and_expand_dataset(config: TrainConfig) -> Any:
    from datasets import Dataset, load_dataset

    train_path = Path(config.train_file)
    if not train_path.exists():
        raise FileNotFoundError(f"Training file not found: {train_path}")

    raw_dataset = load_dataset("json", data_files=str(train_path), split="train")
    if config.limit > 0:
        raw_dataset = raw_dataset.select(range(min(config.limit, len(raw_dataset))))

    tasks = [task.strip() for task in config.task_mix.split(",") if task.strip()]
    if config.include_full and "full" not in tasks:
        tasks.append("full")

    rows: List[Dict[str, str]] = []
    for example in raw_dataset:
        rows.extend(make_split_examples(example, tasks, config.include_full))

    if not rows:
        raise RuntimeError("No trainable rows were created. Check that outputs contain code blocks.")

    print("Raw rows:", len(raw_dataset))
    print("Expanded train rows:", len(rows))
    print("Task counts:")
    for task in sorted(set(row["task"] for row in rows)):
        print(f"- {task}: {sum(row['task'] == task for row in rows)}")
    return Dataset.from_list(rows)


def print_sample(dataset: Any, tokenizer: Any, config: TrainConfig) -> None:
    row = dataset[0]
    tokenized = tokenize_example(row, tokenizer, config.max_length)
    trained_labels = [token_id for token_id in tokenized["labels"] if token_id != -100]
    print("\n========== Training Sample ==========")
    print("task:", row["task"])
    print(tokenizer.decode(tokenized["input_ids"], skip_special_tokens=False)[:2500])
    print("\n========== Label Preview ==========")
    print(tokenizer.decode(trained_labels, skip_special_tokens=False)[:1200])
    print("\ninput length:", len(tokenized["input_ids"]))
    print("label tokens:", len(trained_labels))
    print("=====================================\n")


def build_training_args(config: TrainConfig, device: Any) -> Any:
    from transformers import TrainingArguments

    return TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=2,
        fp16=(device.type == "cuda"),
        bf16=False,
        optim="adamw_torch",
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=(device.type == "cuda"),
        dataloader_num_workers=0,
        gradient_checkpointing=True,
    )


def main() -> None:
    config = parse_args()
    verify_training_dependencies()

    from qwen_base_model import BaseModelConfig, load_base_model_and_tokenizer, select_device
    from transformers import Trainer

    base_config = BaseModelConfig(
        model_name=config.base_model_name,
        cuda_device=config.cuda_device,
        require_cuda=config.require_cuda,
    )
    device = select_device(base_config)
    base_model, tokenizer = load_base_model_and_tokenizer(base_config, device)
    model = create_lora_model(base_model, config, device)

    dataset = load_and_expand_dataset(config)
    print_sample(dataset, tokenizer, config)

    tokenized_dataset = dataset.map(
        lambda row: tokenize_example(row, tokenizer, config.max_length),
        remove_columns=dataset.column_names,
        desc="Tokenizing split LoRA dataset",
    ).filter(lambda row: any(label != -100 for label in row["labels"]))

    trainer = Trainer(
        model=model,
        args=build_training_args(config, device),
        train_dataset=tokenized_dataset,
        data_collator=CausalLMCollator(tokenizer),
    )

    trainer.train()
    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    print(f"LoRA adapter saved: {config.output_dir}")


if __name__ == "__main__":
    main()