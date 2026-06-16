from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from eval_common import (
    ALLOWED_VOCAB,
    DEFAULT_OUTPUT_ROOT,
    SAFE_KEYWORDS,
    BaseModelRunner,
    add_common_args,
    build_problem_text,
    keyword_list,
    rate,
    read_dataset,
    save_outputs,
)

SYSTEM_PROMPT = f"""
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

@dataclass
class KeywordRow:
    sample_index: int
    repeat_index: int
    api_success: bool
    keyword_ok: bool
    normalized_keyword_ok: bool
    exact_6: bool
    normalized_exact_6: bool
    unique: bool
    snake_case: bool
    allowed_vocab_ok: bool
    normalized_allowed_vocab_ok: bool
    allowed_vocab_count: int
    invalid_keywords: str
    normalized_rag_keywords: str
    generated_tokens: int
    truncated: bool
    generation_seconds: float
    generation_time_ok: bool
    rag_keywords: str
    errors: str
    response: str


KEYWORD_ALIASES = {
    "array_manipulation": "array",
    "conditional_statements": "condition",
    "depth_first_search": "dfs",
    "breadth_first_search": "bfs",
    "greedy_algorithm": "greedy",
    "graph_traversal": "graph",
    "integer_arithmetic": "math",
    "input_formatting": "parsing",
    "loop_control": "loop",
    "modular_arithmetic": "modulo",
    "recursion_function": "recursive_function",
    "sorting_algorithm": "sorting",
    "string_handling": "string_processing",
}


def clean_keyword(keyword: str) -> str:
    cleaned = re.sub(r"[^a-z0-9_]+", "_", keyword.strip().lower())
    return re.sub(r"_+", "_", cleaned).strip("_")


def safe_keyword_list() -> List[str]:
    return [item.strip() for item in SAFE_KEYWORDS.split("|") if item.strip()]


def normalize_keywords(keywords: List[str]) -> List[str]:
    normalized: List[str] = []
    for keyword in keywords:
        cleaned = clean_keyword(keyword)
        candidate = cleaned if cleaned in ALLOWED_VOCAB else KEYWORD_ALIASES.get(cleaned, "")
        if candidate in ALLOWED_VOCAB and candidate not in normalized:
            normalized.append(candidate)
        if len(normalized) == 6:
            return normalized

    for fallback in safe_keyword_list():
        if fallback in ALLOWED_VOCAB and fallback not in normalized:
            normalized.append(fallback)
        if len(normalized) == 6:
            break
    return normalized


def evaluate_response(
    sample_index: int,
    repeat_index: int,
    response: str,
    tokens: int,
    truncated: bool,
    seconds: float,
    max_seconds: float,
) -> KeywordRow:
    keywords = keyword_list(response)
    normalized_keywords = normalize_keywords(keywords) if keywords else []
    exact_6 = len(keywords) == 6
    normalized_exact_6 = len(normalized_keywords) == 6
    unique = len(set(keywords)) == len(keywords)
    snake_case = all(re.fullmatch(r"[a-z0-9][a-z0-9_]{1,40}", k or "") for k in keywords)
    invalid = [k for k in keywords if k not in ALLOWED_VOCAB]
    allowed_vocab_count = sum(k in ALLOWED_VOCAB for k in keywords)
    allowed_vocab_ok = exact_6 and allowed_vocab_count >= 3
    normalized_allowed_vocab_ok = normalized_exact_6 and all(k in ALLOWED_VOCAB for k in normalized_keywords)
    keyword_ok = exact_6 and unique and snake_case and allowed_vocab_ok
    normalized_keyword_ok = normalized_exact_6 and len(set(normalized_keywords)) == 6 and normalized_allowed_vocab_ok
    api_success = bool(response)
    
    time_ok = seconds <= max_seconds
    checks = {
        "api_failed": not api_success,
        "keyword_failed": not keyword_ok,
        "normalized_keyword_failed": not normalized_keyword_ok,
        "not_exact_6": not exact_6,
        "normalized_not_exact_6": not normalized_exact_6,
        "not_unique": not unique,
        "not_snake_case": not snake_case,
        "invalid_vocab": not allowed_vocab_ok,
        "normalized_invalid_vocab": not normalized_allowed_vocab_ok,
        "truncated": truncated,
        "generation_time_failed": not time_ok,
    }
    return KeywordRow(
        sample_index=sample_index,
        repeat_index=repeat_index,
        api_success=api_success,
        keyword_ok=keyword_ok,
        normalized_keyword_ok=normalized_keyword_ok,
        exact_6=exact_6,
        normalized_exact_6=normalized_exact_6,
        unique=unique,
        snake_case=snake_case,
        allowed_vocab_ok=allowed_vocab_ok,
        normalized_allowed_vocab_ok=normalized_allowed_vocab_ok,
        allowed_vocab_count=sum(k in ALLOWED_VOCAB for k in keywords),
        invalid_keywords=" | ".join(invalid),
        normalized_rag_keywords=" | ".join(normalized_keywords),
        generated_tokens=tokens,
        truncated=truncated,
        generation_seconds=seconds,
        generation_time_ok=time_ok,
        rag_keywords=" | ".join(keywords),
        errors=";".join(name for name, failed in checks.items() if failed),
        response=response,
    )


def summarize(rows: List[KeywordRow]) -> Dict[str, Any]:
    return {
        "total_generations": len(rows),
        "keyword_accuracy": rate(sum(r.keyword_ok for r in rows), len(rows)),
        "normalized_keyword_accuracy": rate(sum(r.normalized_keyword_ok for r in rows), len(rows)),
        "rag_exact_6_rate": rate(sum(r.exact_6 for r in rows), len(rows)),
        "normalized_exact_6_rate": rate(sum(r.normalized_exact_6 for r in rows), len(rows)),
        "rag_allowed_vocabulary_accuracy": rate(sum(r.allowed_vocab_ok for r in rows), len(rows)),
        "normalized_allowed_vocabulary_accuracy": rate(sum(r.normalized_allowed_vocab_ok for r in rows), len(rows)),
        "rag_unique_rate": rate(sum(r.unique for r in rows), len(rows)),
        "rag_snake_case_rate": rate(sum(r.snake_case for r in rows), len(rows)),
        "not_truncated_rate": rate(sum(not r.truncated for r in rows), len(rows)),
        "generation_time_compliance": rate(sum(r.generation_time_ok for r in rows), len(rows)),
        "avg_generation_seconds": round(sum(r.generation_seconds for r in rows) / len(rows), 4) if rows else 0,
        "max_generation_seconds": round(max((r.generation_seconds for r in rows), default=0), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate BASE keyword generation only.")
    add_common_args(parser, DEFAULT_OUTPUT_ROOT / "keyword", max_new_tokens=48)
    args = parser.parse_args()

    samples = read_dataset(Path(args.dataset))
    if args.limit > 0:
        samples = samples[: args.limit]

    runner = BaseModelRunner(args.model_name, args.max_new_tokens, args.device_map)
    rows: List[KeywordRow] = []
    for sample_index, sample in enumerate(samples):
        print(f"[keyword {sample_index + 1}/{len(samples)}] generating")
        problem_text = build_problem_text(sample)
        for repeat_index in range(args.repeat):
            try:
                response, tokens, truncated, seconds = runner.generate(
                    SYSTEM_PROMPT, problem_text, args.do_sample, args.temperature, args.top_p
                )
            except Exception as e:
                response, tokens, truncated, seconds = f"[GENERATION_ERROR] {e}", 0, False, 0.0
            rows.append(evaluate_response(sample_index, repeat_index, response, tokens, truncated, seconds, args.max_generation_seconds))

    summary = summarize(rows)
    save_outputs(Path(args.output_dir), "summary.json", "details.csv", summary, rows)
    print("\nBASE KEYWORD EVALUATION")
    for key, value in summary.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    main()
