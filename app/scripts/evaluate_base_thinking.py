from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from eval_common import (
    DEFAULT_OUTPUT_ROOT,
    BaseModelRunner,
    add_common_args,
    build_problem_text,
    is_english_only,
    rate,
    read_dataset,
    save_outputs,
)

SYSTEM_PROMPT = """
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


@dataclass
class ThinkingRow:
    sample_index: int
    repeat_index: int
    api_success: bool
    thinking_ok: bool
    labels_ok: bool
    english_only: bool
    no_bold: bool
    no_extra_sections: bool
    no_code_block: bool
    concise_ok: bool
    generated_tokens: int
    truncated: bool
    generation_seconds: float
    generation_time_ok: bool
    errors: str
    response: str


def evaluate_response(
    sample_index: int,
    repeat_index: int,
    response: str,
    tokens: int,
    truncated: bool,
    seconds: float,
    max_seconds: float,
) -> ThinkingRow:
    lower = response.lower()
    labels_ok = all(label.lower() in lower for label in REQUIRED_LABELS)
    english_only = is_english_only(response)
    no_bold = "**" not in response
    no_extra = not re.search(r"Verification Points|Complexity|Notes|Examples|Test Cases|## Final Answer|## RAG Keywords", response, re.I)
    no_code = "```" not in response
    words = re.findall(r"[A-Za-z0-9']+", response)
    concise_ok = 30 <= len(words) <= 250
    api_success = bool(response)
    time_ok = seconds <= max_seconds
    thinking_ok = labels_ok and english_only and no_bold and no_extra and no_code and concise_ok and not truncated
    checks = {
        "api_failed": not api_success,
        "thinking_failed": not thinking_ok,
        "label_failed": not labels_ok,
        "non_english": not english_only,
        "bold_found": not no_bold,
        "extra_section_found": not no_extra,
        "code_block_found": not no_code,
        "not_concise": not concise_ok,
        "truncated": truncated,
        "generation_time_failed": not time_ok,
    }
    return ThinkingRow(
        sample_index=sample_index,
        repeat_index=repeat_index,
        api_success=api_success,
        thinking_ok=thinking_ok,
        labels_ok=labels_ok,
        english_only=english_only,
        no_bold=no_bold,
        no_extra_sections=no_extra,
        no_code_block=no_code,
        concise_ok=concise_ok,
        generated_tokens=tokens,
        truncated=truncated,
        generation_seconds=seconds,
        generation_time_ok=time_ok,
        errors=";".join(name for name, failed in checks.items() if failed),
        response=response,
    )


def summarize(rows: List[ThinkingRow]) -> Dict[str, Any]:
    return {
        "total_generations": len(rows),
        "thinking_quality": rate(sum(r.thinking_ok for r in rows), len(rows)),
        "label_accracy": rate(sum(r.labels_ok for r in rows), len(rows)),
        "english_only_rate": rate(sum(r.english_only for r in rows), len(rows)),
        "no_bold_rate": rate(sum(r.no_bold for r in rows), len(rows)),
        "no_extra_section_rate": rate(sum(r.no_extra_sections for r in rows), len(rows)),
        "concise_rate": rate(sum(r.concise_ok for r in rows), len(rows)),
        "not_truncated_rate": rate(sum(not r.truncated for r in rows), len(rows)),
        "generation_time_compliance": rate(sum(r.generation_time_ok for r in rows), len(rows)),
        "avg_generation_seconds": round(sum(r.generation_seconds for r in rows) / len(rows), 4) if rows else 0,
        "max_generation_seconds": round(max((r.generation_seconds for r in rows), default=0), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate BASE Thinking Process only.")
    add_common_args(parser, DEFAULT_OUTPUT_ROOT / "thinking", max_new_tokens=384)
    args = parser.parse_args()

    samples = read_dataset(Path(args.dataset))
    if args.limit > 0:
        samples = samples[: args.limit]

    runner = BaseModelRunner(args.model_name, args.max_new_tokens, args.device_map)
    rows: List[ThinkingRow] = []
    for sample_index, sample in enumerate(samples):
        print(f"[thinking {sample_index + 1}/{len(samples)}] generating")
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
    print("\nBASE THINKING EVALUATION")
    for key, value in summary.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    main()
