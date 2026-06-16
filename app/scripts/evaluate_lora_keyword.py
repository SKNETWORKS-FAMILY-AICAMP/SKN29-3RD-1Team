from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import evaluate_base_keyword as base
from eval_common import (
    DEFAULT_KEYWORD_ADAPTER,
    DEFAULT_LORA_OUTPUT_ROOT,
    LoraModelRunner,
    add_lora_args,
    build_problem_text,
    read_dataset,
    save_outputs,
)


def build_keyword_prompt(problem_text: str) -> str:
    return (
        "Return only RAG Keywords for this coding problem.\n"
        "Do not write Analysis, Problem Understanding, Core Concept, Thinking Process, or Final Answer.\n"
        "Your answer must be exactly two lines.\n\n"
        "Coding Problem:\n"
        f"{problem_text}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate LoRA keyword generation only.")
    add_lora_args(parser, DEFAULT_LORA_OUTPUT_ROOT / "keyword", max_new_tokens=96, adapter_path=DEFAULT_KEYWORD_ADAPTER)
    args = parser.parse_args()

    samples = read_dataset(Path(args.dataset))
    if args.limit > 0:
        samples = samples[: args.limit]

    runner = LoraModelRunner(args.model_name, args.adapter_path, args.max_new_tokens, args.device_map)
    rows: List[base.KeywordRow] = []
    for sample_index, sample in enumerate(samples):
        print(f"[lora keyword {sample_index + 1}/{len(samples)}] generating")
        problem_text = build_keyword_prompt(build_problem_text(sample))
        for repeat_index in range(args.repeat):
            try:
                response, tokens, truncated, seconds = runner.generate(
                    base.SYSTEM_PROMPT, problem_text, args.do_sample, args.temperature, args.top_p
                )
            except Exception as e:
                response, tokens, truncated, seconds = f"[GENERATION_ERROR] {e}", 0, False, 0.0
            rows.append(base.evaluate_response(sample_index, repeat_index, response, tokens, truncated, seconds, args.max_generation_seconds))

    summary = base.summarize(rows)
    save_outputs(Path(args.output_dir), "summary.json", "details.csv", summary, rows)
    print("\nLORA KEYWORD EVALUATION")
    for key, value in summary.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    main()
