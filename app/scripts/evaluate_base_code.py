from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from eval_common import (
    DEFAULT_OUTPUT_ROOT,
    BaseModelRunner,
    add_common_args,
    build_problem_text,
    one_python_code_block,
    rate,
    read_dataset,
    save_outputs,
)


SYSTEM_PROMPT = """
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
class CodeRow:
    sample_index: int
    repeat_index: int
    api_success: bool
    code_ok: bool
    strict_code_ok: bool
    relaxed_code_ok: bool
    starts_with_final_answer: bool
    one_code_block: bool
    syntax_ok: bool
    no_placeholder: bool
    no_example_usage: bool
    ends_after_code_block: bool
    generated_tokens: int
    truncated: bool
    generation_seconds: float
    generation_time_ok: bool
    syntax_message: str
    errors: str
    response: str


def syntax_check(code: str) -> tuple[bool, str]:
    try:
        ast.parse(code)
    except SyntaxError as e:
        return False, f"line {e.lineno}: {e.msg}"
    return True, "ok"


def evaluate_response(
    sample_index: int,
    repeat_index: int,
    response: str,
    tokens: int,
    truncated: bool,
    seconds: float,
    max_seconds: float,
) -> CodeRow:
    blocks = one_python_code_block(response)
    starts = response.startswith("## Final Answer")
    one_block = len(blocks) == 1
    syntax_ok, syntax_message = syntax_check(blocks[0]) if one_block else (False, "expected exactly one python code block")
    no_placeholder = "write final code here" not in response.lower() and "complete_python_solution" not in response
    no_example = "# Example usage" not in response
    ends_after = response.rstrip().endswith("```")
    api_success = bool(response)
    time_ok = seconds <= max_seconds
    strict_code_ok = starts and one_block and syntax_ok and no_placeholder and no_example and ends_after and not truncated
    relaxed_code_ok = one_block and syntax_ok and no_placeholder and not truncated
    code_ok = relaxed_code_ok
    checks = {
        "api_failed": not api_success,
        "code_failed": not code_ok,
        "strict_code_failed": not strict_code_ok,
        "relaxed_code_failed": not relaxed_code_ok,
        "missing_final_answer_header": not starts,
        "code_block_count_failed": not one_block,
        "syntax_failed": not syntax_ok,
        "placeholder_found": not no_placeholder,
        "example_usage_found": not no_example,
        "text_after_code_block": not ends_after,
        "truncated": truncated,
        "generation_time_failed": not time_ok,
    }
    return CodeRow(
        sample_index=sample_index,
        repeat_index=repeat_index,
        api_success=api_success,
        code_ok=code_ok,
        strict_code_ok=strict_code_ok,
        relaxed_code_ok=relaxed_code_ok,
        starts_with_final_answer=starts,
        one_code_block=one_block,
        syntax_ok=syntax_ok,
        no_placeholder=no_placeholder,
        no_example_usage=no_example,
        ends_after_code_block=ends_after,
        generated_tokens=tokens,
        truncated=truncated,
        generation_seconds=seconds,
        generation_time_ok=time_ok,
        syntax_message=syntax_message,
        errors=";".join(name for name, failed in checks.items() if failed),
        response=response,
    )


def summarize(rows: List[CodeRow]) -> Dict[str, Any]:
    return {
        "total_generations": len(rows),
        "code_quality": rate(sum(r.code_ok for r in rows), len(rows)),
        "strict_code_quality": rate(sum(r.strict_code_ok for r in rows), len(rows)),
        "relaxed_code_quality": rate(sum(r.relaxed_code_ok for r in rows), len(rows)),
        "final_answer_header_rate": rate(sum(r.starts_with_final_answer for r in rows), len(rows)),
        "one_code_block_rate": rate(sum(r.one_code_block for r in rows), len(rows)),
        "syntax_pass_rate": rate(sum(r.syntax_ok for r in rows), len(rows)),
        "no_placeholder_rate": rate(sum(r.no_placeholder for r in rows), len(rows)),
        "no_example_usage_rate": rate(sum(r.no_example_usage for r in rows), len(rows)),
        "ends_after_code_block_rate": rate(sum(r.ends_after_code_block for r in rows), len(rows)),
        "not_truncated_rate": rate(sum(not r.truncated for r in rows), len(rows)),
        "generation_time_compliance": rate(sum(r.generation_time_ok for r in rows), len(rows)),
        "avg_generation_seconds": round(sum(r.generation_seconds for r in rows) / len(rows), 4) if rows else 0,
        "max_generation_seconds": round(max((r.generation_seconds for r in rows), default=0), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate BASE final code only.")
    add_common_args(parser, DEFAULT_OUTPUT_ROOT / "code", max_new_tokens=512)
    args = parser.parse_args()

    samples = read_dataset(Path(args.dataset))
    if args.limit > 0:
        samples = samples[: args.limit]

    runner = BaseModelRunner(args.model_name, args.max_new_tokens, args.device_map)
    rows: List[CodeRow] = []
    for sample_index, sample in enumerate(samples):
        print(f"[code {sample_index + 1}/{len(samples)}] generating")
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
    print("\nBASE CODE EVALUATION")
    for key, value in summary.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    main()
