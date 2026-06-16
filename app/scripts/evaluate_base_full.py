from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

from eval_common import (
    ALLOWED_VOCAB,
    DEFAULT_OUTPUT_ROOT,
    is_english_only,
    keyword_list,
    one_python_code_block,
    rate,
    save_outputs,
)


REQUIRED_LABELS = [
    "## RAG Keywords",
    "## Thinking Process",
    "1. Problem Understanding:",
    "2. Input/Output Analysis:",
    "3. Core Concept:",
    "4. Solving Strategy:",
    "5. Implementation Plan:",
    "## Final Answer",
]

@dataclass
class FullRow:
    sample_index: int
    repeat_index: int
    api_success: bool
    format_ok: bool
    relaxed_format_ok: bool
    usable_response_ok: bool
    baseline_50_ok: bool
    baseline_50_score: float
    keyword_ok: bool
    keyword_usable: bool
    thinking_label_ok: bool
    thinking_usable: bool
    code_ok: bool
    strict_code_ok: bool
    relaxed_code_ok: bool
    code_block_ok: bool
    explanation_ok: bool
    length_ok: bool
    truncated: bool
    stable_ok: bool
    generated_tokens: int
    response_chars: int
    generation_seconds: float
    generation_time_ok: bool
    rag_keywords: str
    rag_keyword_count: int
    rag_allowed_vocab_count: int
    errors: str
    response: str


def as_bool(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def as_int(value: Any) -> int:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return 0


def as_float(value: Any) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return 0.0


def load_details(path: Path) -> Dict[Tuple[int, int], Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing split evaluation file: {path}")

    rows: Dict[Tuple[int, int], Dict[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (as_int(row.get("sample_index")), as_int(row.get("repeat_index")))
            rows[key] = row
    return rows


def keyword_section(row: Dict[str, str]) -> str:
    keywords = (row.get("rag_keywords") or "").strip()
    if keywords:
        return f"## RAG Keywords\n{keywords}"
    return (row.get("response") or "").strip()


def text_without_code_blocks(response: str) -> str:
    return re.sub(r"```python\s*\n.*?\n```", "", response or "", flags=re.I | re.S)


def join_response(keyword_row: Dict[str, str], thinking_row: Dict[str, str], code_row: Dict[str, str]) -> str:
    parts = [
        keyword_section(keyword_row),
        (thinking_row.get("response") or "").strip(),
        (code_row.get("response") or "").strip(),
    ]
    return "\n\n".join(part for part in parts if part)


def evaluate_joined_response(
    sample_index: int,
    repeat_index: int,
    keyword_row: Dict[str, str],
    thinking_row: Dict[str, str],
    code_row: Dict[str, str],
    max_seconds: float,
    max_tokens: int,
) -> FullRow:
    response = join_response(keyword_row, thinking_row, code_row)
    lower = response.lower()
    keywords = keyword_list(response)

    keyword_ok = as_bool(keyword_row.get("keyword_ok"))
    keyword_usable = len(keywords) == 6 and len(set(keywords)) == 6 and sum(k in ALLOWED_VOCAB for k in keywords) >= 3
    thinking_ok = as_bool(thinking_row.get("thinking_ok"))
    thinking_label_ok = as_bool(thinking_row.get("labels_ok")) and all(label.lower() in lower for label in REQUIRED_LABELS)
    thinking_usable = thinking_label_ok and as_bool(thinking_row.get("english_only", True))
    strict_code_ok = as_bool(code_row.get("strict_code_ok", code_row.get("code_ok")))
    relaxed_code_value = code_row.get("relaxed_code_ok")
    relaxed_code_ok = as_bool(relaxed_code_value) if relaxed_code_value not in (None, "") else as_bool(code_row.get("code_ok"))
    code_ok = relaxed_code_ok
    code_block_ok = as_bool(code_row.get("one_code_block")) and len(one_python_code_block(response)) == 1

    text_only = text_without_code_blocks(response)
    no_bold = "**" not in text_only
    no_extra = not re.search(r"Verification Points|Complexity|Notes|Examples|Test Cases", text_only, re.I)
    english = is_english_only(text_only)
    thinking_section = re.search(r"##\s*Thinking Process\s*(.*?)(?=##\s*Final Answer|\Z)", text_only, re.I | re.S)
    thinking_words = re.findall(r"[A-Za-z0-9']+", thinking_section.group(1) if thinking_section else "")
    explanation_ok = thinking_label_ok and english and no_bold and no_extra and 30 <= len(thinking_words) <= 250

    generated_tokens = (
        as_int(keyword_row.get("generated_tokens"))
        + as_int(thinking_row.get("generated_tokens"))
        + as_int(code_row.get("generated_tokens"))
    )
    truncated = (
        as_bool(keyword_row.get("truncated"))
        or as_bool(thinking_row.get("truncated"))
        or as_bool(code_row.get("truncated"))
    )
    generation_seconds = round(
        as_float(keyword_row.get("generation_seconds"))
        + as_float(thinking_row.get("generation_seconds"))
        + as_float(code_row.get("generation_seconds")),
        4,
    )
    length_ok = 0 < generated_tokens <= max_tokens
    time_ok = generation_seconds <= max_seconds

    api_success = bool(response) and as_bool(keyword_row.get("api_success")) and as_bool(thinking_row.get("api_success")) and as_bool(code_row.get("api_success"))
    format_ok = (
        response.startswith("## RAG Keywords")
        and keyword_ok
        and thinking_label_ok
        and code_ok
        and code_block_ok
        and no_bold
        and no_extra
    )
    relaxed_format_ok = (
        response.startswith("## RAG Keywords")
        and keyword_usable
        and thinking_usable
        and code_ok
        and code_block_ok
        and length_ok
    )
    usable_response_ok = (
        api_success
        and keyword_usable
        and thinking_usable
        and code_ok
        and code_block_ok
        and length_ok
        and not truncated
    )
    baseline_checks = [
        api_success,
        keyword_usable,
        thinking_usable,
        code_ok,
        code_block_ok,
        explanation_ok,
        length_ok,
        not truncated,
        time_ok,
    ]
    baseline_50_score = round(sum(baseline_checks) / len(baseline_checks) * 100, 2)
    baseline_50_ok = baseline_50_score >= 50.0
    stable_ok = api_success and format_ok and explanation_ok and length_ok and not truncated

    checks = {
        "api_failed": not api_success,
        "format_failed": not format_ok,
        "relaxed_format_failed": not relaxed_format_ok,
        "usable_response_failed": not usable_response_ok,
        "baseline_50_failed": not baseline_50_ok,
        "keyword_failed": not keyword_ok,
        "keyword_unusable": not keyword_usable,
        "thinking_failed": not thinking_ok,
        "thinking_label_failed": not thinking_label_ok,
        "thinking_unusable": not thinking_usable,
        "code_failed": not code_ok,
        "strict_code_failed": not strict_code_ok,
        "relaxed_code_failed": not relaxed_code_ok,
        "code_block_failed": not code_block_ok,
        "explanation_failed": not explanation_ok,
        "bold_found": not no_bold,
        "extra_section_found": not no_extra,
        "length_failed": not length_ok,
        "truncated": truncated,
        "generation_time_failed": not time_ok,
    }

    return FullRow(
        sample_index=sample_index,
        repeat_index=repeat_index,
        api_success=api_success,
        format_ok=format_ok,
        relaxed_format_ok=relaxed_format_ok,
        usable_response_ok=usable_response_ok,
        baseline_50_ok=baseline_50_ok,
        baseline_50_score=baseline_50_score,
        keyword_ok=keyword_ok,
        keyword_usable=keyword_usable,
        thinking_label_ok=thinking_label_ok,
        thinking_usable=thinking_usable,
        code_ok=code_ok,
        strict_code_ok=strict_code_ok,
        relaxed_code_ok=relaxed_code_ok,
        code_block_ok=code_block_ok,
        explanation_ok=explanation_ok,
        length_ok=length_ok,
        truncated=truncated,
        stable_ok=stable_ok,
        generated_tokens=generated_tokens,
        response_chars=len(response),
        generation_seconds=generation_seconds,
        generation_time_ok=time_ok,
        rag_keywords=" | ".join(keywords),
        rag_keyword_count=len(keywords),
        rag_allowed_vocab_count=sum(k in ALLOWED_VOCAB for k in keywords),
        errors=";".join(name for name, failed in checks.items() if failed),
        response=response,
    )


def summarize(rows: List[FullRow]) -> Dict[str, Any]:
    return {
        "total_generations": len(rows),
        "format_compliance": rate(sum(r.format_ok for r in rows), len(rows)),
        "relaxed_format_compliance": rate(sum(r.relaxed_format_ok for r in rows), len(rows)),
        "usable_response_rate": rate(sum(r.usable_response_ok for r in rows), len(rows)),
        "baseline_50_pass_rate": rate(sum(r.baseline_50_ok for r in rows), len(rows)),
        "avg_baseline_50_score": round(sum(r.baseline_50_score for r in rows) / len(rows), 2) if rows else 0,
        "keyword_accuracy": rate(sum(r.keyword_ok for r in rows), len(rows)),
        "keyword_usable_rate": rate(sum(r.keyword_usable for r in rows), len(rows)),
        "thinking_label_accuracy": rate(sum(r.thinking_label_ok for r in rows), len(rows)),
        "thinking_usable_rate": rate(sum(r.thinking_usable for r in rows), len(rows)),
        "code_quality": rate(sum(r.code_ok for r in rows), len(rows)),
        "strict_code_quality": rate(sum(r.strict_code_ok for r in rows), len(rows)),
        "relaxed_code_quality": rate(sum(r.relaxed_code_ok for r in rows), len(rows)),
        "code_block_accuracy": rate(sum(r.code_block_ok for r in rows), len(rows)),
        "explanation_quality": rate(sum(r.explanation_ok for r in rows), len(rows)),
        "response_stability": rate(sum(r.stable_ok for r in rows), len(rows)),
        "response_length_compliance": rate(sum(r.length_ok for r in rows), len(rows)),
        "generation_time_compliance": rate(sum(r.generation_time_ok for r in rows), len(rows)),
        "not_truncated_rate": rate(sum(not r.truncated for r in rows), len(rows)),
        "avg_generated_tokens": round(sum(r.generated_tokens for r in rows) / len(rows), 2) if rows else 0,
        "avg_response_chars": round(sum(r.response_chars for r in rows) / len(rows), 2) if rows else 0,
        "avg_generation_seconds": round(sum(r.generation_seconds for r in rows) / len(rows), 4) if rows else 0,
        "max_generation_seconds": round(max((r.generation_seconds for r in rows), default=0), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and evaluate BASE full answers from split outputs.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT / "full"))
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--max-generation-seconds", type=float, default=20.0)
    args = parser.parse_args()

    root = Path(args.output_root)
    keyword_rows = load_details(root / "keyword" / "details.csv")
    thinking_rows = load_details(root / "thinking" / "details.csv")
    code_rows = load_details(root / "code" / "details.csv")

    common_keys = sorted(set(keyword_rows) & set(thinking_rows) & set(code_rows))
    if args.limit > 0:
        common_keys = [key for key in common_keys if key[0] < args.limit]

    rows: List[FullRow] = []
    for sample_index, repeat_index in common_keys:
        rows.append(
            evaluate_joined_response(
                sample_index,
                repeat_index,
                keyword_rows[(sample_index, repeat_index)],
                thinking_rows[(sample_index, repeat_index)],
                code_rows[(sample_index, repeat_index)],
                args.max_generation_seconds,
                args.max_new_tokens,
            )
        )

    summary = summarize(rows)
    save_outputs(Path(args.output_dir), "summary.json", "details.csv", summary, rows)

    print("\nBASE FULL EVALUATION")
    print("- source: split keyword/thinking/code details.csv")
    print(f"- matched_generations: {len(rows)}")
    for key, value in summary.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    main()
