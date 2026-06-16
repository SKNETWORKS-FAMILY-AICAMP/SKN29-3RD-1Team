from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import evaluate_base_full as base
from eval_common import DEFAULT_LORA_OUTPUT_ROOT, save_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and evaluate LoRA full answers from split outputs.")
    parser.add_argument("--output-root", default=str(DEFAULT_LORA_OUTPUT_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_LORA_OUTPUT_ROOT / "full"))
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--max-generation-seconds", type=float, default=20.0)
    args = parser.parse_args()

    root = Path(args.output_root)
    keyword_rows = base.load_details(root / "keyword" / "details.csv")
    thinking_rows = base.load_details(root / "thinking" / "details.csv")
    code_rows = base.load_details(root / "code" / "details.csv")

    common_keys = sorted(set(keyword_rows) & set(thinking_rows) & set(code_rows))
    if args.limit > 0:
        common_keys = [key for key in common_keys if key[0] < args.limit]

    rows: List[base.FullRow] = []
    for sample_index, repeat_index in common_keys:
        rows.append(
            base.evaluate_joined_response(
                sample_index,
                repeat_index,
                keyword_rows[(sample_index, repeat_index)],
                thinking_rows[(sample_index, repeat_index)],
                code_rows[(sample_index, repeat_index)],
                args.max_generation_seconds,
                args.max_new_tokens,
            )
        )

    summary = base.summarize(rows)
    save_outputs(Path(args.output_dir), "summary.json", "details.csv", summary, rows)

    print("\nLORA FULL EVALUATION")
    print("- source: split keyword/thinking/code details.csv")
    print(f"- matched_generations: {len(rows)}")
    for key, value in summary.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    main()
