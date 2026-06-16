from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from eval_common import DEFAULT_OUTPUT_ROOT


STAGES = ["keyword", "thinking", "code", "full"]


def load_summary(root: Path, stage: str) -> Dict[str, Any]:
    path = root / stage / "summary.json"
    if not path.exists():
        return {"missing": True, "path": str(path)}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Mege split BASE evaluation summaries.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    args = parser.parse_args()

    root = Path(args.output_root)
    merged = {
        "output_root": str(root),
        "stages": {stage: load_summary(root, stage) for stage in STAGES},
        "interpretation": {
            "keyword": "RAG keyword-only capability.",
            "thinking": "Thinking Process label and explanation capability.",
            "code": "Final Python code block capability.",
            "full": "End-to-end answer generation capability.",
        },
    }
    root.mkdir(parents=True, exist_ok=True)
    out_path = root / "final_summary.json"
    out_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nBASE SPLIT EVALUATION SUMMARY")
    for stage in STAGES:
        print(f"\n[{stage}]")
        summary = merged["stages"][stage]
        if summary.get("missing"):
            print(f"- missing: {summary['path']}")
            continue
        for key, value in summary.items():
            if isinstance(value, (int, float, str, bool)):
                print(f"- {key}: {value}")
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
