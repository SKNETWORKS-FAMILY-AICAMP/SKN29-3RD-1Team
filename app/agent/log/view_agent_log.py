# app/agent/log/view_agent_log.py

import json
import sys
from pathlib import Path
from pprint import pprint


LOG_DIR = Path(__file__).parent


def print_separator():
    print("\n" + "=" * 100)


def load_jsonl(file_path: Path):
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            try:
                yield json.loads(line)
            except Exception as e:
                print(f"[ERROR] Invalid JSON line: {e}")


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python view_agent_log.py retrieve")
        print("  python view_agent_log.py retrieve.jsonl")
        return

    file_name = sys.argv[1]

    if not file_name.endswith(".jsonl"):
        file_name += ".jsonl"

    file_path = LOG_DIR / file_name

    if not file_path.exists():
        print(f"[ERROR] File not found: {file_path}")
        return

    count = 0

    for log in load_jsonl(file_path):
        count += 1

        print_separator()

        print(f"[#{count}]")
        print(f"timestamp : {log.get('timestamp')}")
        print(f"agent     : {log.get('agent')}")

        print("\n[state]")
        pprint(
            log.get("state", {}),
            sort_dicts=False,
            width=120
        )

    print_separator()
    print(f"Total Logs: {count}")


if __name__ == "__main__":
    main()