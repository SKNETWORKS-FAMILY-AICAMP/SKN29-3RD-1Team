import json
import re
from pprint import pformat
import os
from datetime import datetime

def extract_json(text: str) -> dict:

    match = re.search(
        r"\{.*\}",
        text,
        re.DOTALL
    )

    if not match:
        raise ValueError(
            "json not found"
        )

    return json.loads(
        match.group()
    )



def pretty_print(data, title=None):
    if title:
        print(f"\n{'=' * 20} {title} {'=' * 20}")

    print(pformat(
        data,
        indent=2,
        width=120,
        sort_dicts=False,
        compact=False
    ))

    if title:
        print("=" * (42 + len(title)))


def dump_agent_state(
    agent_name: str,
    state: dict,
    file_name: str | None = None
):
    """
    Agent state를 JSONL 형태로 append 저장

    Args:
        agent_name: 현재 에이전트 이름
        state: 저장할 state
        file_name: 저장 파일명 (확장자 제외)
    """

    if not file_name:
        file_name = agent_name

    base_path = "app/agent/log"
    os.makedirs(base_path, exist_ok=True)

    file_path = os.path.join(base_path, f"{file_name}.jsonl")

    log_row = {
        "timestamp": datetime.now().isoformat(),
        "agent": agent_name,
        "state": state
    }

    with open(file_path, "a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                log_row,
                ensure_ascii=False,
                default=str
            )
            + "\n"
        )