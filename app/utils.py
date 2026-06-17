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

def safe_file_name(input_str: str) -> str:
    # 1. 윈도우 금지 특수문자(\ / : * ? " < > |)를 언더바(_)로 치환
    forbidden_chars = r'[\\/:*?"<>|]'
    filename = re.sub(forbidden_chars, '_', input_str)
    
    # 2. 파일 이름 앞뒤의 공백과 마침표(.) 제거
    filename = filename.strip(' .')
    
    # 3. 윈도우 시스템 예약어 처리 (CON, PRN 등 사용 시 앞에 언더바 추가)
    reserved_names = {
        "CON", "PRN", "AUX", "NUL",
        "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
        "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9"
    }
    
    # 확장자를 제외한 본래 파일명만 추출하여 비교
    base_name = filename.split('.')[0].upper()
    if base_name in reserved_names:
        filename = f"_{filename}"
        
    # 4. 치환 후 빈 문자열이 되었을 경우 기본값 지정
    if not filename:
        filename = "untitled"
        
    return filename