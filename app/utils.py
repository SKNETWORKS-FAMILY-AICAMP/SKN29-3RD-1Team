import json
import re


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