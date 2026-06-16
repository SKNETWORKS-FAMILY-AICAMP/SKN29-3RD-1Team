import json
import re
from pprint import pformat


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