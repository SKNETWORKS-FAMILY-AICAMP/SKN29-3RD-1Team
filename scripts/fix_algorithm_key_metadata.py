from __future__ import annotations

import json
import sqlite3
from pathlib import Path

DB_PATH = Path("chroma_db/chroma.sqlite3")

SOURCE_FILE_MAPPING = {
    "number_type_ai1.md": ("Python 숫자형", "python_number_type"),
    "string_ai1.md": ("Python 문자열", "python_string"),
    "list_ai1.md": ("Python 리스트", "python_list"),
    "tuple_ai1.md": ("Python 튜플", "python_tuple"),
    "dictionary_ai1.md": ("Python 딕셔너리", "python_dictionary"),
    "set_ai1.md": ("Python 집합", "python_set"),
    "function_ai1.md": ("Python 함수", "python_function"),
    "class_ai1.md": ("Python 클래스", "python_class"),
    "exception_ai1.md": ("Python 예외처리", "python_exception"),
    "regular_expression_ai1.md": ("Python 정규표현식", "python_regex"),
    "itertools_ai1.md": ("Python itertools", "python_itertools"),
}


def upsert_string_metadata(cur: sqlite3.Cursor, embedding_id: int, key: str, value: str) -> None:
    exists = cur.execute(
        "SELECT 1 FROM embedding_metadata WHERE id=? AND key=?",
        (embedding_id, key),
    ).fetchone()
    if exists:
        cur.execute(
            """
            UPDATE embedding_metadata
               SET string_value=?, int_value=NULL, float_value=NULL, bool_value=NULL
             WHERE id=? AND key=?
            """,
            (value, embedding_id, key),
        )
    else:
        cur.execute(
            "INSERT INTO embedding_metadata(id, key, string_value) VALUES (?, ?, ?)",
            (embedding_id, key, value),
        )


def main() -> None:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Chroma DB not found: {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    blank_ids = [
        row[0]
        for row in cur.execute(
            """
            SELECT id
              FROM embedding_metadata
             WHERE key='algorithm_key'
               AND COALESCE(string_value, '')=''
            """
        ).fetchall()
    ]

    fixed = 0
    skipped = []
    for embedding_id in blank_ids:
        row = cur.execute(
            "SELECT string_value FROM embedding_metadata WHERE id=? AND key='source_file'",
            (embedding_id,),
        ).fetchone()
        source_file = row[0] if row else ""
        if source_file not in SOURCE_FILE_MAPPING:
            skipped.append((embedding_id, source_file))
            continue

        algorithm, algorithm_key = SOURCE_FILE_MAPPING[source_file]
        upsert_string_metadata(cur, embedding_id, "algorithm", algorithm)
        upsert_string_metadata(cur, embedding_id, "algorithm_key", algorithm_key)
        upsert_string_metadata(cur, embedding_id, "display_name", algorithm)
        fixed += 1

    # Chroma queue metadata에도 동일하게 반영해서 DB 재처리/동기화 시 빈값이 재발하지 않게 한다.
    queue_fixed = 0
    rows = cur.execute(
        "SELECT seq_id, metadata FROM embeddings_queue WHERE metadata IS NOT NULL"
    ).fetchall()
    for seq_id, metadata_json in rows:
        try:
            metadata = json.loads(metadata_json)
        except Exception:
            continue
        source_file = metadata.get("source_file") or metadata.get("filename")
        if source_file in SOURCE_FILE_MAPPING and not metadata.get("algorithm_key"):
            algorithm, algorithm_key = SOURCE_FILE_MAPPING[source_file]
            metadata["algorithm"] = algorithm
            metadata["algorithm_key"] = algorithm_key
            metadata["display_name"] = algorithm
            cur.execute(
                "UPDATE embeddings_queue SET metadata=? WHERE seq_id=?",
                (json.dumps(metadata, ensure_ascii=False), seq_id),
            )
            queue_fixed += 1

    conn.commit()

    remaining_blank = cur.execute(
        """
        SELECT COUNT(*)
          FROM embedding_metadata
         WHERE key='algorithm_key'
           AND COALESCE(string_value, '')=''
        """
    ).fetchone()[0]
    conn.close()

    print(f"fixed_embedding_metadata={fixed}")
    print(f"fixed_embeddings_queue={queue_fixed}")
    print(f"remaining_blank_algorithm_key={remaining_blank}")
    if skipped:
        print(f"skipped={skipped}")


if __name__ == "__main__":
    main()
