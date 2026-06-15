"""
KoNLPy 기반 한국어 전처리 유틸리티

목적
- 평가표의 '정규표현식 + KoNLPy 형태소 분석 + 불용어 처리' 항목 대응
- 기존 Retrieval V1 성능을 건드리지 않고, 전처리/TF-IDF baseline에서 재사용

주의
- KoNLPy가 설치되어 있지 않으면 regex 기반 fallback 토크나이저로 동작한다.
- 현재 Retrieval V1(Dense Retrieval)에는 영향을 주지 않는다.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Optional, Sequence


DEFAULT_STOPWORDS = {
    "은", "는", "이", "가", "을", "를", "의", "에", "에서", "으로", "로",
    "와", "과", "도", "만", "에게", "한테", "하다", "되다", "있다", "없다",
    "그리고", "또는", "그러나", "하지만", "때", "수", "것", "좀", "더", "잘",
    "알려", "설명", "방법", "문제", "사용", "언제", "어떤", "무엇", "뭐야",
}

# 코딩테스트 도메인 핵심 토큰은 불용어 제거 대상에서 보호한다.
DOMAIN_KEEPWORDS = {
    "dfs", "bfs", "dp", "heap", "queue", "deque", "stack", "hash",
    "trie", "sort", "greedy", "graph", "binary", "search", "priority",
    "dijkstra", "bellman", "floyd", "union", "find", "segment", "tree",
    "bitmask", "backtracking", "bruteforce", "mst", "lcs", "stringbuilder",
    "collections", "itertools", "counter", "defaultdict", "heapq",
}


def clean_text(text: str) -> str:
    """정규표현식 기반 텍스트 정규화."""
    if not text:
        return ""

    text = str(text)

    # 코드블록 마커/HTML/URL/이미지 태그 정리
    text = re.sub(r"```[\s\S]*?```", " CODE_BLOCK ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " IMAGE ", text)

    # 한글, 영어, 숫자, 코딩테스트에서 자주 쓰는 기호 일부만 유지
    text = re.sub(r"[^0-9a-zA-Z가-힣_+/#\-\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _get_okt():
    try:
        from konlpy.tag import Okt  # type: ignore
        return Okt()
    except Exception:
        return None


def tokenize_korean(
    text: str,
    stopwords: Optional[Iterable[str]] = None,
    keep_pos: Optional[Sequence[str]] = None,
) -> List[str]:
    """KoNLPy 형태소 분석 + 불용어 제거.

    KoNLPy가 없으면 정규표현식 기반 토큰화로 fallback한다.
    """
    cleaned = clean_text(text).lower()
    stopword_set = set(DEFAULT_STOPWORDS)
    if stopwords:
        stopword_set.update(stopwords)

    okt = _get_okt()
    tokens: List[str] = []

    if okt is not None:
        # 명사/동사/형용사/알파벳/숫자 중심으로 유지
        allowed_pos = set(keep_pos or ["Noun", "Verb", "Adjective", "Alpha", "Number"])
        try:
            for word, pos in okt.pos(cleaned, stem=True):
                word = word.strip().lower()
                if not word:
                    continue
                if pos not in allowed_pos and word not in DOMAIN_KEEPWORDS:
                    continue
                if word in stopword_set and word not in DOMAIN_KEEPWORDS:
                    continue
                if len(word) <= 1 and word not in DOMAIN_KEEPWORDS:
                    continue
                tokens.append(word)
        except Exception:
            # JVM/KoNLPy 런타임 문제 발생 시 fallback
            tokens = re.findall(r"[0-9a-zA-Z_+/#\-]+|[가-힣]{2,}", cleaned)
    else:
        tokens = re.findall(r"[0-9a-zA-Z_+/#\-]+|[가-힣]{2,}", cleaned)

    return [t for t in tokens if t not in stopword_set or t in DOMAIN_KEEPWORDS]


def preprocess_for_tfidf(text: str) -> str:
    """TfidfVectorizer 입력용 공백 구분 토큰 문자열 생성."""
    return " ".join(tokenize_korean(text))


if __name__ == "__main__":
    sample = "정렬된 배열에서 원하는 값을 빠르게 찾는 방법 알려줘. lower bound도 설명해줘."
    print("[clean_text]")
    print(clean_text(sample))
    print("\n[tokens]")
    print(tokenize_korean(sample))
    print("\n[tfidf_text]")
    print(preprocess_for_tfidf(sample))
