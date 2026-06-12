import re
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


# ============================================================
# 1. 구조 자동 감지
# ============================================================
def detect_format(text):
    if "# 본문" in text and "# 메타데이터" in text:
        return "A"
    if re.search(r"^## \d+\.", text, re.MULTILINE) and "## 메타데이터" in text:
        return "B"
    return "C"


# ============================================================
# 2. 본문 추출
# ============================================================
def extract_body(text, fmt):
    """구조에 맞게 본문만 추출"""
    if fmt == "A":
        m = re.search(r"# 본문\s*\n(.*?)(?=\n# 메타데이터)", text, re.DOTALL)
        return m.group(1).strip() if m else text
    elif fmt == "B":
        m = re.search(r"(## 1\..+?)(?=\n## 메타데이터)", text, re.DOTALL)
        return m.group(1).strip() if m else text
    return text
 
# ============================================================
# 3. 전처리
# ============================================================
def clean_text(text):
    """마크다운 노이즈 제거, 코드블록 보존"""
    blocks = []
    def _save(m):
        blocks.append(m.group(0))
        return f"__CODE_{len(blocks)-1}__"
    text = re.sub(r"```[\s\S]*?```", _save, text)
    text = re.sub(r"<IMAGE>(.*?)</IMAGE>", r"[그림: \1]", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\*(.*?)\*", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    for i, b in enumerate(blocks):
        text = text.replace(f"__CODE_{i}__", b)
    return text.strip()

# ============================================================
# 4. 청킹 전략 3가지
# ============================================================
 
def chunk_fixed(text, size=1000, overlap=200):
    """전략 1: 고정 크기 분할"""
    chunks, start = [], 0
    while start < len(text):
        t = text[start:start+size].strip()
        if t:
            chunks.append({"text": t, "title": f"fixed_{len(chunks)}", "type": "concept"})
        start += size - overlap
    return chunks
 
 
def chunk_recursive(text, max_size=2000):
    """전략 2: 구분자 우선순위 재귀 분할"""
    seps = ["\n## ", "\n### ", "\n\n", "\n", ". ", " "]
 
    def _split(t, si):
        if len(t) <= max_size or si >= len(seps):
            return [t] if t.strip() else []
        parts = t.split(seps[si])
        res, cur = [], ""
        for p in parts:
            cand = cur + seps[si] + p if cur else p
            if len(cand) <= max_size:
                cur = cand
            else:
                if cur: res.extend(_split(cur, si + 1))
                cur = p
        if cur: res.extend(_split(cur, si + 1))
        return res
 
    raw = _split(text, 0)
    return [{"text": c.strip(), "title": f"recursive_{i}", "type": "concept"}
            for i, c in enumerate(raw) if c.strip()]
 
 
def chunk_semantic(text, doc_format="A"):
    chunks, cur = [], {"title": "", "lines": [], "type": "concept"}
 
    for line in text.split("\n"):
        if doc_format == "A" and re.match(r"^### 문제 \d+", line):
            if cur["lines"]: chunks.append(cur)
            cur = {"title": line.lstrip("#").strip(), "lines": [line], "type": "problem"}
            continue
 
        if re.match(r"^## ", line):
            skip = ["실전 문제 풀이", "메타데이터"]
            if any(s in line for s in skip):
                if cur["lines"]: chunks.append(cur)
                cur = {"title": "", "lines": [], "type": "concept"}
                continue
            if cur["lines"]: chunks.append(cur)
            title = line.lstrip("#").strip()
            title = re.sub(r"^\d+\.\s*", "", title)
            cur = {"title": title, "lines": [line], "type": "concept"}
            continue
 
        cur["lines"].append(line)
 
    if cur["lines"]: chunks.append(cur)
 
    return [{"text": "\n".join(c["lines"]).strip(), "title": c["title"], "type": c["type"]}
            for c in chunks if len("\n".join(c["lines"]).strip()) > 30]

# ============================================================
# 5. 전략 평가
# ============================================================
def evaluate_strategy(chunks):
    """코드보존율 + 문맥완결성 측정"""
    texts = [c["text"] for c in chunks]
    sizes = [len(t) for t in texts]
 
    # 코드 블록 보존율
    broken = sum(1 for t in texts if t.count("```") % 2 != 0)
    code_pct = round((1 - broken / max(len(chunks), 1)) * 100, 1)
 
    # 문맥 완결성
    markers = ["#### 핵심 개념", "#### 풀이 전략", "#### 소스코드"]
    ctx_ok, ctx_total = 0, 0
    for t in texts:
        found = [m for m in markers if m in t]
        if found:
            ctx_total += 1
            if len(found) == len(markers):
                ctx_ok += 1
    ctx_pct = round((ctx_ok / max(ctx_total, 1)) * 100, 1)
 
    # 검색 정확도
    tests = [
        {"query_kw": ["삽입"], "answer_kw": ["sift", "부모", "Push", "위로", "넣"]},
        {"query_kw": ["시간 복잡도", "O("], "answer_kw": ["O(1)", "O(n)", "O(log"]},
        {"query_kw": ["코드", "구현"], "answer_kw": ["def ", "public ", "void ", "return"]},
        {"query_kw": ["실수", "주의"], "answer_kw": ["실수", "주의", "헷갈", "틀리"]},
        {"query_kw": ["언제", "사용"], "answer_kw": ["신호", "경우", "패턴", "문제"]},
    ]
    hits = 0
    for tc in tests:
        best_idx, best_score = 0, 0
        for i, t in enumerate(texts):
            score = sum(1 for kw in tc["query_kw"] if kw.lower() in t.lower())
            if score > best_score:
                best_score, best_idx = score, i
        if any(kw.lower() in texts[best_idx].lower() for kw in tc["answer_kw"]):
            hits += 1
    ret_pct = round(hits / len(tests) * 100, 1)
 
    total = code_pct * 0.25 + ctx_pct * 0.35 + ret_pct * 0.40
    return {"code_integrity": code_pct, "context_completeness": ctx_pct,
            "retrieval": ret_pct, "total": round(total, 1)}


# ============================================================
# 6. 최적 전략 자동 선택 + Document 변환
# ============================================================
 
def split_documents(documents):
    """
    기존 호환: LangChain Document 리스트를 받아서 청킹 후 반환.
    일반 텍스트면 RecursiveCharacterTextSplitter 사용.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    return splitter.split_documents(documents)
 
 
def split_md_smart(text, metadata=None):
    if metadata is None:
        metadata = {}
 
    fmt = detect_format(text)
 
    # 일반 텍스트면 기존 방식으로 폴백
    if fmt == "C":
        doc = Document(page_content=text, metadata=metadata)
        return split_documents([doc])
 
    # 본문 추출 + 전처리
    body = extract_body(text, fmt)
    cleaned = clean_text(body)
 
    # 3가지 전략 비교
    strategies = {
        "fixed_size": chunk_fixed(cleaned),
        "recursive": chunk_recursive(cleaned),
        "semantic": chunk_semantic(cleaned, fmt),
    }
 
    best_name, best_score = "", 0
    for name, chunks in strategies.items():
        ev = evaluate_strategy(chunks)
        if ev["total"] > best_score:
            best_score = ev["total"]
            best_name = name
 
    # 최적 전략의 청크를 Document로 변환
    best_chunks = strategies[best_name]
    documents = []
    for chunk in best_chunks:
        chunk_meta = {
            **metadata,
            "chunk_type": chunk["type"],
            "section_title": chunk["title"],
            "strategy": best_name,
        }
        documents.append(Document(page_content=chunk["text"], metadata=chunk_meta))
 
    return documents