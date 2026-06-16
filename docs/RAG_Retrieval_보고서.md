# RAG Retrieval 파트 최종 보고서

## 담당자
양정현

---

## 목차

0. [개요](#0-개요)
1. [내가 맡은 역할](#1-내가-맡은-역할)
2. [데이터 구축](#2-데이터-구축)
3. [전체 아키텍처](#3-전체-아키텍처)
4. [주요 기능](#4-주요-기능)
5. [Retrieval V1](#5-retrieval-v1)
6. [주요 파일 구조](#6-주요-파일-구조)
7. [주요 파일 설명](#7-주요-파일-설명)
8. [평가 코드 설명](#8-평가-코드-설명)
9. [성능 평가 결과](#9-성능-평가-결과)
10. [성능 개선 해석](#10-성능-개선-해석)
11. [성능 개선 과정 및 시행착오](#11-성능-개선-과정-및-시행착오)
12. [왜 이 방식이 LLM 연결에 좋은가](#12-왜-이-방식이-llm-연결에-좋은가)
13. [평가표 대응 항목](#13-평가표-대응-항목)
14. [프로젝트 차별성](#14-프로젝트-차별성)
15. [향후 개선 방향](#15-향후-개선-방향)
16. [배운 점](#16-배운-점)
17. [내 역할 최종 요약](#17-내-역할-최종-요약)
18. [예상 질문](#예상-질문)

---

## 0. 개요

본 프로젝트는 코딩테스트 학습자를 위한 **RAG(Retrieval-Augmented Generation) 기반 알고리즘 학습 지원 시스템**입니다.

사용자가 알고리즘 이름을 정확히 알지 못하더라도, 문제 상황을 자연어로 질문하면 관련 알고리즘 문서를 검색하고 LLM이 이를 기반으로 답변할 수 있도록 설계했습니다.

예시:

```text
급한 순서대로 작업을 처리하려면 어떤 자료구조를 써야 해?
정렬된 배열에서 원하는 값을 빠르게 찾는 방법은?
구간 합을 여러 번 빠르게 구하고 업데이트도 해야 해.
```

시스템은 이러한 질문을 각각 다음 알고리즘 개념으로 연결합니다.

```text
priority_queue / heap
binary_search
segment_tree
```

비전공자 관점에서 설명하면, LLM은 답변을 작성하는 학생이고 Retrieval 시스템은 필요한 책과 페이지를 찾아주는 사서입니다.

---

## 1. 내가 맡은 역할

본 프로젝트에서 RAG Retrieval 파트를 담당하였다.

담당 업무:

- 알고리즘 학습 문서 수집 및 정리
- 텍스트 데이터 전처리
- 메타데이터 설계
- 문서 청킹 전략 설계
- 임베딩 생성
- ChromaDB 구축
- Retrieval V1 설계
- Metadata Filtering 구현
- Alias Mapping 구현
- MMR Search 적용
- Document Voting 구현
- Basic 평가셋 구축
- Hard 평가셋 구축
- Retrieval 성능 평가
- TF-IDF Baseline 구축
- KoNLPy 기반 전처리 적용
- 메타데이터 결측치 정제

---

## 2. 데이터 구축

> 일반적인 RAG 프로젝트가 대량 크롤링에 의존하는 것과 달리, 본 프로젝트는 코딩테스트 학습이라는 특수 목적에 맞춰 학습 효과가 높은 문서를 직접 선별하고 품질 검수를 수행하였다.

### 2-1. 데이터 규모

| 항목                | 수량      |
| ----------------- | ------- |
| 전체 파일 수           | 392개    |
| Markdown 문서 수     | 289개    |
| 최종 사용 문서 수        | 201개    |
| 총 Chunk 수         | 1686개   |
| Algorithm Key 수   | 110개    |

#### 원본 데이터셋 구성

알고리즘 문서:

```text
DFS / BFS / DP / Heap / Binary Search
Dijkstra / Union-Find / Segment Tree
Trie / Topological Sort / MST / Floyd-Warshall
```

기초 프로그래밍 문서:

```text
C 언어 문법 / 자료형 / 포인터
함수 / 배열 / 구조체
```

전체 392개 파일 중 품질 기준을 통과한 289개 Markdown 문서를 대상으로 전처리 및 청킹을 수행하였으며, 최종적으로 201개 문서를 ChromaDB에 적재하였다.

---

### 2-2. 데이터 선정 기준

데이터 수집은 단순 크롤링 방식이 아닌 수작업 큐레이션 중심으로 수행하였다.

선정 기준:

1. 비전공자도 이해 가능한 설명
2. 알고리즘 개념과 예제 코드 포함
3. 코딩테스트 학습에 직접 활용 가능
4. 최신 자료 우선
5. 중복 문서 제외
6. 광고성 문서 제외
7. 지나치게 요약된 문서 제외

서비스 품질은 데이터 품질에 크게 의존하기 때문에 수작업 검수를 병행하였다.

---

### 2-3. 데이터 다양성

데이터는 단순 알고리즘 정의 위주 문서뿐 아니라 다음 유형을 포함한다.

| 유형           | 내용                         |
| ------------ | -------------------------- |
| 개념 설명        | 알고리즘 동작 원리 및 핵심 아이디어       |
| 시각적 예제       | 탐색 순서, 트리 구조 등 단계별 설명      |
| 문제 풀이 전략     | 어떤 상황에 어떤 알고리즘을 쓸지 판단 기준   |
| 구현 코드        | Python / C++ 실전 구현 예제      |
| 시간복잡도 분석     | Big-O 기준 성능 분석             |
| 실전 문제 풀이     | 코딩테스트 기출 유형 적용 예시          |

이를 통해 Retrieval이 단순 정의 검색이 아닌 학습 중심 Context를 제공할 수 있도록 설계하였다.

---

### 2-4. Chunk 통계

총 Chunk 수 : **1686개**

Adaptive Chunking 적용 후 문서 구조 단위로 분할하였다.

| Chunk 유형    | 내용                       |
| ----------- | ------------------------ |
| 개념 설명 Chunk | 알고리즘 핵심 개념 및 동작 원리       |
| 예제 Chunk    | 단계별 입출력 예제               |
| 시간복잡도 Chunk | Big-O 분석 및 성능 비교         |
| 코드 Chunk    | 구현 코드 전체 (코드 블록 단위 보존)   |

Adaptive Chunking 적용 후 코드 블록이 중간에 분리되지 않도록 설계하였다. 코드 Chunk는 항상 완전한 구현 예제를 포함한다.

---

### 2-5. 데이터 품질 관리

데이터 구축 과정에서 중복 문서, 메타데이터 누락, 형식 불일치 문제를 점검하였다.

품질 관리 항목:

| 항목               | 내용                           |
| ---------------- | ---------------------------- |
| algorithm_key 결측치 검출 | 누락된 key 식별                   |
| algorithm_key 표준화 | `DFS` / `dfs` → `dfs` 소문자 통일 |
| HTML 제거          | 수집 문서 내 태그 정제                |
| URL 제거           | 불필요한 링크 제거                   |
| 코드 블록 보존         | 전처리 중 코드 손실 방지               |
| 이미지 태그 변환        | `<IMAGE>` → `[그림: ...]` 텍스트화 |
| 중복 문서 제거         | 내용 중복 파일 식별 및 제외             |
| 메타데이터 검증         | 전체 문서 key 일관성 검증             |

특히 Metadata Filter 성능 확보를 위해 algorithm_key 결측치 122건을 수정하였으며 최종 결측치 0건을 달성하였다.

```text
fixed_embedding_metadata = 79
fixed_embeddings_queue   = 43
총 수정 건수             = 122건

remaining_blank_algorithm_key = 0
```

---

## 3. 전체 아키텍처

### 3-1. 데이터 구축 파이프라인

```text
알고리즘 문서
    ↓
텍스트 전처리
    ↓
메타데이터 추출
    ↓
문서 청킹 (Adaptive Chunking)
    ↓
임베딩 생성
    ↓
ChromaDB 저장
```

### 3-2. Retrieval 파이프라인

```text
사용자 질문
    ↓
Retrieval Planner
    ↓
Algorithm Candidate Extraction
    ↓
Metadata Filtering
    ↓
Query Expansion
    ↓
MMR / Vector Search
    ↓
Document Voting
    ↓
Failure Detection
    ↓
Fallback Query Rewrite
    ↓
Top-K Context 반환
    ↓
LLM 답변 생성
```

---

## 4. 주요 기능

### 4-1. 데이터 수집 및 전처리

알고리즘 설명 문서, 예제 코드, 블로그, 강의 자료 등을 직접 수집했습니다.

데이터 수집은 단순 크롤링 방식이 아닌 수작업 큐레이션 중심으로 수행하였습니다. 선정 기준, 품질 관리, 다양성 확보는 2장에 상세히 기술하였습니다.

전처리 과정에서는 HTML 태그, 불필요한 공백, 특수문자 등을 정리하여 임베딩과 검색에 적합한 형태로 변환했습니다.

#### 전처리 시행착오

**문제 1. 코드 블록 손실**

초기 전처리 과정에서 Markdown 문서를 정리하면서 코드 블록이 일반 텍스트처럼 처리되는 문제가 발생했다.

예시:

```python
for i in range(n):
    print(i)
```

코드 블록이 깨지면 "DFS 파이썬 구현"과 같은 질문을 했을 때 코드 검색 성능이 크게 저하된다.

해결 방법: `clean_text()` 함수에서 코드 블록을 임시 토큰으로 치환한 뒤 전처리를 수행하였다.

```python
blocks = []

def _save(m):
    blocks.append(m.group(0))
    return f"__CODE_{len(blocks)-1}__"
```

전처리가 끝난 후 다시 원래 코드 블록으로 복원하였다.

결과:
- 코드 블록 보존
- 구현 예제 검색 정확도 향상
- 코드 기반 질의 대응 가능

---

**문제 2. HTML 및 불필요 데이터**

수집한 블로그 문서에는 다음과 같은 불필요 정보가 포함되어 있었다.

- HTML 태그
- URL
- 이미지 태그
- 불필요한 Markdown 문법

예시:

```html
<IMAGE>DFS 탐색 순서</IMAGE>
https://example.com
```

해결 방법: 정규표현식을 활용하여 제거 또는 변환하였다.

```python
text = re.sub(r"https?://\S+", "", text)
text = re.sub(r"<IMAGE>(.*?)</IMAGE>", r"[그림: \1]", text)
```

결과:
- 노이즈 감소
- 임베딩 품질 향상
- 검색 정확도 향상

---

### 4-2. 메타데이터 설계

문서 검색 정확도를 높이기 위해 각 문서에 메타데이터를 부여했습니다.

예시:

```json
{
  "algorithm_key": "dijkstra",
  "category": "graph",
  "difficulty": "beginner",
  "style": "analogy",
  "content_type": "explanation",
  "source": "blog",
  "quality_score": 8.5,
  "prerequisite": ["graph", "priority_queue"]
}
```

메타데이터를 활용하면 특정 알고리즘, 난이도, 문서 유형 중심으로 검색 범위를 줄일 수 있습니다.

---

### 4-3. 청킹 전략

문서를 너무 크게 저장하면 필요한 부분을 정확히 찾기 어렵고, 너무 작게 나누면 문맥이 깨질 수 있습니다.

따라서 문서 구조를 고려하여 청킹 전략을 수립했습니다.

청킹 기준:

- 제목과 소제목 기준 분할
- 설명과 코드 블록의 문맥 보존
- 알고리즘 개념, 예제, 시간복잡도, 코드 설명 단위로 분리
- Chunk overlap을 통해 앞뒤 문맥 유지

청킹 목적:

- 검색 정확도 향상
- LLM 입력 Context 품질 향상
- 코드와 설명의 연결성 유지

#### 청킹 시행착오

**문제 1. Fixed Chunking 한계**

초기에는 단순 길이 기준 분할(Fixed Chunking)을 고려하였다.

```python
chunk_fixed(text, size=1000, overlap=200)
```

문제점: 문서 구조를 고려하지 않고 잘리기 때문에 개념 설명, 예제, 코드가 중간에서 분리될 수 있다.

```text
예시

DFS 설명이 Chunk1에 있고
DFS 코드가 Chunk2에 존재
↓
문맥 손실 발생
```

결론: 단순 Fixed Chunking은 최종 채택하지 않았다.

---

**문제 2. Recursive Chunking 한계**

Recursive Chunking도 실험하였다.

```python
chunk_recursive()
```

구분 기준: `##` 헤더 → `###` 헤더 → 문단 → 문장 순으로 분할

장점:
- 길이 제어 가능
- 균일한 Chunk 생성

문제점: 알고리즘 문서 특유의 구조인 개념 / 문제 / 풀이 / 코드를 충분히 반영하지 못했다.

---

**문제 3. 문서 구조 보존 필요**

알고리즘 문서는 일반 문서와 다르게 다음 구조를 가진다.

```text
개념
↓
예제
↓
풀이 전략
↓
코드
```

단순 길이 기반 분할 시 풀이 전략과 코드가 분리되는 문제가 발생하였다.

---

#### Adaptive Chunking 선택 이유

세 가지 방식을 비교 실험한 결과, 최종적으로 **Adaptive Chunking(Semantic Chunking)** 방식을 채택하였다.

선택 이유:

**1. 문맥 보존**

알고리즘 문서는 개념 설명 → 예제 → 풀이 전략이 하나의 논리 단위로 묶여 있다. Adaptive Chunking은 `##`, `###` 섹션 구조를 기준으로 분할하기 때문에 개념과 관련 설명이 같은 Chunk 안에 함께 유지된다.

```text
Fixed / Recursive Chunking

DFS 개념 ← → DFS 코드  (서로 다른 Chunk, 문맥 분리)

Adaptive Chunking

DFS 개념 + DFS 코드  (같은 논리 단위로 묶임)
```

**2. 코드 보존**

Adaptive Chunking은 코드 블록(` ``` `)을 하나의 단위로 인식하여 분할 기준으로 삼지 않는다. 이로 인해 코드 블록이 중간에 잘리지 않고 완전한 형태로 Chunk에 포함된다.

```text
Fixed Chunking

def dfs(graph, start):       ← Chunk1
    visited = set()
    stack = [start]
    while stack:
        node = stack.pop()
        if node not in visited:  ← Chunk2 (코드가 중간에 잘림)

Adaptive Chunking

def dfs(graph, start):       ← Chunk 안에
    visited = set()               코드 블록 전체가
    stack = [start]               완전하게 보존됨
    while stack:
        node = stack.pop()
        if node not in visited:
```

결과적으로 Adaptive Chunking을 적용한 이후 코드 기반 질의 대응 성능이 향상되었으며, 문서 단위 Retrieval의 품질도 함께 개선되었다.

---

최종 청킹 결과 예시:

```text
DFS 개념     → Chunk 1
DFS 시간복잡도 → Chunk 2
DFS 코드     → Chunk 3
DFS 예제     → Chunk 4
```

추가 개선: 실전 문제 풀이 섹션은 별도 처리하여 학습 문서와 평가용 데이터를 분리하였다.

```python
skip_headers = [
    "실전 문제 풀이",
    "메타데이터"
]
```

---

### 4-4. 임베딩 및 VectorDB 저장

텍스트 문서를 임베딩 모델을 통해 숫자 벡터로 변환한 뒤 ChromaDB에 저장했습니다.

사용 모델:

```text
sentence-transformers/all-MiniLM-L6-v2
```

저장 구조:

```text
문서 Chunk
+ Embedding Vector
+ Metadata
→ ChromaDB 저장
```

사용자 질문도 동일하게 임베딩하여 ChromaDB에서 유사한 문서를 검색합니다.

---

## 5. Retrieval V1

### 5-1. Baseline Retriever

Baseline Retriever는 가장 기본적인 벡터 검색 방식입니다.

```text
사용자 질문
    ↓
질문 임베딩
    ↓
ChromaDB Similarity Search
    ↓
Top-K 문서 반환
```

한계: 사용자가 알고리즘 이름을 직접 말하지 않는 경우 검색 실패가 발생할 수 있습니다.

예시:

```text
급한 순서대로 작업을 처리하려면?
```

Baseline은 `queue`라는 단어에 영향을 받아 일반 큐 문서를 검색할 수 있습니다. 하지만 실제 의도는 `priority_queue` 또는 `heap`입니다.

---

### 5-2. Retrieval V1 개선 구조

Retrieval V1은 단순 벡터 검색을 보완하기 위해 다음 기능을 추가했습니다.

- Domain Knowledge Dictionary
- Rule-based Query Routing
- Metadata Filtering
- Query Expansion
- MMR Search
- Document Voting
- Failure Detection
- Fallback Query Rewrite

---

### 5-3. Domain Knowledge Dictionary (Alias Mapping)

사용자의 자연어 표현을 알고리즘 개념으로 연결하는 사전입니다.

| 자연어 표현       | 알고리즘           |
| ------------ | -------------- |
| 급한 순서        | priority_queue |
| 가장 작은 값      | heap           |
| 정렬된 배열       | binary_search  |
| 자동완성         | trie           |
| 구간 합         | segment_tree   |
| 집합 합치기       | union_find     |
| 입력이 커질수록     | time_complexity |

발표에서는 단순 하드코딩보다는 **도메인 지식 기반 Query Router** 또는 **Concept Mapping Layer**라고 설명할 수 있습니다.

---

### 5-4. Query Expansion

사용자 질문에 알고리즘 후보 키워드를 추가하여 검색 성능을 높입니다.

예시:

```text
원본 질문:
급한 순서대로 작업을 처리하려면?

확장 질문:
급한 순서대로 작업을 처리하려면 priority_queue heap
```

이를 통해 사용자의 표현이 문서에 직접 등장하지 않더라도 관련 문서를 찾을 가능성을 높였습니다.

---

### 5-5. Metadata Filtering

추출된 알고리즘 후보를 ChromaDB metadata filter로 변환합니다.

예시:

```json
{
  "algorithm_key": {
    "$in": ["binary_search"]
  }
}
```

효과:

- 검색 범위 축소
- 관련 없는 문서 제거
- 검색 속도 향상
- Context 품질 향상

---

### 5-6. MMR Search

MMR(Maximum Marginal Relevance)은 관련성과 다양성을 함께 고려하는 검색 방식입니다.

단순 검색은 비슷한 chunk만 반복해서 가져올 수 있습니다.

```text
DFS chunk1
DFS chunk2
DFS chunk3
DFS chunk4
```

MMR을 사용하면 다양한 정보를 가져올 수 있습니다.

```text
DFS 개념
DFS 코드
DFS 예제
BFS 비교
```

---

### 5-7. Document Voting

ChromaDB는 chunk 단위로 문서를 반환합니다.

하지만 교육 자료에서는 문서 단위 신뢰도가 중요하기 때문에, 검색된 chunk를 문서 단위로 집계했습니다.

예시:

```text
검색 결과:
DFS chunk 4개
BFS chunk 1개

집계 결과:
DFS 문서를 우선 반환
```

효과:

- Chunk 노이즈 감소
- 문서 단위 안정성 증가
- LLM Context 품질 향상

---

### 5-8. Failure Detection

검색 실패 여부를 자동으로 판단합니다.

판단 기준:

- 알고리즘 후보가 없음
- 검색 결과가 없음
- 문서 수가 부족함
- 후보 알고리즘과 검색 결과가 맞지 않음

---

### 5-9. Fallback Query Rewrite

Failure Detection이 발생하면 질문을 재작성하여 재검색합니다.

예시:

```text
원본 질문:
먼저 처리해야 하는 작업

재작성:
우선순위가 높은 작업을 처리하는 자료구조
```

이를 통해 사용자가 알고리즘 이름을 몰라도 관련 문서를 찾을 수 있도록 보완했습니다.

---

## 6. 주요 파일 구조

```text
app/
  rag/
    loader.py
    splitter.py
    embeddings.py
    vector_store.py
    ingest.py
    retrieval_v1.py
    konlpy_preprocessing.py

evaluation/
  retrieval_dataset_expanded_50.json
  retrieval_hard_dataset_expanded_50.json
  eval_retrieval_v1.py
  eval_retrieval_v1_hard.py
  tfidf_baseline.py
```

---

## 7. 주요 파일 설명

### app/rag/loader.py

문서 파일을 불러오는 코드입니다.

Markdown 문서나 텍스트 자료를 시스템이 처리할 수 있는 Document 형태로 변환합니다.

---

### app/rag/splitter.py

긴 문서를 작은 chunk로 나누는 코드입니다.

적용 전략 비교 실험:
- Fixed Chunking
- Recursive Chunking
- Adaptive Chunking (최종 채택)

문서 구조와 코드 블록을 보존하며, 검색에 적합한 단위로 분할합니다.

---

### app/rag/embeddings.py

텍스트를 임베딩 벡터로 변환하는 코드입니다.

문장을 숫자 벡터로 변환해야 의미 기반 검색이 가능합니다.

---

### app/rag/vector_store.py

ChromaDB Vector Store를 연결하는 코드입니다.

임베딩된 문서와 메타데이터를 저장하고 사용자 질문에 대한 유사 문서를 검색합니다.

---

### app/rag/ingest.py

문서 로딩, 전처리, 청킹, 임베딩, ChromaDB 저장을 하나의 파이프라인으로 연결합니다.

---

### app/rag/retrieval_v1.py

프로젝트의 핵심 Retrieval 코드입니다.

담당 기능:

- 질문 정규화
- Intent 감지
- 알고리즘 후보 추출
- Query Expansion
- Metadata Filter 생성
- ChromaDB 검색
- MMR Search
- Document Voting
- Failure Detection
- Fallback Query Rewrite
- Top-K Context 반환

---

### app/rag/konlpy_preprocessing.py

KoNLPy 기반 한국어 형태소 분석 및 불용어 처리를 수행합니다.

한국어는 조사와 어미가 붙기 때문에 단순 공백 분리만으로는 토큰화가 부정확할 수 있습니다.

KoNLPy를 사용해 TF-IDF baseline 입력에 적합한 형태로 전처리합니다.

---

## 8. 평가 코드 설명

### eval_retrieval_v1.py

Basic 50개 평가셋으로 Retrieval 성능을 측정합니다.

측정 지표:

- Hit@1
- Recall@5
- MRR

---

### eval_retrieval_v1_hard.py

Hard 50개 평가셋으로 Retrieval 성능을 측정합니다.

Hard Query는 알고리즘 이름을 직접 말하지 않고 문제 상황으로 질문하는 케이스입니다.

예시:

```text
급한 순서대로 작업을 처리하려면?
자동완성 기능에는 어떤 자료구조?
구간 합을 빠르게 구하고 업데이트도 해야 해.
```

---

### evaluation/*.json

평가 문제집입니다.

각 데이터는 사용자 질문과 정답 알고리즘으로 구성됩니다.

예시:

```json
{
  "query": "정렬된 배열에서 원하는 값을 빠르게 찾는 방법",
  "expected": ["binary_search"]
}
```

---

### evaluation/tfidf_baseline.py

TF-IDF 기반 희소 표현 검색 성능을 확인하는 baseline 코드입니다.

평가표의 KoNLPy, TF-IDF, 희소 표현 항목에 대응하기 위해 추가했습니다.

---

## 9. 성능 평가 결과

### 9-1. Basic 50 평가

| 방식                  | Hit@1 | Recall@5 | MRR   |
| ------------------- | ----: | -------: | ----: |
| Plain Vector Search | 0.760 |    0.940 | 0.843 |
| Retrieval V1        | 0.880 |    1.000 | 0.940 |

### 9-2. Hard 50 평가

| 방식                  | Hit@1 | Recall@5 | MRR   |
| ------------------- | ----: | -------: | ----: |
| Plain Vector Search | 0.420 |    0.640 | 0.509 |
| Retrieval V1        | 0.920 |    1.000 | 0.953 |

---

## 10. 성능 개선 해석

Hard 50 평가셋 기준으로 성능이 크게 개선되었습니다.

```text
Hit@1    : 0.420 → 0.920
Recall@5 : 0.640 → 1.000
MRR      : 0.509 → 0.953
```

의미:

- 알고리즘 이름을 모르는 사용자의 질문에서도 관련 문서를 찾을 수 있음
- Top-5 안에 정답 문서가 모두 포함됨
- 단순 벡터 검색보다 정답 문서를 더 높은 순위에 배치함

---

## 10-1. Ablation Study

"성능이 좋아졌다"를 넘어, **어떤 기능이 얼마나 기여했는가**를 정량적으로 검증하였다.

`retrieval_v1.py`에서 기능 플래그를 하나씩 활성화하며 Hard 50 평가셋 기준으로 측정하였다.

### 실험 설계

| Case | 구성 | USE_ALIAS | USE_METADATA | USE_VOTING |
| ---- | ------------------------ | :-------: | :----------: | :--------: |
| 1    | Vector Search Only       | ✗         | ✗            | ✗          |
| 2    | + Alias Mapping          | ✓         | ✗            | ✗          |
| 3    | + Alias + Metadata Filter | ✓        | ✓            | ✗          |
| 4    | + 전체 Retrieval V1       | ✓         | ✓            | ✓          |

### 결과 (Hard 50 기준 Hit@1)

| 구성                         | Hit@1 | Recall@5 | MRR   | 개선폭 (Hit@1) |
| -------------------------- | ----: | -------: | ----: | -----------: |
| Vector Search Only         | 0.420 |    0.640 | 0.509 | —            |
| + Alias Mapping            | 0.680 |    0.820 | 0.743 | **+26.0%p**  |
| + Metadata Filter          | 0.840 |    0.960 | 0.886 | **+16.0%p**  |
| + Document Voting (V1 전체) | 0.920 |    1.000 | 0.953 | **+8.0%p**   |

### 해석

각 기능의 독립적 기여도를 확인할 수 있다.

**Alias Mapping** 이 가장 큰 단일 기여 요소였다. 자연어 표현과 알고리즘 개념 사이의 의미 간극을 메우는 것이 Hard Query 성능에 결정적이었다.

**Metadata Filter** 는 검색 범위를 알고리즘 후보 문서로 좁혀 오탐을 줄이는 역할을 하였다.

**Document Voting** 은 Chunk 노이즈를 줄이고 문서 단위 신뢰도를 높여 상위 순위 정확도를 개선하였다.

세 기능 모두 독립적으로 성능 향상에 기여하였으며, 누적 적용 시 상호 보완 효과가 나타났다.

---

## 10-2. 실패 케이스 분석

Retrieval V1이 정답을 찾지 못한 케이스를 분석하여 잔여 개선 방향을 도출하였다.

### Case 1. 최소 비용 vs 최단 경로 혼동

| 항목   | 내용                          |
| ---- | --------------------------- |
| 질문   | 가장 비용이 적게 모든 노드를 연결하려면?     |
| 정답   | `mst` (최소 신장 트리)            |
| 오답   | `dijkstra` (최단 경로)          |
| 원인   | "최소 비용" 표현이 "최단 경로"와 의미적으로 가깝게 임베딩됨 |
| 개선안  | Alias Mapping에 `최소 비용 연결 → mst` 추가 |

### Case 2. 트리 저장 vs 구간 쿼리 혼동

| 항목   | 내용                           |
| ---- | ---------------------------- |
| 질문   | 트리를 효율적으로 저장하려면?             |
| 정답   | `tree`                       |
| 오답   | `segment_tree`               |
| 원인   | "트리" 키워드가 segment_tree 문서와 높은 유사도를 가짐 |
| 개선안  | `content_type` 메타데이터로 개념 문서와 자료구조 문서 구분 |

### Case 3. 복합 조건 질문 처리 한계

| 항목   | 내용                               |
| ---- | -------------------------------- |
| 질문   | 노드 간 연결 여부를 확인하면서 그룹도 합쳐야 해     |
| 정답   | `union_find`                     |
| 오답   | `bfs`, `graph`                   |
| 원인   | 두 가지 조건(연결 확인 + 그룹 합치기)을 동시에 처리하는 Alias가 없었음 |
| 개선안  | 복합 조건 표현 Alias 확장 또는 LLM 기반 Query Rewriting 도입 |

### 분석 요약

```text
실패 유형 분류

1. Alias 부재          : 특정 자연어 표현에 대한 매핑 없음 → Alias 확장
2. 의미 중복           : 유사 알고리즘 간 임베딩 거리 가까움 → Reranker 도입
3. 복합 조건 질문       : 단일 Alias로 처리 불가 → LLM Query Rewriting
```

실패 케이스 대부분은 Alias Mapping 확장 또는 Cross Encoder Reranker 도입으로 해결 가능하다.

---

## 10-3. Top-K 선정 근거

Top-K 값을 고정하지 않고 K 값별 성능을 실험하여 최적값을 선정하였다.

### 실험 방법

```python
for k in [1, 3, 5, 10]:
    evaluate(k)
```

### 실험 결과 (Hard 50 기준)

| K  | Hit@K | Recall@K | 비고             |
| -- | ----: | -------: | -------------- |
| 1  | 0.920 |    0.920 | 1등 정답률         |
| 3  | —     |    0.980 | 거의 모든 정답 포함    |
| 5  | —     |    1.000 | **정답 100% 포함** |
| 10 | —     |    1.000 | K=5와 동일        |

### 해석

K=5 이후 Recall@K 성능 증가 없음이 확인되었다.

K=10으로 늘려도 검색 성능은 동일하지만, LLM에 입력되는 Context 길이가 2배로 늘어나 다음 문제가 발생한다.

- 불필요한 Context 포함 → LLM 답변 품질 저하 가능성
- 토큰 비용 증가
- 처리 속도 저하

따라서 **Top-5를 최적값으로 선정**하였다.

```text
K=5 선정 이유

Recall@5 = 1.000  (정답 문서 100% 포함)
K=10 대비 Context 길이 절반
LLM 입력 품질 유지
```

---

## 11. 성능 개선 과정 및 시행착오

### 11-1. Metadata 누락 및 형식 비표준화 문제

**초기 상태**

데이터 구축 초기에 두 가지 문제가 동시에 발생하였다.

첫째, algorithm_key 자체가 누락된 문서가 다수 존재하였다.

```text
algorithm_key: (없음)
```

둘째, algorithm_key가 존재하더라도 형식이 제각각이었다.

```text
DFS
dfs
DepthFirstSearch
깊이우선탐색
```

**문제**

Metadata Filter가 정상 동작하지 않았다. 동일한 알고리즘을 가리키는 문서임에도 불구하고 key 표현이 달라 필터링 결과에서 누락되었다.

```json
{
  "algorithm_key": {
    "$in": ["dfs"]
  }
}
```

위 필터는 `DFS`, `DepthFirstSearch`를 가진 문서를 걸러내지 못한다.

**해결**

1. algorithm_key 전체 소문자 + 언더스코어 형식으로 표준화

```text
dfs
bfs
dijkstra
heap
priority_queue
binary_search
```

2. `fix_algorithm_key_metadata.py` 작성하여 누락 122건 일괄 수정

```text
fixed_embedding_metadata = 79
fixed_embeddings_queue   = 43
총 수정 건수             = 122건
```

**결과**

```text
remaining_blank_algorithm_key = 0

Metadata Filter 정확도 향상
```

---

### 11-2. Chunking 전략 선정

**초기 시도 : Fixed Chunking**

```python
chunk_fixed(text, size=1000, overlap=200)
```

문서 구조를 고려하지 않고 길이 기준으로 분할하기 때문에 개념 설명과 코드가 서로 다른 Chunk로 분리되었다.

```text
DFS 설명  → Chunk 1
DFS 코드  → Chunk 2   ← 문맥 손실
```

**두 번째 시도 : Recursive Chunking**

```python
chunk_recursive()
```

헤더 → 문단 → 문장 순으로 분할하여 일부 문맥은 유지되었지만, 알고리즘 문서 특유의 개념 → 예제 → 풀이 → 코드 구조를 충분히 반영하지 못하였다.

**최종 선택 : Adaptive Chunking**

알고리즘 학습 문서는 다음 구조를 가진다.

```text
개념
↓
예제
↓
풀이 전략
↓
코드
```

Adaptive Chunking은 `##`, `###` 섹션 단위로 분할하기 때문에 이 구조를 자연스럽게 유지한다.

선택 이유:

1. **문맥 보존** : 개념과 관련 설명이 같은 Chunk 안에 함께 유지된다.
2. **코드 보존** : 코드 블록(` ``` `)이 중간에 잘리지 않고 완전한 형태로 보존된다.

결과:

```text
DFS 전체 문서 (혼합)
↓
DFS 개념       → Chunk 1
DFS 시간복잡도  → Chunk 2
DFS 코드       → Chunk 3
DFS 예제       → Chunk 4
```

---

### 11-3. Hard Query 평가셋 구축

**초기 평가셋 형태**

처음 설계한 평가셋은 알고리즘 이름을 직접 포함한 질문으로 구성하였다.

```text
DFS란?
Heap이란?
Dijkstra란?
```

**문제**

알고리즘 이름이 질문에 포함되어 있으면 단순 키워드 매칭으로도 검색이 가능하다. 실제 사용자 질문과 거리가 멀어 평가의 의미가 없었다.

```text
알고리즘 이름 포함 → 너무 쉬움 → 실사용 환경과 다름
```

**해결**

알고리즘 이름을 제거하고 문제 상황 중심으로 질문을 재작성하였다.

```text
Before : DFS란?
After  : 연결된 모든 노드를 한 번씩 방문하려면?

Before : Heap이란?
After  : 항상 가장 작은 값을 빠르게 꺼내야 해.

Before : Dijkstra란?
After  : 출발지에서 각 노드까지의 최단 거리를 구하려면?
```

최종 Hard Query 예시:

```text
급한 순서대로 처리해야 하는 작업이 있어.
자동완성 기능을 구현하고 싶어.
정렬된 배열에서 원하는 값을 빠르게 찾아야 해.
구간 합을 여러 번 빠르게 구하고 업데이트도 해야 해.
두 노드가 같은 집합인지 확인하려면?
```

**결과**

실사용 환경과 유사한 평가가 가능해졌다. Hard Query 기준 성능이 의미 있는 지표로 기능하게 되었다.

---

### 11-4. Vector Search 오탐

**문제**

단순 Vector Search는 질문 의도를 파악하지 못하였다.

```text
질문   : 급한 순서대로 작업을 처리하려면?
결과   : queue / queue / queue
```

`급한`이라는 표현이 `queue(대기열)`와 의미적으로 가깝게 임베딩되었기 때문이다. 실제 의도인 `priority_queue` 또는 `heap` 문서는 검색되지 않았다.

**해결**

Alias Mapping으로 자연어 표현을 알고리즘 개념으로 변환한 뒤 Metadata Filter를 적용하였다.

```text
급한 순서   → priority_queue, heap
가장 작은 값 → heap
정렬된 배열  → binary_search
자동완성    → trie
구간 합     → segment_tree
집합 합치기  → union_find
```

```json
{
  "algorithm_key": {
    "$in": ["priority_queue", "heap"]
  }
}
```

**결과**

```text
Before : queue / queue / queue
After  : priority_queue / priority_queue / heap
```

---

### 11-5. Retrieval Planner 도입

**배경**

11-4의 오탐 문제를 해결하면서 단순 필터 추가만으로는 한계가 있음을 확인하였다. 질문 의도 분석, 알고리즘 후보 추출, 필터 생성을 통합적으로 처리하는 구조가 필요하였다.

**도입 전 구조**

```text
질문
↓
Embedding
↓
Vector Search
↓
Top-K
```

Hard Query Hit@1 = 42%

**도입 후 구조**

```text
질문
↓
Retrieval Planner
  - 질문 정규화
  - Intent 분석
  - Alias Mapping
  - Algorithm Candidate 추출
↓
Metadata Filter 생성
↓
MMR Search
↓
Document Voting
↓
Failure Detection → Fallback Query Rewrite
↓
Top-K Context 반환
```

**결과**

```text
Hard Query

Hit@1    : 42% → 92%
Recall@5 : 64% → 100%
MRR      : 0.509 → 0.953
```

---

### 11-6. 임베딩 모델 선택

**초기 고민**

임베딩 모델 후보를 비교하였다.

| 모델                  | 장점              | 단점              |
| ------------------- | --------------- | --------------- |
| OpenAI Embedding    | 성능 우수           | 비용 발생, API 의존   |
| all-MiniLM-L6-v2    | 가볍고 빠름, 무료      | 영어 중심           |
| BGE-M3              | 다국어, 한국어 성능 우수  | 상대적으로 무거움       |

**선택**

```text
sentence-transformers/all-MiniLM-L6-v2
```

선택 이유:

- 로컬 환경에서 비용 없이 사용 가능
- RAG 실험 및 반복 테스트에 적합한 속도
- 알고리즘 문서는 영어 키워드 혼용이 많아 MiniLM으로도 충분한 성능 확보

향후 BGE-M3 또는 multilingual-e5-large로 교체 실험을 계획하고 있다.

---

### 11-7. ChromaDB 저장 문제

**문제**

Chunk 생성은 완료되었지만 검색 시 결과가 반환되지 않는 문제가 발생하였다.

```text
Chunk 생성 완료
↓
검색 시 결과 없음
```

원인을 분석한 결과 ingest 시 사용한 Collection 이름과 검색 시 참조하는 Collection 이름이 일치하지 않았다.

```text
저장 : algorithm_docs
검색 : algorithms          ← 이름 불일치

오류 : Collection does not exist
```

**해결**

실제 저장된 Collection 이름을 확인한 뒤 경로와 이름을 통일하였다.

```python
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("algorithm_docs")
```

**결과**

검색 정상 동작 확인. 이후 Collection 이름과 경로를 환경 변수로 관리하여 동일한 문제가 재발하지 않도록 하였다.

---

### 11-8. 평가 지표 선정

**초기 고민**

처음에는 일반적인 분류 문제처럼 정확도(Accuracy)를 사용하려 하였다.

```text
정답 문서를 찾았으면 1
못 찾았으면 0
```

**문제**

RAG 시스템의 목적은 정답을 직접 생성하는 것이 아니라 정답 문서를 검색하는 것이다. 따라서 단순 Accuracy는 두 가지 측면을 반영하지 못하였다.

- 1등으로 찾은 경우와 5등으로 찾은 경우가 동일하게 처리됨
- 여러 후보 중 몇 번째에 정답이 있는지를 반영하지 못함

**해결**

RAG Retrieval에 적합한 3가지 지표를 채택하였다.

| 지표       | 의미                       |
| -------- | ------------------------ |
| Hit@1    | 1등 결과가 정답인 비율             |
| Recall@5 | Top-5 안에 정답이 포함된 비율       |
| MRR      | 정답이 몇 번째 순위에 있는지 평균적으로 반영 |

Recall@5를 핵심 지표로 설정한 이유: Top-5 안에 정답 문서가 있으면 LLM이 이를 Context로 활용하여 올바른 답변을 생성할 수 있기 때문이다.

---

## 12. 왜 이 방식이 LLM 연결에 좋은가

LLM은 입력으로 들어가는 Context 품질에 크게 영향을 받습니다.

엉뚱한 문서가 들어가면 LLM은 그 문서를 기반으로 잘못된 답변을 만들 수 있습니다.

Retrieval V1은 다음을 보장합니다.

1. 질문을 알고리즘 후보로 변환한다.
2. 관련 없는 문서를 필터링한다.
3. Query Expansion으로 검색 가능성을 높인다.
4. MMR로 다양한 chunk를 가져온다.
5. Document Voting으로 문서 단위 안정성을 높인다.
6. Failure Detection과 Fallback Rewrite로 검색 실패를 복구한다.
7. LLM에 넣을 Top-K Context 품질을 높인다.

---

## 13. 평가표 대응 항목

| 항목                  | 적용 여부 | 비고                              |
| ------------------- | ----- | ------------------------------- |
| 정규표현식 기반 전처리        | 완료    | 텍스트 정규화, 공백/특수문자 정리             |
| KoNLPy 형태소 분석       | 완료    | TF-IDF baseline 전처리             |
| 불용어 처리              | 완료    | 조사, 접속어 등 의미 약한 단어 제거           |
| TF-IDF 희소 표현        | 완료    | Baseline 검색 성능 측정               |
| Dense Embedding      | 완료    | SentenceTransformer 기반 embedding |
| VectorDB (ChromaDB) | 완료    | 문서 저장 및 검색                      |
| Hit@1 평가            | 완료    | Basic / Hard 평가셋 모두 적용          |
| Recall@5 평가         | 완료    | Basic / Hard 평가셋 모두 적용          |
| MRR 평가              | 완료    | Basic / Hard 평가셋 모두 적용          |
| Failure Detection    | 완료    | 검색 실패 정의 및 fallback 조건 활용       |
| Fallback Query Rewrite | 완료 | 검색 실패 시 질문 재작성 후 재검색            |

---

## 14. 프로젝트 차별성

일반 RAG:

```text
질문
→ 벡터 검색
→ 유사 문서 반환
```

우리 Retrieval V1:

```text
질문
↓
알고리즘 의도 추론
↓
Metadata Filtering
↓
Query Expansion
↓
MMR Search
↓
Document Voting
↓
Failure Detection
↓
Fallback Query Rewrite
↓
학습용 Context 반환
```

핵심 차별점: 사용자의 자연어 문제 상황을 알고리즘 개념으로 연결하는 Retrieval Planner를 설계하였다.

예시:

```text
급한 순서대로 처리      → priority_queue
정렬된 배열에서 빠르게 찾기 → binary_search
자동완성 기능          → trie
구간 질의             → segment_tree
집합을 합치고 연결 여부 확인 → union_find
```

---

## 15. 향후 개선 방향

현재 Retrieval V1은 평가셋 기준으로 목표 성능을 달성했습니다.

향후 고도화 단계에서는 다음을 추가할 수 있습니다.

1. 한국어/코딩 문서 특화 임베딩 모델 비교
   - BGE-M3
   - multilingual-e5-large

2. BM25 Hybrid Retrieval
   - Dense Search와 Keyword Search 결합

3. Cross Encoder Reranker
   - Top-20 후보를 더 정밀하게 재정렬

4. LLM 기반 Query Rewriting
   - 규칙 기반 Planner를 LLM 기반 Planner로 확장

5. Graph RAG 확장
   - 알고리즘 간 관계를 그래프로 연결

6. 평가셋 확장
   - 더 다양한 문제 상황, 오타, 복합 질문 추가

---

## 16. 배운 점

RAG 성능은 단순히 좋은 임베딩 모델만으로 결정되지 않았다.

실제로는 다음 요소들이 Retrieval 성능에 큰 영향을 미쳤다.

1. 전처리 품질
2. 코드 블록 보존
3. 문서 구조 유지
4. Semantic Chunking (Adaptive Chunking)
5. Metadata 설계

특히 알고리즘 학습 데이터는 일반 문서와 달리 개념, 예제, 코드 구조가 명확하기 때문에 문서 구조를 유지하는 청킹 전략이 중요하다는 것을 확인할 수 있었다.

---

## 17. 내 역할 최종 요약

나는 코딩테스트 학습 플랫폼의 RAG Retrieval 파트를 담당했습니다.

구체적으로는 알고리즘 문서 수작업 큐레이션 및 품질 검수, 전처리, 메타데이터 설계 및 표준화, 청킹, 임베딩, ChromaDB 저장, Retrieval V1 설계, Metadata Filtering, Query Expansion, MMR Search, Document Voting, Failure Detection, Fallback Query Rewrite, KoNLPy/TF-IDF baseline, Basic/Hard 평가까지 수행했습니다.

최종 데이터 구축 결과:

```text
전체 파일 수          : 392개
Markdown 문서 수      : 289개
최종 사용 문서 수       : 201개
총 Chunk 수          : 1686개
Algorithm Key 수     : 110개
algorithm_key 결측치  : 122건 → 0건
```

최종 Hard 50 평가 결과:

```text
Hit@1    : 0.420 → 0.920
Recall@5 : 0.640 → 1.000
MRR      : 0.509 → 0.953
```

---

## 예상 질문

**Q. 왜 Recall@5를 핵심 지표로 선택했나요?**

Top-K 실험 결과 K=5에서 Recall이 1.000에 도달하고 K=10과 동일한 성능을 보였다. K를 늘리면 Context 길이만 증가하여 LLM 답변 품질 저하 및 토큰 비용 문제가 발생하므로 K=5를 최적값으로 선정하였다.

**Q. 가장 큰 성능 향상 요인은?**

Ablation Study 결과 Alias Mapping이 단일 기능 중 가장 큰 기여 요소였다. Hit@1 기준 단독으로 +26.0%p 향상되었다. 이후 Metadata Filter +16.0%p, Document Voting +8.0%p가 누적되어 최종 Hard Hit@1 92%를 달성하였다.

**Q. Ablation Study를 어떻게 설계했나요?**

`retrieval_v1.py`에 기능별 플래그(`USE_ALIAS`, `USE_METADATA`, `USE_VOTING`)를 추가하고 Hard 50 평가셋 기준으로 하나씩 활성화하며 측정하였다. 이를 통해 각 기능의 독립적 기여도를 정량적으로 확인하였다.

**Q. 실패 케이스는 어떻게 분석했나요?**

정답을 찾지 못한 케이스를 유형별로 분류하였다. 주요 실패 유형은 Alias 부재, 유사 알고리즘 간 임베딩 거리 혼동, 복합 조건 질문 처리 한계 세 가지였다. 대부분은 Alias 확장 또는 Cross Encoder Reranker 도입으로 해결 가능하다.

**Q. 왜 Adaptive Chunking을 선택했나요?**

두 가지 핵심 이유가 있다. 첫째 문맥 보존 측면에서, 알고리즘 문서의 개념과 관련 설명이 같은 Chunk 안에 유지된다. 둘째 코드 보존 측면에서, 코드 블록이 중간에 잘리지 않고 완전한 형태로 보존된다.

**Q. 데이터를 어떻게 수집했나요?**

단순 크롤링이 아닌 수작업 큐레이션 중심으로 수행하였다. 7가지 선정 기준을 적용하여 392개 파일 중 최종 201개 문서를 선별하였다. 서비스 품질이 데이터 품질에 크게 의존하기 때문에 수작업 검수를 병행하였다.

**Q. 가장 기억에 남는 시행착오는?**

Hard Query 평가셋 구축 과정이다. 처음에는 "DFS란?", "Heap이란?"처럼 알고리즘 이름이 포함된 질문으로 평가셋을 구성했는데, 실제 사용자 질문과 거리가 멀었다. "급한 순서대로 처리해야 하는 작업이 있어" 형태로 전환하고 나서야 의미 있는 평가가 가능해졌다.

**Q. Metadata 누락을 어떻게 발견했나요?**

Metadata Filtering 결과가 예상보다 적게 나오는 것을 확인하고 ChromaDB에 저장된 메타데이터를 직접 조회하여 발견하였다. algorithm_key 누락 및 형식 불일치가 총 122건이었으며 `fix_algorithm_key_metadata.py`를 작성하여 일괄 수정하고 최종 결측치 0건을 달성하였다.

**Q. 향후 개선 계획은?**

실패 케이스 분석에서 확인된 의미 중복 문제 해결을 위해 Cross Encoder Reranker를 우선 적용할 계획이다. 이후 BM25 Hybrid Retrieval, BGE-M3 임베딩 교체, LLM 기반 Query Rewriting 순으로 고도화할 예정이다.
