# SKN29-3RD-1Team

## 프로젝트명
코딩테스트 학습자를 위한 **RAG 기반 알고리즘 문제 풀이 AI 지원 시스템**

<br/>

## 프로젝트 개요

알고리즘 문제를 입력하면 AI가 풀이 전략을 수립하고, 코드를 직접 생성·실행·검증한 뒤, 관련 알고리즘 개념 문서를 검색하여 접근법·알고리즘 설명·코드 해설이 포함된 최종 답변을 제공하는 시스템입니다.

단순한 LLM 응답이 아니라 **코드 실행 검증 → 자동 디버깅 → RAG 기반 개념 보강** 파이프라인을 통해 학습자가 알고리즘을 이해하고 실력을 키울 수 있도록 돕습니다.

<br/>

## 프로젝트 기간

**2026/06/08 ~ 2026/06/17** (총 10일)

<br/>

## 프로젝트 구성원 및 역할

| 김진욱 | 양정현 | 김재홍 | 김정민 |
| :---: | :---: | :---: | :---: |
| <img src="./docs/image/김진욱.png" height="150"> | <img src="./docs/image/양정현.png" height="150"> | <img src="./docs/image/김재홍.png" height="150"> | <img src="./docs/image/김정민.png" height="150"> |
| [@](https://github.com/) | [@](https://github.com/) | [@](https://github.com/) | [@min1i](https://github.com/min1i) |
| LangGraph Agent 설계 | RAG Retrieval 구현 | Qwen3 모델 파인튜닝 | 데이터 구축 · 청킹 · 평가 |
|- | Retrieval 설계 | LoRA 어댑터 학습 | LoRA 어댑터 학습 |
| - | - | 모델 성능 평가 | 모델 성능 평가 |

<br/>

## 프로젝트 배경 및 필요성

코딩테스트는 취업 관문에서 필수 관문이 되었지만, 학습자들은 다음과 같은 어려움을 겪고 있습니다.

- 알고리즘 이름을 몰라 검색 자체가 어려움
- 코드를 짜더라도 어디가 틀렸는지 파악하기 어려움
- 정답 코드를 보더라도 왜 그 알고리즘을 쓰는지 이해하기 어려움

기존 풀이 사이트나 챗봇은 **코드를 직접 실행하여 검증하거나, 학습자 맞춤형 개념 설명을 연결하지 못한다**는 한계가 있었습니다.

본 프로젝트는 다음 세 가지 기술을 결합하여 이 문제를 해결합니다.

- **LangGraph 에이전트**: 문제 분석 → 코드 생성 → 실행 검증 → 자동 디버깅을 하나의 흐름으로 처리
- **RAG(Retrieval-Augmented Generation)**: 알고리즘 개념 문서를 검색하여 답변에 근거 제공
- **Qwen3 LoRA 파인튜닝**: 알고리즘 사고 과정과 쿼리 생성에 특화된 로컬 모델 활용

<br/>

## 프로젝트 목표

1. **알고리즘 문제 자동 풀이**: LangGraph 기반 에이전트가 문제 분석부터 최종 답변까지 자동 처리
2. **코드 실행 기반 정확성 검증**: 생성된 코드를 직접 실행하여 정답 여부를 확인하고 오류 발생 시 자동 디버깅
3. **RAG 기반 개념 보강**: 201개 알고리즘 학습 문서·1686개 청크를 구축하여 관련 개념을 LLM 컨텍스트로 제공
4. **Retrieval 성능 최적화**: Alias Mapping, Metadata Filtering, MMR Search, Document Voting 등을 조합하여 Hard Query Hit@1 90% 달성
5. **Qwen3 LoRA 파인튜닝**: 사고 과정(think)과 쿼리 생성(query) 단계에 특화된 경량 로컬 모델 구축
6. **FastAPI 기반 서비스 제공**: RESTful API로 전체 파이프라인을 단일 엔드포인트로 제공

<br/>

## 기대 효과

| 대상 | 기대 효과 |
|------|-----------|
| **코딩테스트 학습자** | 알고리즘 이름을 몰라도 문제 상황으로 관련 개념을 찾고, 코드 설명까지 한 번에 받을 수 있음 |
| **교육 플랫폼** | 학습자 질문에 즉시 대응하는 AI 튜터 기능으로 학습 효율 향상 |
| **AI 개발자** | RAG + LangGraph + LoRA를 결합한 실전 AI 파이프라인 구축 사례 참고 가능 |

<br/>

## 기술 스택

| 구분 | 기술 |
|------|------|
| 서버 | FastAPI, Uvicorn |
| 에이전트 | LangGraph |
| LLM (추론) | OpenAI GPT-4o-mini |
| LLM (로컬) | Qwen3-0.6B, PEFT/LoRA, Transformers, HuggingFace |
| 임베딩 | OpenAI text-embedding-3-small, sentence-transformers |
| 벡터DB | ChromaDB |
| RAG | LangChain, LangChain-Chroma, LangChain-OpenAI |
| 코드 실행 | FastMCP |
| 전처리 | KoNLPy, regex |
| 환경 | conda (Python 3.11), python-dotenv |

<br/>

## 주요 기능

| 기능 | 설명 |
|------|------|
| **문제 풀이 에이전트** | 알고리즘 문제를 입력하면 think → code → execute → debug → RAG → 최종 답변을 자동 생성 |
| **코드 생성 및 실행** | GPT-4o-mini가 파이썬 코드를 생성하고 인메모리 샌드박스로 직접 실행하여 정답 여부 검증 |
| **자동 디버깅** | 실행 결과가 기대값과 다를 경우 오류 원인 분석 후 코드 수정 (최대 3회 재시도) |
| **RAG 기반 개념 검색** | 코드 검증 성공 후 관련 알고리즘 개념 문서를 ChromaDB에서 검색하여 설명에 활용 |
| **Retrieval V1** | Alias Mapping, Metadata Filtering, Query Expansion, MMR Search, Document Voting, Failure Detection, Fallback Query Rewrite를 조합한 고도화 검색 |
| **Qwen3 LoRA** | 알고리즘 사고 과정과 RAG 쿼리 생성에 특화된 Qwen3-0.6B 파인튜닝 모델 |
| **Swagger UI** | `localhost:8000/docs` 에서 전체 API 엔드포인트 테스트 가능 |

<br/>

## 시스템 아키텍처

```text
┌──────────────────────────────────────────────────────────────┐
│                         사용자                                │
└────────────────────────────┬─────────────────────────────────┘
                             │ GET /health/solve?request=<문제>
                             ▼
┌──────────────────────────────────────────────────────────────┐
│                     FastAPI 서버                              │
│               app/main.py  (uvicorn)                         │
└────────────────────────────┬─────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────┐
│                   LangGraph Agent                            │
│               app/agent/solve_agent.py                       │
│                                                              │
│  think → generate_code → execute → [debug loop]             │
│       → algorithm_query → retrieve → final_answer            │
└──────────┬─────────────────────────────────┬────────────────┘
           │                                 │
           ▼                                 ▼
┌──────────────────────┐         ┌───────────────────────────┐
│      LLM 레이어       │         │       RAG 레이어           │
│                      │         │                           │
│  OpenAI GPT-4o-mini  │         │  Retrieval V1             │
│  (generate/debug/    │         │  ├── Alias Mapping        │
│   final_answer)      │         │  ├── Metadata Filtering   │
│                      │         │  ├── Query Expansion      │
│  Qwen3-0.6B + LoRA   │         │  ├── MMR Search           │
│  (think/query)       │         │  ├── Document Voting      │
└──────────────────────┘         │  └── Fallback Rewrite     │
                                 └──────────────┬────────────┘
                                                │
                                                ▼
                                 ┌───────────────────────────┐
                                 │       ChromaDB            │
                                 │  알고리즘 개념 문서        │
                                 │  201개 문서 / 1686 청크   │
                                 │  text-embedding-3-small   │
                                 └───────────────────────────┘
```

<br/>

## 데이터 구축 파이프라인

```text
알고리즘 학습 문서 (.md)
    ↓
run_ingest.py 실행
    ↓
splitter.py — 포맷 감지(A/B/C) + Adaptive Semantic Chunking
    ↓
embeddings.py — text-embedding-3-small 벡터화
    ↓
vector_store.py — ChromaDB 저장
    ↓
chroma_db/ (로컬 디렉토리)
```

| 항목 | 수량 |
|------|------|
| 문서 수 | 201개 |
| 청크 수 | 1686개 |
| 알고리즘 키 수 | 110종 |

<br/>

## RAG 성능 평가 결과

### Basic 50 평가

| 방식 | Hit@1 | Recall@5 | MRR |
|------|------:|---------:|----:|
| Plain Vector Search | 0.760 | 0.940 | 0.843 |
| Retrieval V1 | 0.880 | 1.000 | 0.940 |

### Hard 50 평가

| 방식 | Hit@1 | Recall@5 | MRR |
|------|------:|---------:|----:|
| Plain Vector Search | 0.420 | 0.640 | 0.509 |
| Retrieval V1 | 0.900 | 1.000 | 0.943 |

<br/>

## 디렉토리 구조

```text
SKN29-3RD-1Team/
│
├── app/
│   ├── main.py                     FastAPI 앱 진입점
│   ├── config.py                   환경변수 및 로깅 설정
│   ├── utils.py                    JSON 파싱 유틸
│   │
│   ├── api/
│   │   ├── health_router.py        헬스체크 및 solve 엔드포인트
│   │   └── rag_router.py           문서 저장 엔드포인트
│   │
│   ├── agent/
│   │   └── solve_agent.py          LangGraph 에이전트 그래프
│   │
│   ├── llm/
│   │   ├── openai_model.py         GPT-4o-mini 래퍼
│   │   ├── qwen_model.py           Qwen3 + LoRA 래퍼
│   │   └── prompts.py              프롬프트 템플릿 5종
│   │
│   ├── mcp/
│   │   └── tools.py                Python 코드 실행 도구 (FastMCP)
│   │
│   ├── rag/
│   │   ├── retrieval_v1.py         Retrieval V1 핵심 로직
│   │   ├── retriever.py            기본 retriever 래퍼
│   │   ├── vector_store.py         ChromaDB 연결
│   │   ├── embeddings.py           임베딩 모델
│   │   ├── splitter.py             Adaptive Chunking
│   │   ├── loader.py               문서 로더
│   │   ├── ingest.py               데이터 수집 파이프라인
│   │   └── konlpy_preprocessing.py 형태소 분석 (TF-IDF baseline용)
│   │
│   └── scripts/                    모델 학습 및 평가 스크립트
│       ├── train_qwen_adapter.py
│       ├── evaluate_base_*.py
│       └── evaluate_lora_*.py
│
├── evaluation/                     평가셋 및 TF-IDF baseline
│   ├── retrieval_dataset_expanded_50.json
│   ├── retrieval_hard_dataset_expanded_50.json
│   └── tfidf_baseline.py
│
├── docs/                           보고서
│   ├── RAG_Retrieval_최종보고서.md
│   ├── LangGraph_Agent_설계보고서.md
│   ├── 전체_시스템_아키텍처_보고서.md
│   └── 모델링_베이스_LoRA_결과_보고서.md
│
├── chroma_db/                      ChromaDB 저장소 (로컬, git 제외)
├── adapters/                       Qwen LoRA 어댑터 (로컬, git 제외)
├── run_ingest.py                   데이터 ingestion 실행 스크립트
├── eval_retrieval_v1.py            Basic 평가 실행 스크립트
├── eval_retrieval_v1_hard.py       Hard 평가 실행 스크립트
├── environment.yml                 conda 환경 정의
└── .env                            API 키 (git 제외)
```

<br/>

## 시작하기

### 1. 환경 세팅

```bash
conda env create -f environment.yml
conda activate skn3rd
```

### 2. 환경변수 설정

프로젝트 루트에 `.env` 파일을 생성합니다.

```env
OPENAI_API_KEY=sk-...
ENABLE_LOCAL_MODEL=True # False
```

### 3. 데이터 Ingestion (RAG 검색용 ChromaDB 구축)

알고리즘 학습 문서가 담긴 폴더 경로를 지정하여 실행합니다.

```bash
python run_ingest.py "알고리즘_자료_폴더_경로"
```

### 4. 서버 실행

```bash
uvicorn app.main:app --reload
```

### 5. API 확인

브라우저에서 Swagger UI로 전체 API를 테스트할 수 있습니다.

```
http://localhost:8000/docs
```

전체 파이프라인 실행 예시:

```
GET /health/solve?request=두 수의 합이 target이 되는 두 인덱스를 찾아라.
```

### 6. 시연용 Streamlit 실행
```
python -m streamlit run preview_app.py
```

### 7. RAG 성능 평가 (선택)

```bash
# Basic 50 평가
python eval_retrieval_v1.py

# Hard 50 평가
python eval_retrieval_v1_hard.py
```

<br/>

## 팀원 회고

**김진욱**: 

<br/>

**양정현**: 데이터 수집부터, 전처리, 청킹, 임베딩, 리트리빙에 대해 많이 공부하였고, 처음에는 뭔가 어려워서 이해도 안되고 gpt로 돌리면서 내용 이해하기 위해 많은 노력이 필요 했습니다. 막상 하니 공부도 되고 구조에 대한것과 많은 것을 배웠습니다. 아쉬운점은 너무 RAG 파트만 해서 다음에 파인튜닝 파트도 해보았으면 좋겠다라는 생각이 들었습니다.

<br/>

**김재홍**: 

<br/>

**김정민**: 데이터 수집부터 SFT 구성, 베이스 평가, LoRA 튜닝까지 모델링 파이프라인 전 과정을 직접 경험하였습니다. 수치가 좋아도 실제 응답을 눈으로 확인해야 한다는 것을 깨달았고, 경량 모델의 한계를 직접 마주하며 튜닝 방향을 스스로 판단하는 과정이 값진 경험이었던 것 같습니다.
