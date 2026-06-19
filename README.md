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
- **Qwen LoRA 파인튜닝**: 알고리즘 사고 과정과 쿼리 생성에 특화된 로컬 모델 활용

<br/>

## 프로젝트 목표

1. **알고리즘 문제 자동 풀이**: LangGraph 기반 에이전트가 문제 분석부터 최종 답변까지 자동 처리
2. **코드 실행 기반 정확성 검증**: 생성된 코드를 직접 실행하여 정답 여부를 확인하고 오류 발생 시 자동 디버깅
3. **RAG 기반 개념 보강**: 201개 알고리즘 학습 문서·1686개 청크를 구축하여 관련 개념을 LLM 컨텍스트로 제공
4. **Retrieval 성능 최적화**: Alias Mapping, Metadata Filtering, MMR Search, Document Voting 등을 조합하여 Hard Query Hit@1 90% 달성
5. **Qwen LoRA 파인튜닝**: 사고 과정(think)과 쿼리 생성(query) 단계에 특화된 경량 로컬 모델 구축
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
| LLM (로컬) | Qwen2.5-3B-Instruct, PEFT/LoRA, Transformers, HuggingFace |
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
| **Qwen LoRA** | 알고리즘 사고 과정과 RAG 쿼리 생성에 특화된 Qwen2.5-3B-Instruct 파인튜닝 모델 |
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
│  Qwen3-2.5B-Instruct │         │  ├── MMR Search           │
│   + LoRA             │         │  ├── Document Voting      │
│  (think/query)       │         │  └── Fallback Rewrite     │
└──────────────────────┘         └──────────────┬────────────┘
                                                │
                                                ▼
                                 ┌───────────────────────────┐
                                 │       ChromaDB            │
                                 │  알고리즘 개념 문서          │
                                 │  201개 문서 / 1686 청크     │ 
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

## sLLM 모델링 파이프라인
Pipeline Steps
| 단계	| 내용	| 주요 파일 |
|---|---:|---:|
| 데이터셋 구축	| OpenCodeReasoning 데이터셋을 학습용/평가용으로 분리	| `opencode_reasoning_train_4000.jsonl`, `opencode_reasoning_test_1000.jsonl` | 
| Base 모델 평가	| 튜닝 전 Qwen/Solar 계열 모델의 기본 생성 성능 측정	| evaluate_base_*.py | 
| LoRA/QLoRA 학습	| 제한된 GPU 환경에서 adapter 기반 파인튜닝 수행	| train_qwen_split_adapter.py | 
| 개선 모델 평가	| keyword, thinking, code, full 응답별 품질 평가	| evaluate_lora_*.py | 
| 결과 비교 분석	| Base와 LoRA 모델의 성능 차이, 실패 유형, 개선 방향 분석	| - | 

## sLLM 적용 개요

| 태스크 | 출력 형식 | 설명 |
|--------|----------|------|
| keyword | `## RAG Keywords` | 풀이에 필요한 알고리즘 키워드 6개 생성 |
| thinking | `## Thinking Process` | 문제 이해부터 구현 계획까지 5단계 사고과정 생성 |
| code | `## Final Answer` | 완전한 Python 풀이 코드 생성 |

## sLLM SFT 데이터 구성

| 파일 | 레코드 수 | 용도 |
|------|----------|------|
| `opencode_reasoning_train_4000.jsonl` | 4,000개 | LoRA 튜닝 학습 데이터 |
| `opencode_reasoning_test_1000.jsonl` | 1,000개 | 베이스 / LoRA 평가 데이터 |

# sLLM 베이스모델 및 LoRA튜닝 목표

최종 선정된 Qwen2.5-3B-Instruct 모델을 베이스로 LoRA SFT 튜닝을 적용한다.

| 지표 | 설명 | 베이스모델 목표 | LoRA튜닝 목표|
|------|------|------|------|
| `keyword_accuracy` | RAG 키워드 6개 형식 및 vocab 일치율 | 50% 이상 | 90% 이상 |
| `thinking_quality` | Thinking Process 5개 항목 완성도 | 50% 이상 | 90% 이상 |
| `code_quality` | 코드 문법 통과 및 형식 준수율 | 50% 이상 | 90% 이상 |
| `format_compliance` | 전체 응답 형식 준수율 | 50% 이상 | 70% 이상 |
| `response_stability` | 형식 + 내용 + 길이 모두 통과율 | 50% 이상 | 70% 이상 |
| `baseline_50_score` | 9개 항목 종합 점수 (50점 이상 통과) | 50점 이상 | 80점 이상 |

## LoRA 하이퍼파라미터 튜닝 기준

LoRA 튜닝은 BASE 모델과의 비교 공정성을 유지하기 위해 프롬프트는 변경하지 않고, 학습 하이퍼파라미터만 조정하였다.

| 항목 | 1차 설정 | 2차 설정 | 변경 목적 |
|---|---:|---:|---|
| learning rate | `2e-4` | `1e-4` | 업데이트 강도를 낮춰 BASE 지식 손상을 줄이고 학습 안정성 확보 |
| gradient accumulation | `8` | `16` | 실질 batch 크기를 늘려 학습 업데이트를 안정화 |
| keyword epochs | `1.0` | `1.0` | keyword 형식 안정성이 충분하여 변경하지 않음 |
| keyword max length | `768` | `768` | keyword 출력은 짧은 응답이므로 변경하지 않음 |
| keyword LoRA rank / alpha | `8 / 16` | `8 / 16` | 기존 keyword 튜닝 설정 유지 |
| thinking epochs | `3.0` | `2.0` | 과도한 형식 학습과 reasoning 압축을 줄임 |
| thinking max length | `768` | `1024` | 풀이 근거, 제약조건, 세부 reasoning이 잘리지 않도록 확장 |
| thinking LoRA rank / alpha | `8 / 32` | `16 / 32` | reasoning 표현 용량을 늘림 |
| code epochs | `1.0` | `1.0` | code는 epoch보다 adapter 용량 조정 중심으로 개선 |
| code max length | `1024` | `1024` | 코드 출력 길이는 기존 설정 유지 |
| code LoRA rank / alpha | `8 / 16` | `16 / 32` | 코드 구조와 코드블록 안정성 개선 |

## sLLM 평가 요약

본 프로젝트에서는 모델 응답 품질을 한 가지 기준으로만 판단하지 않고, 서로 다른 관점의 평가 방식을 함께 사용하여 1~5점까지의 등급으로 OpenAI Chatgpt-5.5 모델이 판단하였다.

| 평가 방식 | 목적 |
|---|---|
| 형식 평가 | 모델이 요구한 출력 형식과 구조를 잘 지켰는지 확인 |
| Reference 기반 평가 | 모델 답변이 기준 정답 reasoning/code와 얼마나 내용적으로 일치하는지 확인 |
| 요구사항 기반 평가 | 모델 답변이 실제 문제 해결 요구사항을 만족하는지 확인 |

## 기존 자동 형식 평가 점수

| 평가 항목 | BASE | TUNING 1차 | TUNING 2차 |
|---|---:|---:|---:|
| keyword_accuracy | 50.00 | 100.00 | 100.00 |
| thinking_quality | 33.33 | 90.00 | 96.67 |
| code_quality | 76.67 | 96.67 | 100.00 |
| full avg_baseline_50_score | 69.63 | 89.26 | 90.00 |
| full format_compliance | 16.67 | 86.67 | 96.67 |
| full usable_response_rate | 40.00 | 96.67 | 100.00 |

## Reference 기반 평가 점수
| Stage | BASE | TUNING 1차 | TUNING 2차 |
|---|---:|---:|---:|
| keyword | 4.17 | 4.00 | 4.03 |
| thinking | 4.07 | 3.63 | 4.20 |
| code | 4.73 | 4.60 | 4.83 |
| full | 4.57 | 4.23 | 4.63 |

## 요구사항 기반 평가 점수

| Stage | BASE | TUNING 1차 | TUNING 2차 |
|---|---:|---:|---:|
| keyword | 3.10 | 3.33 | 3.17 |
| thinking | 4.60 | 4.43 | 4.67 |
| code | 4.93 | 4.93 | 5.00 |
| full | 4.87 | 4.73 | 4.83 |

## 평가 점수 종합 해석

| 관점 | 가장 우수한 모델 | 판단 근거 |
|---|---|---|
| 형식 안정성 | TUNING 2차 | keyword, thinking, code, full 형식 지표가 가장 높음 |
| Reference 내용 일치도 | TUNING 2차 | thinking, code, full에서 BASE를 소폭 상회 |
| 실제 문제 요구사항 충족 | BASE | full 요구사항 기반 점수가 가장 높음 |
| 튜닝 개선 흐름 | TUNING 2차 | 1차 대비 내용 일치도와 코드 안정성이 회복됨 |

최종적으로 TUNING 2차는 형식 안정성과 reference 기반 내용 일치도에서는 가장 좋은 결과를 보였고, 요구사항 기반 full 평가는 BASE와 거의 비슷하지만 아직 소폭 낮은 수준으로 평가되었다.

---


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
│   ├── data/
│   │   ├── opencode_reasoning_train_4000.jsonl   # 학습 데이터 (4,000개)
│   │   └── opencode_reasoning_test_1000.jsonl    # 평가 데이터 (1,000개)
│   │   
│   └── scripts/                    모델 학습 및 평가 스크립트
│       ├── qwen_base_model.py      Qwen2.5-3B-Instruct_BASE모델 생성
│       ├── train_qwen_adapter.py   1차 SFT LoRA Adapter 생성(형식 우선)
│       ├── train_qwen_split_adapter.py 2차 SFT LoRA Adapter 생성(형식 완화, 내용 우선)
│       ├── eval_common.py          형식 자체평가 공통코드
│       ├── evaluate_base_*.py      베이스모델 형식 자체평가 코드
│       ├── evaluate_lora_*.py      LoRA모델 형식 자체평가 코드
│       └── outputs/
│           ├── base_eval_split/    
│           │   ├── code/          (details.csv, details.json, summary.json)  코드 형식 평가 결과
│           │   ├── keyword/       (details.csv, details.json, summary.json)  키워드 형식 평가 결과  
│           │   ├── thinking/      (details.csv, details.json, summary.json)  사고과정 형식 평가 결과
│           │   ├── full/          (details.csv, details.json, summary.json)  전체 형식 평가 결과
│           │   └── final_summary.json      베이스모델 최종 요약
│           │
│           ├── lora_eval_split/
│           │   ├── code/          (details.csv, details.json, summary.json)  코드 형식 평가 결과  
│           │   ├── keyword/       (details.csv, details.json, summary.json)  키워드 형식 평가 결과
│           │   ├── thinking/      (details.csv, details.json, summary.json)  사고과정 형식 평가 결과
│           │   ├── full/          (details.csv, details.json, summary.json)  전체 형식 평가 결과
│           │   └── final_summary.json      1차 lora튜닝 최종 요약
│           │
│           └── lora_eval_split_tuning_2/
│               ├── code/          (details.csv, details.json, summary.json)  코드 형식 평가 결과
│               ├── keyword/       (details.csv, details.json, summary.json)  키워드 형식 평가 결과
│               ├── thinking/      (details.csv, details.json, summary.json)  사고과정 형식 평가 결과
│               ├── full/          (details.csv, details.json, summary.json)  전체 형식 평가 결과
│               └── final_summary.json     2차 lora튜닝 최종 요약
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
ENABLE_LOCAL_MODEL=True
```

> `ENABLE_LOCAL_MODEL=True`로 설정했을 시 아래 링크에서 Adapter 다운로드 후
> `app/llm/qwen_model.py`의 `ADAPTER_MAPPING` 경로 수정

[LoRA Adapter G-Drive](https://drive.google.com/drive/folders/1U3f4zt3wFRT2zX3OjdKIBITGnj6zlRIl)

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

# state dump 결과 보기
python -m app/agent/log/app.py

# solve_Qwen_Qwen2.5-3B-Instruct_gpt-4o-mini.jsonl 입력 or 업로드
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

**김진욱**: RAG 문서 구조 설계, 데이터 수집·정제, 리트리버 고도화 과정을 거치며 RAG 시스템에 대한 이해를 많이 넓힐 수 있었습니다. 또한 LoRA 튜닝을 직접 경험하면서 sLLM의 서비스 적용 가능성을 고민해볼 수 있었고, 동시에 모델의 한계와 평가 체계의 중요성도 크게 느낄 수 있었습니다. 전체적으로 LLM 시스템을 구성하고 개선하는 과정에 대해 한 단계 더 깊이 이해할 수 있었던 경험이었습니다.


<br/>

**양정현**: 데이터 수집부터, 전처리, 청킹, 임베딩, 리트리빙에 대해 많이 공부하였고, 처음에는 뭔가 어려워서 이해도 안되고 gpt로 돌리면서 내용 이해하기 위해 많은 노력이 필요 했습니다. 막상 하니 공부도 되고 구조에 대한것과 많은 것을 배웠습니다. 아쉬운점은 너무 RAG 파트만 해서 다음에 파인튜닝 파트도 해보았으면 좋겠다라는 생각이 들었습니다.

<br/>

**김재홍**: 로컬 sLLM 구현 및 파인튜닝을 경험하였습니다. 경량 모델 한계를 극복하기 위해 LoRA 하이퍼 파라미터 튜닝을 통해 응답 지표가 크게 개선되어 실 서비스 적용 가능성을 확인하였습니다. 추론 속도와 GPU 부담을 줄이기 위한 분리 생성, 템플릿 조립, 캐싱, 메모리 최적화 등과 같은 전략은 실무 연계 시 비용 절감으로 연결되는 점을 알게 되었습니다.

<br/>

**김정민**: 데이터 수집부터 SFT 구성, 베이스 평가, LoRA 튜닝까지 모델링 파이프라인 전 과정을 직접 경험하였습니다. 수치가 좋아도 실제 응답을 눈으로 확인해야 한다는 것을 깨달았고, 경량 모델의 한계를 직접 마주하며 튜닝 방향을 스스로 판단하는 과정이 값진 경험이었던 것 같습니다.
