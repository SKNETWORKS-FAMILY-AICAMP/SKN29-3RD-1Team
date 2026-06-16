# 모델링 베이스 LoRA 결과 보고서
> Qwen2.5 베이스 모델 평가 → LoRA 튜닝 → LoRA 평가 전체 파이프라인 정리

## 담당자
- 김재홍
- 김정민 

---

## 개요

코딩테스트 문제를 입력받아 학습자에게 풀이 사고과정을 설명하는 AI 튜터 모델을 구축한다.  
Qwen2.5-3B-Instruct 모델을 베이스로 LoRA SFT 튜닝을 적용하여 아래 세 가지 태스크를 수행하도록 학습시킨다.

| 태스크 | 출력 형식 | 설명 |
|--------|----------|------|
| keyword | `## RAG Keywords` | 풀이에 필요한 알고리즘 키워드 6개 생성 |
| thinking | `## Thinking Process` | 문제 이해부터 구현 계획까지 5단계 사고과정 생성 |
| code | `## Final Answer` | 완전한 Python 풀이 코드 생성 |

---

## 데이터셋

### 출처

**HuggingFace**: [nvidia/OpenCodeReasoning](https://huggingface.co/datasets/nvidia/OpenCodeReasoning)

NVIDIA가 공개한 코딩 문제 추론 데이터셋으로, 다양한 온라인 저지 플랫폼의 문제와  
GPT 계열 모델이 생성한 `<think>` 태그 기반 추론 과정 및 Python 풀이 코드를 포함한다.  
라이선스: **CC-BY-4.0** (상업적 이용 포함 자유롭게 사용 가능)

전체 데이터셋에서 총 **5,000개**를 샘플링하여 학습용 4,000개 / 평가용 1,000개로 분리하였다.

### 데이터 구성

| 파일 | 레코드 수 | 용도 |
|------|----------|------|
| `opencode_reasoning_train_4000.jsonl` | 4,000개 | LoRA 튜닝 학습 데이터 |
| `opencode_reasoning_test_1000.jsonl` | 1,000개 | 베이스 / LoRA 평가 데이터 |

### 데이터 구조

각 레코드는 아래 4개 필드로 구성된다.

```json
{
  "instruction": "Act as a coding learning test assistant...",
  "input": "코딩 문제 본문 (영어)",
  "output": "<think>\n추론 과정...\n</think>\n\n```python\n풀이 코드\n```",
  "metadata": {
    "id": "고유 식별자",
    "source": "codeforces",
    "license": "cc-by-4.0",
    "dataset": "code_contests",
    "difficulty": "8",
    "has_solution": true
  }
}
```

### 문제 출처 분포

| 플랫폼 | train (4,000개) | test (1,000개) |
|--------|----------------|----------------|
| Codeforces | 2,229개 (55.7%) | 553개 (55.3%) |
| Aizu | 637개 (15.9%) | 167개 (16.7%) |
| HackerEarth | 458개 (11.5%) | 111개 (11.1%) |
| AtCoder | 386개 (9.7%) | 87개 (8.7%) |
| CodeChef | 290개 (7.2%) | 82개 (8.2%) |

### LoRA 학습 데이터 변환 방식

`train_qwen_adapter.py`는 원본 데이터를 그대로 쓰지 않고 아래 방식으로 변환하여 학습한다.

1. `output` 필드에서 ` ``` python ... ``` ` 코드블록 추출
2. 문제 텍스트 + 코드 분석으로 ALLOWED_VOCAB 기반 키워드 6개 자동 선택
3. `<think>` 태그 내용을 추출하여 5개 항목 Thinking Process로 변환 (v2 개선 사항)
4. keyword / thinking / code 태스크별 학습 샘플로 분리 → 최대 **12,000개** 학습 샘플 생성


---

## 폴더 구조

```
app/
├── data/
│   ├── opencode_reasoning_train_4000.jsonl   # 학습 데이터 (4,000개)
│   └── opencode_reasoning_test_1000.jsonl    # 평가 데이터 (1,000개)
│
└── scripts/
    ├── qwen_base_model.py
    ├── eval_common.py
    ├── train_qwen_adapter.py
    ├── train_qwen_split_adapter.py
    │
    ├── evaluate_base_code.py
    ├── evaluate_base_full.py
    ├── evaluate_base_keyword.py
    ├── evaluate_base_summary.py
    ├── evaluate_base_thinking.py
    │
    ├── evaluate_lora_code.py
    ├── evaluate_lora_full.py
    ├── evaluate_lora_keyword.py
    ├── evaluate_lora_summary.py
    ├── evaluate_lora_thinking.py
    │
    └── outputs/
        ├── base_eval_split/
        │   ├── code/          (details.csv, details.json, summary.json)
        │   ├── full/          (details.csv, details.json, summary.json)
        │   ├── keyword/       (details.csv, details.json, summary.json)
        │   ├── thinking/      (details.csv, details.json, summary.json)
        │   └── final_summary.json
        │
        ├── lora_eval_split/
        │   ├── code/          (details.csv, details.json, summary.json)
        │   ├── full/          (details.csv, details.json, summary.json)
        │   ├── keyword/       (details.csv, details.json, summary.json)
        │   ├── thinking/      (details.csv, details.json, summary.json)
        │   └── final_summary.json
        │
        ├── lora_eval_split_tuning_2/
        │   ├── code/          (details.csv, details.json, summary.json)
        │   ├── full/          (details.csv, details.json, summary.json)
        │   ├── keyword/       (details.csv, details.json, summary.json)
        │   ├── thinking/      (details.csv, details.json, summary.json)
        │   └── final_summary.json
        │
        ├── split_adapters.zip          # 1차 실험 adapter (1.5B + 3B)
        └── split_adapters_tuning_2.zip # 최종 adapter (3B, thinking 개선)
```

> ⚠️ **주의**: `evaluate_lora_*.py`가 `evaluate_base_*.py`를 직접 import하므로  
> 두 파일 그룹을 서로 다른 폴더로 분리하면 import 오류가 발생한다. 반드시 같은 폴더에 두어야 한다.

---

## 각 파일 역할

### 공통 기반 파일

| 파일 | 역할 |
|------|------|
| `eval_common.py` | 모든 평가 스크립트가 공유하는 공통 로직. 데이터 로딩, 모델 실행(BaseModelRunner / LoraModelRunner), 결과 저장, ALLOWED_VOCAB 정의 포함 |
| `qwen_base_model.py` | Qwen 베이스 모델과 토크나이저 로딩 함수 모음. GPU/CPU 장치 선택 포함 |

### 튜닝 파일

| 파일 | 역할 |
|------|------|
| `train_qwen_adapter.py` | keyword / thinking / code 태스크를 하나의 LoRA adapter로 합쳐서 학습. `--task-mix` 옵션으로 태스크 선택 가능 |
| `train_qwen_split_adapter.py` | keyword / thinking / code 각각 별도의 LoRA adapter로 분리 학습. `train_qwen_adapter.py`를 태스크별로 자동 호출 |

### 베이스 모델 평가 파일

| 파일 | 역할 |
|------|------|
| `evaluate_base_keyword.py` | RAG 키워드 6개 생성 능력 평가. ALLOWED_VOCAB 일치율, snake_case, 중복 여부 검사 |
| `evaluate_base_thinking.py` | Thinking Process 5개 항목 생성 능력 평가. 라벨 정확도, 영어 여부, 불필요 섹션 포함 여부 검사 |
| `evaluate_base_code.py` | Final Answer 코드 생성 능력 평가. 문법 통과율, 코드블록 형식, placeholder 포함 여부 검사 |
| `evaluate_base_full.py` | keyword / thinking / code 결과를 조합하여 전체 응답 품질 평가. 위 세 파일 실행 후 실행 |
| `evaluate_base_summary.py` | keyword / thinking / code / full 네 단계 결과를 `final_summary.json`으로 요약 |

### LoRA 모델 평가 파일

| 파일 | 역할 |
|------|------|
| `evaluate_lora_keyword.py` | LoRA adapter 적용 후 RAG 키워드 생성 능력 평가 |
| `evaluate_lora_thinking.py` | LoRA adapter 적용 후 Thinking Process 생성 능력 평가 |
| `evaluate_lora_code.py` | LoRA adapter 적용 후 코드 생성 능력 평가 |
| `evaluate_lora_full.py` | LoRA 결과 조합 후 전체 응답 품질 평가 |
| `evaluate_lora_summary.py` | LoRA 평가 결과 최종 요약 |

---

## 실행 순서

### STEP 0. 패키지 설치

```bash
pip install transformers torch accelerate peft trl datasets -q
```

### STEP 1. 베이스 모델 평가 (기준선 측정)

```bash
cd app/scripts

python evaluate_base_keyword.py --limit 30
python evaluate_base_thinking.py --limit 30
python evaluate_base_code.py --limit 30
python evaluate_base_full.py --limit 30
python evaluate_base_summary.py
```

> `--limit` 숫자를 높일수록 더 정확한 평가가 가능하다. 100 권장.

### STEP 2. LoRA 튜닝

#### 방법 A — 단일 adapter (빠름, 권장)

```bash
python train_qwen_adapter.py \
    --task-mix keyword,thinking,code \
    --epochs 2 \
    --limit 0
```

#### 방법 B — 태스크별 분리 adapter (각 태스크 성능 극대화)

```bash
python train_qwen_split_adapter.py \
    --tasks keyword,thinking,code
```

### STEP 3. LoRA 모델 평가

```bash
python evaluate_lora_keyword.py --limit 100
python evaluate_lora_thinking.py --limit 100
python evaluate_lora_code.py --limit 100
python evaluate_lora_full.py --limit 100
python evaluate_lora_summary.py
```

---

## 주요 평가 지표

| 지표 | 설명 | 목표 |
|------|------|------|
| `keyword_accuracy` | RAG 키워드 6개 형식 및 vocab 일치율 | 90% 이상 |
| `thinking_quality` | Thinking Process 5개 항목 완성도 | 90% 이상 |
| `code_quality` | 코드 문법 통과 및 형식 준수율 | 90% 이상 |
| `format_compliance` | 전체 응답 형식 준수율 | 70% 이상 |
| `response_stability` | 형식 + 내용 + 길이 모두 통과율 | 70% 이상 |
| `baseline_50_score` | 9개 항목 종합 점수 (50점 이상 통과) | 80점 이상 |

---

## LoRA Adapter 저장 현황

### split_adapters.zip — 1차 실험 (Qwen2.5-1.5B + 3B 비교)

```
split_adapters/
├── qwen2.5-1.5b-keyword-lora/
├── qwen2.5-1.5b-thinking-lora/
├── qwen2.5-1.5b-code-lora/
├── qwen2.5-3b-keyword-lora/
├── qwen2.5-3b-thinking-lora/
└── qwen2.5-3b-code-lora/
```

- 1.5B와 3B 두 모델에 대해 keyword / thinking / code 태스크별 분리 adapter를 각각 학습
- 모델 크기에 따른 성능 비교 실험용
- base model: `Qwen/Qwen2.5-1.5B-Instruct`, `Qwen/Qwen2.5-3B-Instruct`
- PEFT 버전: 0.14.0

### split_adapters_tuning_2.zip — 최종 (Qwen2.5-3B, thinking 개선)

```
split_adapters_tuning_2/
├── qwen2.5-3b-keyword-lora/
├── qwen2.5-3b-thinking-lora/
└── qwen2.5-3b-code-lora/
```

- 1차 실험의 Thinking 고정 템플릿 문제를 개선한 최종 adapter
- `<think>` 태그 기반 문제별 사고과정을 추출하여 학습 데이터 품질 향상
- base model: `Qwen/Qwen2.5-3B-Instruct`
- PEFT 버전: 0.14.0
- **서비스 연동 시 이 adapter 사용 권장**

### Adapter 선택 기준

| 목적 | 사용할 adapter |
|------|---------------|
| 최종 서비스 연동 | `split_adapters_tuning_2/qwen2.5-3b-*` |
| 1.5B 경량 모델 테스트 | `split_adapters/qwen2.5-1.5b-*` |
| 1차/2차 성능 비교 | `split_adapters/qwen2.5-3b-*` vs `split_adapters_tuning_2/qwen2.5-3b-*` |

---

## 평가 결과 (실험 기록)

> 평가 샘플 수: 30개 / 베이스 모델: Qwen2.5-3B-Instruct

### keyword (RAG 키워드 생성)

| 지표 | 베이스 | LoRA v1 | LoRA v2 (최종) |
|------|--------|---------|----------------|
| keyword_accuracy | 50.0% | 100.0% | **100.0%** |
| normalized_keyword_accuracy | 80.0% | 100.0% | **100.0%** |
| rag_exact_6_rate | 73.33% | 100.0% | **100.0%** |
| allowed_vocab_accuracy | 50.0% | 100.0% | **100.0%** |
| not_truncated_rate | 83.33% | 100.0% | **100.0%** |
| avg_generation_seconds | 1.29s | 1.79s | 1.75s |

### thinking (사고과정 생성)

| 지표 | 베이스 | LoRA v1 | LoRA v2 (최종) |
|------|--------|---------|----------------|
| thinking_quality | 33.33% | 90.0% | **96.67%** |
| label_accuracy | 96.67% | 100.0% | **100.0%** |
| no_extra_section_rate | 33.33% | 90.0% | **96.67%** |
| concise_rate | 96.67% | 100.0% | **100.0%** |
| not_truncated_rate | 96.67% | 100.0% | **100.0%** |
| avg_generation_seconds | 7.71s | 11.77s | 12.76s |

> v1 → v2에서 thinking_quality 90% → 96.67%로 개선.  
> `<think>` 태그 기반 문제별 사고과정 학습 데이터 개선 효과.

### code (코드 생성)

| 지표 | 베이스 | LoRA v1 | LoRA v2 (최종) |
|------|--------|---------|----------------|
| code_quality | 76.67% | 96.67% | **100.0%** |
| strict_code_quality | 66.67% | 96.67% | **100.0%** |
| syntax_pass_rate | 76.67% | 96.67% | **100.0%** |
| one_code_block_rate | 76.67% | 96.67% | **100.0%** |
| ends_after_code_block | 70.0% | 96.67% | **100.0%** |
| not_truncated_rate | 83.33% | 96.67% | **100.0%** |
| avg_generation_seconds | 9.22s | 16.03s | 14.95s |

### full (전체 응답 조합)

| 지표 | 베이스 | LoRA v1 | LoRA v2 (최종) |
|------|--------|---------|----------------|
| format_compliance | 16.67% | 86.67% | **96.67%** |
| relaxed_format_compliance | 40.0% | 96.67% | **100.0%** |
| usable_response_rate | 40.0% | 96.67% | **100.0%** |
| baseline_50_pass_rate | 70.0% | 96.67% | **100.0%** |
| avg_baseline_50_score | 69.63점 | 89.26점 | **90.0점** |
| keyword_accuracy | 50.0% | 100.0% | **100.0%** |
| thinking_usable_rate | 73.33% | 100.0% | **100.0%** |
| code_quality | 76.67% | 96.67% | **100.0%** |
| explanation_quality | 26.67% | 90.0% | **96.67%** |
| response_stability | 16.67% | 86.67% | **96.67%** |
| not_truncated_rate | 63.33% | 96.67% | **100.0%** |
| generation_time_compliance | 63.33% | 26.67% | 13.33% |

> `generation_time_compliance` 하락은 모델 품질 문제가 아닌 GPU 부하 문제이다.

### 단계별 핵심 지표 요약

| 단계 | format_compliance | response_stability | baseline_50_score |
|------|------------------|-------------------|-------------------|
| 베이스 | 16.67% | 16.67% | 69.63점 |
| LoRA v1 | 86.67% | 86.67% | 89.26점 |
| LoRA v2 (최종) | **96.67%** | **96.67%** | **90.0점** |

**베이스 대비 최종 개선폭: format_compliance +80%p, response_stability +80%p**

---

## 실험 설정 기록

### 1차 실험 (Qwen2.5-0.5B | 김정민)
 
| 항목 | 내용 |
|------|------|
| 베이스 모델 | `Qwen/Qwen2.5-0.5B-Instruct` |
| 학습 데이터 | `train_sft_merged.jsonl` (11,480개, 한국어 사고과정 형식) |
| 평가 스크립트 | `base_eval.py` 직접 작성 |
| 평가 항목 | 형식 준수율 / 코드 블록 포함률 / 코드 문법 통과율 |
| 형식 준수율 | 0~15% (프롬프트 개선 전/후) |
| 코드 블록 포함률 | 85~100% (정답 코드 바로 생성하는 문제 확인) |
| 코드 문법 통과율 | 80~100% |
| 결론 | 0.5B 경량 모델의 한계 확인, LoRA 튜닝 필요성 수립 |
 
### 2차 실험 — (Qwen2.5-3B | 김재홍)
 
| 항목 | 베이스 평가 | LoRA v1 | LoRA v2 |
|------|------------|---------|---------|
| 모델 | Qwen2.5-3B-Instruct | 동일 + LoRA adapter | 동일 + LoRA adapter v2 |
| 튜닝 방식 | — | 태스크별 분리 adapter | 태스크별 분리 adapter |
| Thinking 데이터 | — | 고정 템플릿 문장 | `<think>` 태그 기반 문제별 추출 |
| epochs | — | keyword/code: 1.0 / thinking: 2.0 | 동일 |
| lora_r | — | keyword/code: 8 / thinking: 16 | 동일 |
| lora_alpha | — | keyword/code: 16 / thinking: 32 | 동일 |
| 평가 샘플 수 | 30 | 30 | 30 |
| 결과 폴더 | base_eval_split/ | lora_eval_split/ | lora_eval_split_tuning_2/ |
| adapter | — | split_adapters.zip | split_adapters_tuning_2.zip |
 
---

## 주의사항

- `evaluate_lora_*.py`가 `evaluate_base_*.py`를 직접 import하므로 반드시 같은 폴더에 두어야 한다.
- 런팟 Pod를 끄기 전에 반드시 adapter 파일을 로컬에 다운로드해야 한다. Pod 종료 시 파일이 삭제된다.
- `evaluate_base_full.py`와 `evaluate_lora_full.py`는 keyword / thinking / code 평가가 모두 완료된 후 실행해야 한다.
- `evaluate_lora_*.py` 실행 시 `--adapter-path` 옵션으로 사용할 adapter 경로를 명시할 수 있다.
