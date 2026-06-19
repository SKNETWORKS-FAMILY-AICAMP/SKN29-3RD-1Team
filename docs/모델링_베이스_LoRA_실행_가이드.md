# 모델링 베이스 LoRA 실행 가이드
> Qwen2.5 베이스 모델 평가 → LoRA 튜닝 → LoRA 평가 전체 파이프라인 정리

#### 담당자
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

1. `output` 필드에서 ` ```python ... ``` ` 코드블록 추출
2. 문제 텍스트 + 코드 분석으로 ALLOWED_VOCAB 기반 키워드 6개 자동 선택
3. `<think>` 태그 내용을 추출하여 5개 항목 Thinking Process로 변환 (v2 개선 사항)
4. keyword / thinking / code 태스크별 학습 샘플로 분리 → 최대 **12,000개** 학습 샘플 생성


--- 

## LoRA 하이퍼파라미터 튜닝 기준

LoRA 튜닝은 BASE 모델과의 비교 공정성을 유지하기 위해 프롬프트는 변경하지 않고, 학습 하이퍼파라미터만 조정하였다.

하이퍼파라미터 변경의 목적은 다음과 같다.

- 형식 과적합을 줄인다.
- BASE 모델이 가진 기존 문제풀이 능력 손상을 줄인다.
- thinking 응답에서 세부 조건과 풀이 근거가 과도하게 줄어드는 문제를 완화한다.
- code 응답의 코드블록 안정성과 구현 구조를 개선한다.

| 항목 | 1차 설정 | 2차 설정 | 변경 목적 |
|--------|--------|--------|--------|
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

## 하이퍼파라미터 변경 해석

- learning rate를 낮춘 이유
  - LoRA가 너무 강하게 학습되면 BASE 모델의 기존 문제풀이 능력이 손상될 수 있다.
  - 특히, 형식은 좋아지지만 내용 밀도가 낮아지는 문제가 발생할 수 있다.
  - 따라서 `2e-4`에서 `1e-4`로 낮춰 안정적인 학습을 유도하였다.

- gradient accumulation을 늘린 이유
  - GPU 메모리 제한으로 실제 batch size를 크게 늘리기 어렵다.
  - gradient accumulation을 늘리면 작은 batch 환경에서도 더 안정적인 업데이트 효과를 얻을 수 있다.

- thinking max length를 늘린 이유
  - thinking 응답은 문제 조건, 알고리즘 선택 이유, 제약조건, 풀이 전략을 포함해야 한다.
  - max length가 짧으면 답변이 지나치게 압축되어 reference reasoning과의 내용 일치도가 낮아질 수 있다.

- thinking epochs를 줄인 이유
  - thinking을 오래 학습하면 출력 형식에는 더 잘 맞지만 내용이 정형화될 수 있다.
  - reasoning 다양성과 세부 조건 보존을 위해 epoch를 줄였다.

- LoRA rank를 늘린 이유
  - rank는 adapter가 학습할 수 있는 표현 용량과 관련된다.
  - thinking과 code는 keyword보다 복잡한 출력을 요구하므로 rank를 늘려 표현력을 보강하였다.

- keyword 설정을 유지한 이유
  - keyword는 이미 형식 안정성이 높은 stage였다.
  - 따라서 불필요한 변경으로 성능이 흔들리는 것을 피하고 기존 설정을 유지하였다.

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
> split_adapters 용량 제한으로 zip 파일로 대체하였으며, 실제 실행 시에는 압축을 풀어서 사용한다.

---

## 각 파일 역할

### 공통 기반 파일

| 파일 | 역할 |
|------|------|
| `eval_common.py` | 모든 평가 스크립트가 공유하는 공통 로직. 데이터 로딩, 모델 실행(BaseModelRunner / LoraModelRunner), 결과 저장, ALLOWED_VOCAB 정의 포함 |
| `qwen_base_model.py` | Qwen 베이스 모델과 토크나이저 로딩 함수 모음. GPU/CPU 장치 선택 포함 |
| `lora_tunning.ipynb`| Qwen/Qwen3-4B-Thinking-2507 모델 Distillation 성능 비교 |

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
| `evaluate_base_full.py` | keyword / thinking / code 결과를 조합하여 전체 응답 품질 평가. 위 세 파일의 결과 파일 생성 후 실행 |
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

> `--limit` 숫자를 높일수록 더 정확한 평가가 가능하지만, 환경에 따라 속도가 다르므로 30 권장.

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

## 주의사항

- `evaluate_lora_*.py`가 `evaluate_base_*.py`를 직접 import하므로 반드시 같은 폴더에 두어야 한다.
- 런팟 Pod를 끄기 전에 반드시 adapter 파일을 로컬에 다운로드해야 한다. Pod 종료 시 파일이 삭제된다.
- `evaluate_base_full.py`와 `evaluate_lora_full.py`는 keyword / thinking / code 평가가 모두 완료된 후 실행해야 한다.
- `evaluate_lora_*.py` 실행 시 `--adapter-path` 옵션으로 사용할 adapter 경로를 명시할 수 있다.
- GPU 환경으로 검증 및 반영하려면 기존 CPU 환경의 torch를 삭제하고 재설치한다.

```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
