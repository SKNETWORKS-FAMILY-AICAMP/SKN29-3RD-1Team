# 모델링 베이스 LoRA 평가 결과
> 베이스 모델 평가 → 1.5B LoRA → 3B LoRA 단계별 실험 결과 및 분석

#### 담당자
- 김재홍
- 김정민

---

## 주요 평가 지표

본 프로젝트에서는 모델 응답 품질을 한 가지 기준으로만 판단하지 않고, 서로 다른 관점의 평가 방식을 함께 사용하여 1~5점까지의 등급으로 OpenAI Chatgpt-5.5 모델이 판단하였다.

| 평가 방식 | 목적 |
|---|---|
| 형식 평가 | 모델이 요구한 출력 형식과 구조를 잘 지켰는지 확인 |
| Reference 기반 평가 | 모델 답변이 기준 정답 reasoning/code와 얼마나 내용적으로 일치하는지 확인 |
| 요구사항 기반 평가 | 모델 답변이 실제 문제 해결 요구사항을 만족하는지 확인 |

---
## 형식 평가

| 지표 | 설명 | 목표 |
|------|------|------|
| `keyword_accuracy` | RAG 키워드 6개 형식 및 vocab 일치율 | 90% 이상 |
| `thinking_quality` | Thinking Process 5개 항목 완성도 | 90% 이상 |
| `code_quality` | 코드 문법 통과 및 형식 준수율 | 90% 이상 |
| `format_compliance` | 전체 응답 형식 준수율 | 70% 이상 |
| `response_stability` | 형식 + 내용 + 길이 모두 통과율 | 70% 이상 |
| `baseline_50_score` | 9개 항목 종합 점수 (50점 이상 통과) | 80점 이상 |

---

## Reference 기반 평가

Reference 기반 평가는 모델 답변을 검증 데이터의 기준 답변과 비교하는 평가이다.

이 평가는 모델이 기준 답변과 얼마나 비슷한 풀이 방향을 유지했는지 확인한다.

| 항목 | 설명 |
|---|---|
| 문제 이해 일치도 | 모델이 원본 문제의 요구사항을 제대로 반영했는지 |
| Reasoning 일치도 | 모델의 사고 과정이 기준 `<think>`와 얼마나 유사한지 |
| 알고리즘 방향 일치도 | 기준 풀이와 같은 핵심 알고리즘을 사용했는지 |
| 코드 구조 일치도 | 기준 코드와 유사한 구현 흐름을 따르는지 |
| 핵심 개념 매칭 | 기준 답변에 포함된 주요 개념을 모델 답변도 포함하는지 |

Reference 기반 평가는 모델이 원본 정답 흐름을 얼마나 잘 유지하는지 확인하는 데 적합하다.

---

## 요구사항 기반 평가

요구사항 기반 평가는 기준 답변과 문장이 얼마나 비슷한지보다, 모델 답변이 실제 문제를 풀기에 충분한지를 평가한다.

```text
이 답변이 문제 요구사항을 만족하는가?
이 알고리즘이 제약조건 안에서 동작 가능한가?
이 코드는 문제 해결 흐름에 맞는가?
```

| 항목 | 설명 |
|---|---|
| 문제 요구사항 반영 | 입력, 출력, 목표 조건을 제대로 이해했는지 |
| 핵심 알고리즘 선택 | 문제에 맞는 알고리즘/자료구조를 선택했는지 |
| 제약조건 고려 | 시간복잡도, 입력 크기, 예외 케이스를 고려했는지 |
| 풀이 근거 설명 | 왜 해당 알고리즘을 선택했는지 설명했는지 |
| 코드 구조 적절성 | 코드가 문제 해결 흐름에 맞고 실행 가능한 구조인지 |
| 출력 형식 준수 | 요구한 섹션, 코드블록, 키워드 형식을 지켰는지 |

요구사항 기반 평가는 실제 문제 해결력과 답변 적절성을 확인하는 데 적합하다.

---

## 평가 점수 요약

평가는 동일한 검증 데이터 1000개 중 30개 샘플을 기준으로 수행하였다.

비교 대상은 다음과 같다.

| 모델 | 설명 |
|---|---|
| BASE | Qwen2.5-3B-Instruct 원본 모델 |
| TUNING 1차 | 1차 LoRA 튜닝 모델 |
| TUNING 2차 | 하이퍼파라미터를 조정한 2차 LoRA 튜닝 모델 |

해석:

- TUNING 1차는 BASE보다 출력 형식 안정성이 크게 개선되었다.
- TUNING 2차는 TUNING 1차보다 thinking, code, full 형식 지표가 추가로 개선되었다.
- 기존 자동 형식 평가 기준에서는 TUNING 2차가 가장 우수하다.
---
## Reference 기반 평가 점수

Reference 기반 평가는 모델 답변이 검증 데이터의 `<input>`, `<think>`, 기준 코드와 얼마나 내용적으로 일치하는지 평가하였다.

| Stage | BASE | TUNING 1차 | TUNING 2차 |
|---|---:|---:|---:|
| keyword | 4.17 | 4.00 | 4.03 |
| thinking | 4.07 | 3.63 | 4.20 |
| code | 4.73 | 4.60 | 4.83 |
| full | 4.57 | 4.23 | 4.63 |

해석:

- TUNING 1차는 형식은 개선되었지만 reference 내용 일치도는 BASE보다 낮아졌다.
- 특히 thinking에서 하락 폭이 컸다.
- TUNING 2차는 thinking, code, full에서 BASE를 소폭 상회하였다.
- keyword는 TUNING 2차에서도 BASE보다 낮아, 문제별 핵심 키워드 선택은 추가 개선이 필요하다.
---
## 요구사항 기반 평가 점수

요구사항 기반 평가는 기준 답변과의 문장 유사도보다 실제 문제 해결 요구사항을 만족하는지를 평가하였다.

| Stage | BASE | TUNING 1차 | TUNING 2차 |
|---|---:|---:|---:|
| keyword | 3.10 | 3.33 | 3.17 |
| thinking | 4.60 | 4.43 | 4.67 |
| code | 4.93 | 4.93 | 5.00 |
| full | 4.87 | 4.73 | 4.83 |

해석:

- 요구사항 기반 평가에서는 BASE가 full 기준에서 가장 높다.
- TUNING 2차는 TUNING 1차보다 thinking, code, full이 개선되었다.
- code는 TUNING 2차가 5.00으로 가장 안정적이다.
- full은 TUNING 2차가 1차보다 개선되었지만 BASE보다 소폭 낮다.
---
## 평가 점수 종합 해석

| 관점 | 가장 우수한 모델 | 판단 근거 |
|---|---|---|
| 형식 안정성 | TUNING 2차 | keyword, thinking, code, full 형식 지표가 가장 높음 |
| Reference 내용 일치도 | TUNING 2차 | thinking, code, full에서 BASE를 소폭 상회 |
| 실제 문제 요구사항 충족 | BASE | full 요구사항 기반 점수가 가장 높음 |
| 튜닝 개선 흐름 | TUNING 2차 | 1차 대비 내용 일치도와 코드 안정성이 회복됨 |

최종적으로 TUNING 2차는 형식 안정성과 reference 기반 내용 일치도에서는 가장 좋은 결과를 보였고, 요구사항 기반 full 평가는 BASE와 거의 비슷하지만 아직 소폭 낮은 수준으로 평가되었다.

---

## 1차 실험 — Qwen2.5-0.5B (김정민)

### 실험 설정

| 항목 | 내용 |
|------|------|
| 베이스 모델 | `Qwen/Qwen2.5-0.5B-Instruct` |
| 학습 데이터 | `train_sft_merged.jsonl` (11,480개, 한국어 사고과정 형식) |
| 평가 스크립트 | `base_eval.py` 직접 작성 |
| 평가 항목 | 형식 준수율 / 코드 블록 포함률 / 코드 문법 통과율 |
| 평가 샘플 수 | 20개 |

### 평가 결과

| 지표 | 1차 평가 (기본 프롬프트) | 2차 평가 (형식 명시 프롬프트) |
|------|------------------------|---------------------------|
| 형식 준수율 | 0.0% | 15.0% |
| 코드 블록 포함률 | 100% | 85% |
| 코드 문법 통과율 | 100% | 80% |

### 결론

0.5B 경량 모델은 `## 사고과정` 형식을 전혀 지키지 못하고 정답 코드를 바로 생성하는 경향을 보였다. 프롬프트 개선만으로는 형식 준수율이 최대 15% 수준에 그쳤으며, LoRA 튜닝의 필요성을 확인하였다.

---

## 2차 실험 — Qwen2.5-1.5B

### 실험 설정

| 항목 | 베이스 평가 | LoRA v1 |
|------|------------|---------|
| 모델 | `Qwen2.5-1.5B-Instruct` | 동일 + LoRA adapter |
| 학습 데이터 | — | `opencode_reasoning_train_4000.jsonl` |
| 평가 데이터 | `opencode_reasoning_test_1000.jsonl` | 동일 |
| 튜닝 방식 | — | 통합 응답 adapter |
| Thinking 데이터 | — | `<think>` 태그 기반 문제별 추출 |
| 최대 토큰 수 | — | keyword/thinking: 768 / code: 1024 |
| epochs | — | keyword/code: 1.0 / thinking: 3.0 |
| lora_r | — | keyword/code: 8 / thinking: 16 |
| lora_alpha | — | keyword/code: 16 / thinking: 32 |
| 평가 샘플 수 | 30 | 30 |

### 베이스 vs LoRA 비교

| 평가 항목 | 베이스 | LoRA | 변화 |
|---|---:|---:|---:|
| keyword accuracy | 50.00% | 70.00% | +20.00%p |
| exact 6 rate | 60.00% | 70.00% | +10.00%p |
| thinking quality | 43.33% | 6.67% | -36.66%p |
| thinking label accuracy | 96.67% | 56.67% | -40.00%p |
| code quality | 76.67% | 93.33% | +16.66%p |
| syntax pass rate | 76.67% | 93.33% | +16.66%p |
| full format compliance | 3.33% | 10.00% | +6.67%p |
| full usable response rate | 40.00% | 33.33% | -6.67%p |
| not truncated rate | 60.00% | 50.00% | -10.00%p |
| avg generated tokens | 468.57 | 608.93 | +140.36 |
| avg generation seconds | 14.53s | 41.80s | +27.27s |

### 1.5B LoRA stage별 결과

| Stage | 주요 지표 | 결과 | 해석 |
|---|---|---:|---|
| keyword | keyword accuracy | 70.00% | 최소 기준 도달했지만 30%는 형식, 개수, vocabulary 실패 |
| keyword | exact 6 rate | 70.00% | 정확히 6개 keyword 생성이 안정적이지 않음 |
| thinking | thinking quality | 6.67% | thinking 형식 학습 실패에 가까움 |
| thinking | label accuracy | 56.67% | label 일부는 맞지만 전체 형식 불안정 |
| code | code quality | 93.33% | 코드 단독 생성은 가장 안정적 |
| code | syntax pass rate | 93.33% | 문법 오류는 적음 |
| full | format compliance | 10.00% | 전체 답변 구성 매우 불안정 |
| full | usable response rate | 33.33% | 실제 사용 가능한 응답은 1/3 수준 |
| full | baseline 50 pass rate | 60.00% | 완화 기준에서는 절반 이상 통과 |

### 문제 발견 내용

#### 베이스 모델에서 확인된 문제

| 실패 항목 | 실패 건수 | 비율 |
|---|---:|---:|
| format_ok 실패 | 30/30 | 100.0% |
| explanation_ok 실패 | 30/30 | 100.0% |
| stable_ok 실패 | 30/30 | 100.0% |
| truncated 발생 | 12/30 | 40.0% |
| generation_time_ok 실패 | 0/30 | 0.0% |

베이스 모델은 응답 시간은 안정적이었지만, 지정된 출력 형식을 거의 지키지 못했다. `## RAG Keywords`, `## Thinking Process`, `## Final Answer` 구조를 일부 생성하더라도 label에 bold markdown을 섞거나 불필요한 bullet과 설명을 추가하는 문제가 반복되었다. 또한 허용 vocabulary 밖 keyword가 생성되어 RAG keyword 평가 기준을 만족하지 못했다.

#### LoRA 모델에서 확인된 문제

| 실패 항목 | 실패 건수 | 비율 |
|---|---:|---:|
| full format_ok 실패 | 27/30 | 90.0% |
| full usable_response_ok 실패 | 20/30 | 66.7% |
| full thinking_label_ok 실패 | 18/30 | 60.0% |
| full explanation_ok 실패 | 27/30 | 90.0% |
| full truncated 발생 | 15/30 | 50.0% |
| full generation_time_ok 실패 | 29/30 | 96.7% |

LoRA 튜닝 후 code stage는 93.33%의 code quality를 보여 개선 효과가 있었다. 그러나 full stage에서는 keyword, thinking, code를 한 번에 안정적으로 조립하지 못했다. keyword만 요구했는데 `Problem Understanding`, `Core Concept` 같은 설명 섹션을 붙이거나, thinking만 요구했는데 code block까지 생성하는 식으로 태스크 경계가 무너졌다.

### 결론

1. 베이스 모델은 빠르지만, 프로젝트가 요구하는 구조화된 학습자료 형식을 따르지 못한다.
2. LoRA 튜닝은 keyword와 code 같은 부분 태스크에는 효과가 있다.
3. 1.5B 모델은 full 응답처럼 긴 구조화 답변을 한 번에 생성할 때 형식 이탈, 과생성, truncation, latency 문제가 커진다.
4. 따라서 full 답변을 직접 생성하기보다, keyword / thinking / code를 분리 생성한 뒤 정해진 템플릿으로 조립하는 방식이 더 적합하다.
5. 이후 3B 모델 실험에서는 1.5B에서 확인된 thinking/full 형식 불안정 문제를 개선하는 방향으로 학습 데이터와 adapter 구성을 조정하였다.

---

## 3차 실험 — Qwen2.5-3B

### 실험 설정

| 항목 | 베이스 평가 | LoRA v1 | LoRA v2 |
|------|------------|---------|---------|
| 모델 | `Qwen2.5-3B-Instruct` | 동일 + LoRA adapter | 동일 + LoRA adapter v2 |
| 학습 데이터 | — | `opencode_reasoning_train_4000.jsonl` | 동일 |
| 평가 데이터 | `opencode_reasoning_test_1000.jsonl` | 동일 | 동일 |
| 튜닝 방식 | — | 태스크별 분리 adapter | 태스크별 분리 adapter |
| Thinking 데이터 | — | `<think>` 태그 기반 문제별 추출 | 동일 |
| 최대 토큰 수 | — | keyword/thinking: 768 / code: 1024 | keyword: 768 / thinking/code: 1024 |
| epochs | — | keyword/code: 1.0 / thinking: 3.0 | keyword/code: 1.0 / thinking: 2.0 |
| lora_r | — | keyword/code: 8 / thinking: 16 | 동일 |
| lora_alpha | — | keyword/code: 16 / thinking: 32 | 동일 |
| 평가 샘플 수 | 30 | 30 | 30 |
| 결과 폴더 | base_eval_split/ | lora_eval_split/ | lora_eval_split_tuning_2/ |
| adapter | — | split_adapters.zip | split_adapters_tuning_2.zip |

### 하이퍼파라미터 변경 이력 (v1 → v2)

| 항목 | v1 | v2 | 변경 이유 |
|---|---:|---:|---|
| learning rate | `2e-4` | `1e-4` | 형식 과적합을 줄이고 베이스 지식 손상 최소화 |
| grad accumulation | `8` | `16` | 안정적인 gradient update 생성 |
| thinking max_length | `768` | `1024` | 긴 문제/답변의 핵심 정보 손실 방지 |
| thinking epochs | `3.0` | `2.0` | thinking 형식 과적합 방지 |
| thinking LoRA r/alpha | `8 / 32` | `16 / 32` | reasoning 표현 capacity 증가 |
| code LoRA r/alpha | `8 / 16` | `16 / 32` | 코드 구조 학습 능력 강화 |
| keyword 설정 | 유지 | 유지 | 이미 형식 기준 100%, 추가 강화 불필요 |
| prompt | 유지 | 유지 | Base/LoRA 비교 조건 동일하게 유지 |


### 베이스 vs LoRA v1 vs LoRA v2 비교

#### keyword

| 지표 | 베이스 | LoRA v1 | LoRA v2 |
|------|--------|---------|---------|
| keyword_accuracy | 50.0% | 100.0% | **100.0%** |
| rag_exact_6_rate | 73.33% | 100.0% | **100.0%** |
| allowed_vocab_accuracy | 50.0% | 100.0% | **100.0%** |
| not_truncated_rate | 83.33% | 100.0% | **100.0%** |
| avg_generation_seconds | 1.29s | 1.79s | 1.75s |

#### thinking

| 지표 | 베이스 | LoRA v1 | LoRA v2 |
|------|--------|---------|---------|
| thinking_quality | 33.33% | 90.0% | **96.67%** |
| label_accuracy | 96.67% | 100.0% | **100.0%** |
| no_extra_section_rate | 33.33% | 90.0% | **96.67%** |
| not_truncated_rate | 96.67% | 100.0% | **100.0%** |
| avg_generation_seconds | 7.71s | 11.77s | 12.76s |

> v1 → v2에서 thinking_quality 90% → 96.67%로 개선. `<think>` 태그 기반 학습 데이터 개선 효과.

#### code

| 지표 | 베이스 | LoRA v1 | LoRA v2 |
|------|--------|---------|---------|
| code_quality | 76.67% | 96.67% | **100.0%** |
| syntax_pass_rate | 76.67% | 96.67% | **100.0%** |
| one_code_block_rate | 76.67% | 96.67% | **100.0%** |
| not_truncated_rate | 83.33% | 96.67% | **100.0%** |
| avg_generation_seconds | 9.22s | 16.03s | 14.95s |

#### full

| 지표 | 베이스 | LoRA v1 | LoRA v2 |
|------|--------|---------|---------|
| format_compliance | 16.67% | 86.67% | **96.67%** |
| usable_response_rate | 40.0% | 96.67% | **100.0%** |
| baseline_50_pass_rate | 70.0% | 96.67% | **100.0%** |
| avg_baseline_50_score | 69.63점 | 89.26점 | **90.0점** |
| thinking_usable_rate | 73.33% | 100.0% | **100.0%** |
| explanation_quality | 26.67% | 90.0% | **96.67%** |
| response_stability | 16.67% | 86.67% | **96.67%** |
| not_truncated_rate | 63.33% | 96.67% | **100.0%** |
| generation_time_compliance | 63.33% | 26.67% | 13.33% |

> `generation_time_compliance` 하락은 모델 품질 문제가 아닌 GPU(RTX 3080 10GB, 100% 사용) 부하 문제이다. 

### 핵심 지표 요약

| 단계 | format_compliance | response_stability | baseline_50_score |
|------|------------------|-------------------|-------------------|
| 베이스 | 16.67% | 16.67% | 69.63점 |
| LoRA v1 | 86.67% | 86.67% | 89.26점 |
| LoRA v2 (최종) | **96.67%** | **96.67%** | **90.0점** |

**베이스 대비 최종 개선폭: format_compliance +80%p, response_stability +80%p**

---

## 실패 및 개선 과정

| 단계 | 시도한 방법 | 발생한 문제 | 원인 분석 | 수정 방향 | 개선 결과 |
|---|---|---|---|---|---|
| 1차 Base 모델 평가 | Qwen 0.5B 생성 | 출력 형식 불일치, 답변 불일치, 코드 생성 실패, keyword vocabulary 이탈 등 목표 지표 달성 실패 | 도메인별 출력 형식, 프롬프트 구체적인 지시문 학습 부족 | 명확한 지시를 통한 프롬프트 재구성, 통합 추론이 아닌 과정별 추론과정 분리 | 유사 형식이 나왔으나 경량모델의 한계로 학습 실패 및 최종 베이스라인 설정 실패 |
| 2차 사용 모델 변경 및 LoRA 학습 | 1.5B 모델로 통합 추론 | thinking 응답이 짧아지고 일반화됨 | 0.5B, 1.5B 경량 모델 프롬프트 학습 한계 | 통합 방식에서 태스크별 개별응답 방식 제안, 로컬 모델 변경 제안 | 유사 형식이 나왔으나 경량모델의 한계로 학습 실패 및 최종 베이스라인 설정 실패 |
| 3차 사용 모델 변경 및 LoRA 학습 | Qwen2.5-3B-Instruct 상위 모델 변경, 태스크(keyword/thinking/code) 분할 추론 | 형식 지표는 맞춰졌지만 reasoning 내용 밀도가 줄어듦, code/full에서 코드블록 닫힘이나 구조 안정성 문제 발생, 1차 튜닝 결과 성능이 떨어짐 | 너무 강한 파라미터 설정 | 형식 과적합 + 내용 밀도 하락 문제 방지를 위해 파라미터 완화 | 3B 베이스모델 설정 완료, 2차 튜닝 후 최종 목표 달성 |
| 3차 요구사항 평가 | 요구사항 기반 validator 적용 | 기존 평가가 형식 중심으로 과대평가 | 실제 문제 풀이 적합성 반영 부족 | content/requirement 평가 분리 | 평가 기준 명확화 |

---

## 1.5B Base vs LoRA 과정 중심 평가

### 평가 진행 과정

| 단계 | 수행 내용 | 확인한 문제 | 결과 해석 |
|---|---|---|---|
| 1단계: Base 모델 기준선 평가 | Qwen2.5-1.5B Base 모델로 통합 응답을 생성하고 기존 지표로 평가 | format, explanation, stability가 모두 실패 | Base 모델은 속도는 안정적이지만 프로젝트가 요구한 구조화된 학습자료 형식을 따르지 못함 |
| 2단계: LoRA 튜닝 적용 | keyword, thinking, code 태스크를 분리하여 LoRA adapter 학습 | 태스크별 성능 차이가 크게 나타남 | code 단독 생성은 좋아졌지만 thinking과 full 응답은 불안정 |
| 3단계: 기존 지표로 LoRA 평가 | keyword/thinking/code/full stage별 평가 수행 | keyword와 code는 개선, thinking/full은 형식 이탈과 과생성 발생 | 1.5B LoRA는 부분 태스크에는 효과가 있으나 긴 구조화 응답 생성은 부족 |
| 4단계: Base vs LoRA full 비교 | Base 통합 응답과 LoRA full stage를 공통 지표로 비교 | LoRA full은 Base보다 느리고 truncation이 증가 | full 응답을 모델이 한 번에 생성하는 방식은 비효율적 |
| 5단계: 개선 방향 도출 | full 직접 생성 대신 태스크별 생성 후 템플릿 조립 방식 검토 | 긴 응답에서 형식 제어 실패 | LMS형 서비스에서는 generation과 formatting을 분리하는 구조가 적합 |

### 1.5B Base vs LoRA full 공통 지표 비교

| 평가 항목 | Base | LoRA full | 변화 | 해석 |
|---|---:|---:|---:|---|
| format compliance | 0.00% | 10.00% | +10.00%p | LoRA 후 소폭 개선됐지만 여전히 낮음 |
| explanation quality | 0.00% | 10.00% | +10.00%p | 설명 품질도 일부 개선됐지만 기준 충족 수준은 아님 |
| response stability | 0.00% | 6.67% | +6.67%p | 안정성은 거의 개선되지 않음 |
| response length compliance | 100.00% | 90.00% | -10.00%p | LoRA가 더 길게 생성하면서 길이 기준이 악화됨 |
| generation time compliance | 100.00% | 3.33% | -96.67%p | LoRA full 응답은 시간 기준을 거의 지키지 못함 |
| not truncated rate | 60.00% | 50.00% | -10.00%p | LoRA full에서 잘림 문제가 더 커짐 |
| avg generated tokens | 468.57 | 608.93 | +140.36 | LoRA가 더 긴 답변을 생성 |
| avg generation seconds | 14.53s | 41.80s | +27.27s | 평균 생성 시간이 크게 증가 |


---

## 1.5B → 3B 개선 과정

### 변경 배경

1.5B 모델은 경량 모델이라 실행 비용은 낮았지만, keyword / thinking / code / full이 명확히 분리된 구조화 응답을 안정적으로 생성하지 못했다. 특히 full 응답에서는 여러 섹션을 한 번에 구성해야 하므로 모델이 task boundary를 지키지 못하고 불필요한 섹션을 추가하거나 응답을 과도하게 길게 생성하는 문제가 반복되었다.

이에 따라 베이스 모델을 Qwen2.5-3B-Instruct로 변경하여, 더 큰 모델 용량이 형식 준수율, 사고과정 구성, 코드 생성 안정성에 어떤 영향을 주는지 확인하였다.

### 단계별 개선 흐름

| 단계 | 모델/방식 | 확인된 문제 | 다음 조치 | 결과 |
|---|---|---|---|---|
| 1단계 | 1.5B Base | format, explanation, stability 모두 0% 수준 | LoRA 튜닝 필요성 확인 | 베이스 모델 단독으로는 학습자료 형식 생성 불가 |
| 2단계 | 1.5B LoRA | code는 개선됐지만 thinking/full 형식 불안정 | 모델 용량 확장 검토 | 부분 태스크는 가능하지만 full 응답은 부족 |
| 3단계 | 3B Base | 1.5B 대비 instruction following 개선 기대 | 동일한 평가 지표로 재평가 | 구조화 응답 가능성 증가 |
| 4단계 | 3B LoRA v1 | keyword/code는 크게 개선, thinking 일부 불안정 | thinking 학습 데이터 개선 | 태스크별 adapter 유지 |
| 5단계 | 3B LoRA v2 | `<think>` 기반 thinking 데이터 보강 | 최종 adapter 재학습 | thinking/full 형식 안정성 개선 |

### 1.5B LoRA vs 3B LoRA v2 비교

| 평가 항목 | 1.5B LoRA | 3B LoRA v2 | 개선 폭 |
|---|---:|---:|---:|
| keyword accuracy | 70.00% | 100.00% | +30.00%p |
| exact 6 rate | 70.00% | 100.00% | +30.00%p |
| thinking quality | 6.67% | 96.67% | +90.00%p |
| thinking label accuracy | 56.67% | 100.00% | +43.33%p |
| code quality | 93.33% | 100.00% | +6.67%p |
| full format compliance | 10.00% | 96.67% | +86.67%p |
| full usable response rate | 33.33% | 100.00% | +66.67%p |
| full explanation quality | 10.00% | 96.67% | +86.67%p |
| full response stability | 6.67% | 96.67% | +90.00%p |
| full not truncated rate | 50.00% | 100.00% | +50.00%p |

### 개선된 부분

**Task boundary 개선** — 1.5B LoRA에서는 keyword만 요구해도 `Problem Understanding`, `Core Concept` 같은 설명 섹션을 추가하거나, thinking만 요구했는데 code block을 생성하는 문제가 있었다. 3B LoRA v2에서는 각 stage의 출력 형식이 안정화되었다.
**Thinking Process 품질 개선** — 1.5B LoRA의 thinking quality는 6.67%로 가장 큰 약점이었다. 3B 실험에서 `<think>` 태그 기반 reasoning을 활용해 문제별 사고과정 데이터를 보강한 결과 96.67%까지 개선되었다.
**Full 응답 안정성 개선** — 1.5B LoRA full stage는 format compliance 10.00%, usable response rate 33.33%로 실제 학습자료로 쓰기 어려웠다. 3B LoRA v2에서는 full format compliance 96.67%, usable response rate 100.00%를 기록하였다.
**Truncation 문제 개선** — 1.5B LoRA full 응답은 50%가 truncation으로 잘렸으나, 3B LoRA v2에서는 not truncated rate가 100%로 개선되었다.

### 남은 문제 및 대응 방향
| 남은 문제 | 대응 방향 |
|---|---|
| 3B 모델 추론 속도 부담 | CUDA 환경 구성, quantization, batch 최적화 검토 |
| full 응답 생성 비용 | keyword/thinking/code 분리 생성 후 템플릿 조립 |
| GPU 메모리 부족 가능성 | device_map, max_memory, offload 설정 적용 |
| 실시간 서비스 응답 지연 | 캐싱, 사전 생성 콘텐츠, 비동기 평가 구조 적용 |

### 최종 결론

1.5B 실험은 실패가 아니라 모델 용량과 구조화 응답 생성 능력의 한계를 확인한 기준선 역할을 했다.
LoRA 튜닝 시에서는 프롬프트를 변경하지 않고 하이퍼파라미터만 조정하여 비교 공정성을 유지하였다.
이 결과를 바탕으로 3B 모델로 확장하고, 태스크별 LoRA adapter와 `<think>` 기반 thinking 데이터 보강을 적용한 결과 keyword / thinking / code / full 대부분의 핵심 지표가 크게 개선되었으며, 최종적으로 코딩테스트 학습 플랫폼에서 사용할 수 있는 구조화 학습자료 생성 가능성을 확인하였다.
