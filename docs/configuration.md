# 설정 레퍼런스 (project.yaml)

> `project.yaml`의 모든 설정 옵션을 확인하세요.

## 1. 개요

`project.yaml`은 slm-factory 파이프라인의 모든 동작을 제어하는 중앙 설정 파일입니다. 코드 수정 없이 문서 파싱, QA 생성, 검증, 학습, 내보내기 등 전체 워크플로우를 구성할 수 있습니다.

**파이프라인 제어**: 문서 파싱부터 모델 내보내기까지 전 단계를 YAML 파일 하나로 제어합니다.

**타입 안전성**: Pydantic v2 기반 검증으로 잘못된 설정을 실행 전에 감지합니다.

**기본값 제공**: 모든 필드에 합리적인 기본값이 설정되어 있어 필요한 부분만 수정하면 됩니다.

**null 섹션 처리**: YAML에서 `null`로 지정된 섹션은 자동으로 기본값이 적용됩니다. 예를 들어 `eval: null`로 설정하면 `eval` 섹션 전체가 기본값으로 초기화됩니다.

### 설정 파일 생성

새 프로젝트를 시작할 때는 다음 명령으로 기본 템플릿을 생성합니다:

```bash
slm-factory init my-project
```

이 명령은 `my-project/project.yaml` 파일을 생성하며, 모든 기본값이 주석과 함께 포함되어 있습니다.

### 파일 위치

`project.yaml` 파일은 프로젝트 루트 디렉토리에 위치해야 합니다:

```
my-project/
├── project.yaml          # 설정 파일
├── documents/            # 입력 문서 디렉토리
└── output/               # 출력 디렉토리 (자동 생성)
```

설정 로딩 프로세스 상세는 [아키텍처 가이드](architecture.md)를 참조하십시오.

---

## 2. project — 프로젝트 메타데이터

> 프로젝트의 기본 정보를 정의합니다. 내보낸 모델의 이름과 버전 관리에 사용됩니다.

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `name` | `str` | `"my-project"` | 프로젝트 식별자. 내보낸 모델 이름에 반영됩니다 (빈 값 불가) |
| `version` | `str` | `"1.0.0"` | 시맨틱 버전. 모델 버전 관리에 사용됩니다 |
| `language` | `str` | `"en"` | 문서 언어 코드. `"en"` (영어), `"ko"` (한국어), `"ja"` (일본어) 등 |

```yaml
project:
  name: "company-policy-assistant"
  version: "2.1.0"
  language: "ko"
```

---

## 3. paths — 경로 설정

> 입력 문서와 출력 파일의 위치를 지정합니다. 지정된 디렉토리가 없으면 자동으로 생성됩니다.

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `documents` | `Path` | `"./documents"` | 입력 문서가 저장된 디렉토리. 파싱할 PDF, HWPX 등의 파일을 여기에 배치합니다 |
| `output` | `Path` | `"./output"` | 모든 출력 파일이 저장되는 디렉토리. QA 데이터셋, 학습된 모델, 체크포인트 등이 생성됩니다 |

```yaml
paths:
  documents: "./data/source_docs"
  output: "./results"
```

상대 경로는 `project.yaml` 파일이 위치한 디렉토리를 기준으로 해석됩니다. 프로젝트 디렉토리 외부에서 `--config my-project/project.yaml`로 실행해도 올바른 경로를 참조합니다.

---

## 4. parsing — 문서 파싱 설정

> 입력 문서를 텍스트로 변환하는 파싱 단계의 설정입니다.

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `formats` | `list[str]` | `["pdf", "txt", "html"]` | 파싱할 문서 형식 목록 |
| `pdf` | `PdfOptions` | (하위 참조) | PDF 파싱 옵션 |
| `hwpx` | `HwpxOptions` | (하위 참조) | HWPX 파싱 옵션 |

### 지원 형식

| 형식 | 확장자 | 필요 패키지 |
|------|--------|-------------|
| `pdf` | `.pdf` | `pymupdf` (기본 포함) |
| `hwpx` | `.hwpx` | `beautifulsoup4`, `lxml` (기본 포함), `pykospacing` (선택) |
| `html` | `.html`, `.htm` | `beautifulsoup4` (기본 포함) |
| `txt` | `.txt` | 없음 |
| `md` | `.md` | 없음 |
| `docx` | `.docx` | `python-docx` (`pip install slm-factory[docx]`) |

### pdf 옵션

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `extract_tables` | `bool` | `true` | PDF 내 표를 마크다운 형식으로 추출합니다. `false`로 설정하면 표를 일반 텍스트로 처리합니다 |

### hwpx 옵션

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `apply_spacing` | `bool` | `true` | 한국어 띄어쓰기 교정을 적용합니다. `pykospacing` 패키지가 필요합니다 (`pip install slm-factory[korean]`) |

```yaml
parsing:
  formats: ["pdf", "hwpx", "html", "txt", "docx"]
  pdf:
    extract_tables: true
  hwpx:
    apply_spacing: true
```

---

## 5. teacher — Teacher LLM 설정

> QA 쌍을 생성하는 Teacher 모델의 설정입니다. Ollama 또는 OpenAI 호환 API를 사용할 수 있습니다.

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `backend` | `"ollama"` \| `"openai"` | `"ollama"` | 사용할 LLM 백엔드. `"ollama"`는 로컬 Ollama 서버, `"openai"`는 OpenAI 호환 API입니다 |
| `model` | `str` | `"qwen3:8b"` | 모델 이름 또는 ID (빈 값 불가) |
| `api_base` | `str` | `"http://localhost:11434"` | API 엔드포인트 URL (빈 값 불가) |
| `api_key` | `str \| None` | `null` | API 키. `backend: "openai"`일 때 필수입니다. Ollama는 불필요합니다 |
| `temperature` | `float` | `0.3` | 생성 온도 (0.0~1.0). 낮을수록 일관성 있는 답변을 생성합니다 |
| `timeout` | `int` | `180` | API 요청 타임아웃 (초). 긴 문서 처리 시 늘려야 할 수 있습니다 |
| `max_context_chars` | `int` | `12000` | 문서 내용 잘라내기 한계 (문자 수). 이 길이를 초과하는 문서는 앞부분만 사용됩니다 |
| `max_concurrency` | `int` | `4` | 비동기 QA 생성 시 최대 동시 요청 수. 높이면 빠르지만 서버 부하가 증가합니다 |

```yaml
teacher:
  backend: "ollama"
  model: "qwen3:8b"
  api_base: "http://localhost:11434"
  temperature: 0.3
  timeout: 180
  max_context_chars: 12000
  max_concurrency: 4
```

OpenAI 호환 백엔드 (vLLM, LiteLLM, OpenRouter 등) 사용 시:

```yaml
teacher:
  backend: "openai"
  model: "gpt-4o-mini"
  api_base: "https://api.openai.com"
  api_key: "sk-proj-..."
  temperature: 0.3
```

권장 Ollama 모델: `qwen3:8b` (한국어/다국어 우수), `llama3.1:8b` (영어 우수), `gemma2:9b` (고품질 QA 생성).

---

## 6. questions — 질문 설정

> Teacher 모델이 문서에 대해 생성할 질문을 정의합니다. 카테고리별 질문 목록 또는 외부 파일로 지정할 수 있습니다.

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `categories` | `dict[str, list[str]]` | `{}` | 카테고리별 질문 목록. 키는 카테고리 이름, 값은 질문 문자열 리스트입니다 |
| `file` | `Path \| None` | `null` | 외부 질문 파일 경로. 한 줄에 하나씩 질문을 작성합니다. 이 필드가 설정되면 `categories`는 무시됩니다 |
| `system_prompt` | `str` | (하위 참조) | Teacher 모델에게 전달되는 시스템 프롬프트. 답변 스타일과 제약 조건을 정의합니다 |
| `output_format` | `str` | `"alpaca"` | QA 데이터셋 출력 형식. 현재는 `"alpaca"` 형식만 지원합니다 |

기본 시스템 프롬프트:

```
You are a helpful assistant that answers questions based strictly on the provided document.
Answer only from the document content. Do not speculate or fabricate information.
Be concise and factual. Include specific numbers, dates, and names when available.
If the document does not contain relevant information, say "The document does not contain this information."
```

한국어 프로젝트 예시:

```yaml
questions:
  categories:
    개요:
      - "이 문서의 주요 목적은 무엇입니까?"
      - "이 정책이 해결하려는 문제는 무엇입니까?"
      - "대상 사용자는 누구입니까?"
      - "주요 변경 사항은 무엇입니까?"
      - "적용 범위와 제한 사항은 무엇입니까?"
    기술:
      - "주요 기술 사양은 무엇입니까?"
      - "사용된 기술이나 도구는 무엇입니까?"
      - "시스템 요구 사항은 무엇입니까?"
      - "기존 방식과 어떻게 다릅니까?"
      - "성능 지표는 무엇입니까?"
    구현:
      - "구현 일정은 어떻게 됩니까?"
      - "필요한 자원은 무엇입니까?"
      - "구현 절차는 무엇입니까?"
      - "예상되는 위험은 무엇입니까?"
      - "기대 효과는 무엇입니까?"
  system_prompt: >
    당신은 제공된 문서를 기반으로 질문에 답변하는 도움이 되는 어시스턴트입니다.
    문서 내용에만 근거하여 답변하십시오. 추측하거나 정보를 만들어내지 마십시오.
    간결하고 사실적으로 답변하십시오. 가능한 경우 구체적인 숫자, 날짜, 이름을 포함하십시오.
    문서에 관련 정보가 없으면 "문서에 해당 정보가 포함되어 있지 않습니다"라고 답변하십시오.
  output_format: "alpaca"
```

외부 파일로 질문을 관리하려면 `questions.txt`에 한 줄씩 작성하고 `file: "./questions.txt"`로 참조합니다. `file`이 설정되면 `categories`는 무시됩니다.

---

## 7. validation — QA 검증 설정

> 생성된 QA 쌍의 품질을 검증하고 필터링하는 규칙을 정의합니다.

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `enabled` | `bool` | `true` | 검증 기능 활성화 여부. `false`로 설정하면 모든 검증을 건너뜁니다 |
| `min_answer_length` | `int` | `20` | 답변 최소 길이 (문자 수). 이보다 짧은 답변은 거부됩니다 |
| `max_answer_length` | `int` | `2000` | 답변 최대 길이 (문자 수). 이보다 긴 답변은 거부됩니다 |
| `remove_empty` | `bool` | `true` | 빈 질문이나 답변을 제거합니다 |
| `deduplicate` | `bool` | `true` | 중복된 질문-답변 쌍을 제거합니다 |
| `reject_patterns` | `list[str]` | (하위 참조) | 답변에서 거부할 정규식 패턴 목록. 패턴이 매칭되면 해당 QA 쌍을 거부합니다 |
| `groundedness` | `GroundednessConfig` | (하위 참조) | 의미적 근거성 검증 설정 |

기본 거부 패턴 3개: `"(?i)i don't know"`, `"(?i)not (available|provided|mentioned|found)"`, `"(?i)the document does not contain"`. `(?i)` 플래그는 대소문자를 구분하지 않습니다.

**제약조건**: `min_answer_length`는 반드시 `max_answer_length`보다 작아야 합니다. 그렇지 않으면 설정 로드 시 오류가 발생합니다.

### groundedness — 의미적 검증

> 답변이 원본 문서 내용에 근거하고 있는지 임베딩 기반으로 검증합니다.

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `enabled` | `bool` | `false` | 의미적 검증 활성화 여부. `pip install slm-factory[validation]`이 필요합니다 |
| `model` | `str` | `"all-MiniLM-L6-v2"` | 사용할 sentence-transformers 모델 |
| `threshold` | `float` | `0.3` | 코사인 유사도 임계값 (0.0~1.0). 높을수록 엄격하게 검증합니다 |

```yaml
validation:
  enabled: true
  min_answer_length: 20
  max_answer_length: 2000
  remove_empty: true
  deduplicate: true
  reject_patterns:
    - "(?i)i don't know"
    - "(?i)not (available|provided|mentioned|found)"
    - "(?i)the document does not contain"
    - "(?i)알 수 없"
    - "(?i)정보가 없"
    - "(?i)언급되지 않"
    - "(?i)포함되어 있지 않"
    - "(?i)제공되지 않"
  groundedness:
    enabled: true
    model: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    threshold: 0.35
```

한국어 문서에는 다국어 임베딩 모델 `paraphrase-multilingual-MiniLM-L12-v2`를 권장합니다.

---

## 8. scoring — 품질 점수 평가 설정

> 생성된 QA 쌍의 품질을 Teacher LLM으로 평가하고 낮은 점수의 QA 쌍을 필터링합니다.

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `enabled` | `bool` | `false` | 품질 점수 평가 기능 활성화 여부 |
| `threshold` | `float` | `3.0` | QA 쌍 최소 합격 점수 (1.0~5.0). 이 점수 미만의 QA 쌍은 데이터셋에서 제거됩니다 |
| `max_concurrency` | `int` | `4` | 비동기 점수 평가 시 최대 동시 요청 수 |

```yaml
scoring:
  enabled: true
  threshold: 3.5
  max_concurrency: 4
```

Teacher LLM이 질문의 명확성, 답변의 정확성, 문서와의 관련성을 종합하여 1~5점으로 평가합니다. `threshold` 미만의 QA 쌍은 데이터셋에서 제거됩니다. 품질은 높아지지만 데이터셋 크기는 줄어들 수 있습니다.

---

## 9. augment — 데이터 증강 설정

> 기존 QA 쌍의 질문을 패러프레이즈하여 데이터셋을 확장합니다.

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `enabled` | `bool` | `false` | 데이터 증강 기능 활성화 여부 |
| `num_variants` | `int` | `2` | 각 QA 쌍당 생성할 패러프레이즈 변형 수. 원본 포함 총 `num_variants + 1`개의 QA 쌍이 생성됩니다 |
| `max_concurrency` | `int` | `4` | 비동기 증강 시 최대 동시 요청 수 |

```yaml
augment:
  enabled: true
  num_variants: 3
  max_concurrency: 4
```

`num_variants: 3`으로 설정하면 100개의 QA 쌍이 400개로 확장됩니다. 데이터가 부족할 때 유용하지만, Teacher LLM 호출 비용이 증가합니다.

---

## 10. analyzer — 데이터 분석 설정

> 생성된 QA 데이터셋의 통계 정보를 분석하고 보고서를 생성합니다.

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `enabled` | `bool` | `true` | 데이터 분석 기능 활성화 여부 |
| `output_file` | `str` | `"data_analysis.json"` | 분석 보고서 JSON 파일명. `paths.output` 디렉토리에 저장됩니다 |

```yaml
analyzer:
  enabled: true
  output_file: "qa_analysis_report.json"
```

분석 보고서에는 전체 QA 쌍 수, 카테고리 분포, 문서별 분포, 질문/답변 길이 통계, 데이터 품질 경고가 포함됩니다. LLM 호출 없이 순수 통계 분석만 수행합니다.

---

## 11. student — Student 모델 설정

> 파인튜닝할 Student 모델을 지정합니다. HuggingFace Hub의 모든 causal language model을 사용할 수 있습니다.

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `model` | `str` | `"google/gemma-3-1b-it"` | HuggingFace 모델 ID. causal LM이어야 합니다 (예: GPT, Llama, Gemma) (빈 값 불가) |
| `max_seq_length` | `int` | `4096` | 학습 데이터의 최대 토큰 길이. 이보다 긴 시퀀스는 잘립니다 |

```yaml
student:
  model: "Qwen/Qwen3-1.7B"
  max_seq_length: 4096
```

권장 모델:

| 모델 | HuggingFace ID | VRAM | 특징 |
|------|----------------|------|------|
| Gemma 3 1B IT | `google/gemma-3-1b-it` | ~4GB | 다국어 지원, 빠른 학습 |
| Gemma 3 4B IT | `google/gemma-3-4b-it` | ~10GB | 고품질 출력, 균형 잡힌 성능 |
| Llama 3.2 1B | `meta-llama/Llama-3.2-1B-Instruct` | ~4GB | 영어 강점, 경량 |
| Llama 3.2 3B | `meta-llama/Llama-3.2-3B-Instruct` | ~8GB | 영어 강점, 중간 크기 |
| Phi-4 Mini | `microsoft/Phi-4-mini-instruct` | ~10GB | 코드 생성, 추론 능력 우수 |
| Qwen3 1.7B | `Qwen/Qwen3-1.7B` | ~5GB | 한국어 지원 양호, 다국어 |

모델 선택 기준: 8GB VRAM 이하는 1B 모델 + 양자화, 한국어 문서는 `Qwen/Qwen3-1.7B` 또는 `google/gemma-3-1b-it` 권장.

---

## 12. training — 학습 설정

> LoRA 파인튜닝의 하이퍼파라미터와 학습 전략을 정의합니다.

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `batch_size` | `int` | `4` | 디바이스당 배치 크기. GPU 메모리에 따라 조정합니다 |
| `gradient_accumulation_steps` | `int` | `4` | 그래디언트 누적 스텝. 실제 배치 크기는 `batch_size × gradient_accumulation_steps` (기본값: 4×4=16) |
| `learning_rate` | `float` | `2e-5` | 학습률. LoRA 파인튜닝에는 `1e-5 ~ 5e-5` 범위를 권장합니다 |
| `lr_scheduler` | `str` | `"cosine"` | 학습률 스케줄러. `"cosine"`, `"linear"`, `"constant"` 등 |
| `warmup_ratio` | `float` | `0.1` | 워밍업 비율. 전체 학습 스텝의 10%를 워밍업에 사용합니다 (0.0~1.0) |
| `num_epochs` | `int` | `20` | 최대 에포크 수. 조기 종료가 활성화되면 이보다 일찍 멈출 수 있습니다 |
| `optimizer` | `str` | `"adamw_torch_fused"` | 옵티마이저. `"adamw_torch_fused"`는 PyTorch의 fused AdamW로 가장 빠릅니다 |
| `bf16` | `bool` | `true` | bfloat16 혼합 정밀도 학습. Ampere 이상 GPU (RTX 30xx, A100 등)에서만 지원됩니다 |
| `train_split` | `float` | `0.9` | 학습 데이터 비율. 0.9는 90% 학습, 10% 검증을 의미합니다 (0.0~1.0) |
| `save_strategy` | `str` | `"epoch"` | 체크포인트 저장 전략. `"epoch"`는 에포크마다, `"steps"`는 일정 스텝마다 저장합니다 |
| `lora` | `LoraConfig` | (하위 참조) | LoRA 어댑터 설정 |
| `early_stopping` | `EarlyStoppingConfig` | (하위 참조) | 조기 종료 설정 |
| `quantization` | `QuantizationConfig` | (하위 참조) | 양자화 설정 |

**제약조건**: `learning_rate`는 반드시 0보다 커야 합니다. 0 이하로 설정하면 설정 로드 시 오류가 발생합니다.

### lora — LoRA 어댑터 설정

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `r` | `int` | `16` | LoRA rank. 높을수록 표현력이 증가하지만 메모리 사용량도 증가합니다 (일반적으로 8~64) |
| `alpha` | `int` | `32` | LoRA scaling factor. 일반적으로 `r`의 2배 값을 사용합니다 |
| `dropout` | `float` | `0.05` | LoRA 레이어의 드롭아웃 비율. 과적합 방지를 위한 정규화 기법입니다 (0.0~1.0) |
| `target_modules` | `str \| list[str]` | `"auto"` | LoRA를 적용할 모듈 이름. `"auto"`는 자동 감지, 또는 `["q_proj", "v_proj"]` 같은 리스트로 명시할 수 있습니다 |
| `use_rslora` | `bool` | `false` | Rank-Stabilized LoRA 사용 여부. 학습 안정성을 높이지만 약간 느려집니다 |

### early_stopping — 조기 종료

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `enabled` | `bool` | `true` | 조기 종료 활성화 여부 |
| `patience` | `int` | `3` | 개선이 없어도 기다릴 에포크 수. 이 횟수 동안 개선이 없으면 학습을 중단합니다 |
| `threshold` | `float` | `0.01` | 개선으로 인정할 최소 변화량. 검증 손실이 이 값 이상 낮아져야 개선으로 간주합니다 |

### quantization — 양자화

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `enabled` | `bool` | `false` | 4비트 양자화 활성화 여부. VRAM 사용량을 약 50~60% 줄입니다 |
| `bits` | `int` | `4` | 양자화 비트 수. 현재는 4비트만 지원합니다 |

```yaml
training:
  lora:
    r: 16
    alpha: 32
    dropout: 0.05
    target_modules: "auto"
    use_rslora: false
  batch_size: 4
  gradient_accumulation_steps: 4
  learning_rate: 2.0e-5
  lr_scheduler: "cosine"
  warmup_ratio: 0.1
  num_epochs: 20
  early_stopping:
    enabled: true
    patience: 3
    threshold: 0.01
  optimizer: "adamw_torch_fused"
  bf16: true
  train_split: 0.9
  save_strategy: "epoch"
  quantization:
    enabled: false
    bits: 4
```

---

## 13. export — 모델 내보내기 설정

> 학습된 모델을 내보내는 방식을 정의합니다.

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `merge_lora` | `bool` | `true` | LoRA 어댑터를 베이스 모델에 병합할지 여부. `true`면 독립 실행 가능한 모델을 생성합니다 |
| `output_format` | `str` | `"safetensors"` | 모델 저장 형식. `"safetensors"` (권장) 또는 `"pytorch"` |
| `ollama` | `OllamaExportConfig` | (하위 참조) | Ollama 내보내기 설정 |

### ollama — Ollama 내보내기

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `enabled` | `bool` | `true` | Ollama 내보내기 활성화 여부. Modelfile을 생성하고 `ollama create` 명령을 실행합니다 |
| `model_name` | `str` | `"my-project-model"` | Ollama에 등록할 모델 이름 (빈 값 불가) |
| `system_prompt` | `str` | `"You are a helpful domain-specific assistant."` | Ollama 모델의 시스템 프롬프트. 모델의 역할과 동작을 정의합니다 |
| `parameters` | `dict[str, Any]` | (하위 참조) | Ollama 런타임 파라미터 |

`parameters` 기본값: `temperature: 0.7`, `top_p: 0.9`, `num_ctx: 4096`.

```yaml
export:
  merge_lora: true
  output_format: "safetensors"
  ollama:
    enabled: true
    model_name: "company-policy-assistant"
    system_prompt: "당신은 회사 정책에 대해 답변하는 전문 어시스턴트입니다. 정확하고 간결하게 답변하십시오."
    parameters:
      temperature: 0.5
      top_p: 0.9
      num_ctx: 4096
```

---

## 14. eval — 모델 평가 설정

> 학습된 모델을 BLEU/ROUGE 메트릭으로 자동 평가합니다.

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `enabled` | `bool` | `false` | 모델 평가 기능 활성화 여부 |
| `test_split` | `float` | `0.1` | 평가에 사용할 데이터 비율 (0.0~1.0) |
| `metrics` | `list[str]` | `["bleu", "rouge"]` | 평가 메트릭 목록 |
| `max_samples` | `int` | `50` | 평가에 사용할 최대 샘플 수 |
| `output_file` | `str` | `"eval_results.json"` | 평가 결과 JSON 파일명. `paths.output` 디렉토리에 저장됩니다 |

**제약조건**: `max_samples`는 1 이상이어야 합니다. 0 이하로 설정하면 설정 로드 시 오류가 발생합니다.

```yaml
eval:
  enabled: true
  test_split: 0.1
  metrics: ["bleu", "rouge"]
  max_samples: 50
  output_file: "eval_results.json"
```

---

## 15. compare — 모델 비교 설정

> Base 모델과 Fine-tuned 모델의 답변을 나란히 비교합니다.

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `enabled` | `bool` | `false` | 모델 비교 기능 활성화 여부 |
| `base_model` | `str` | `""` | 비교 기준 모델 이름 (Ollama) |
| `finetuned_model` | `str` | `""` | 파인튜닝된 모델 이름 (Ollama) |
| `metrics` | `list[str]` | `["bleu", "rouge"]` | 비교 메트릭 목록 |
| `max_samples` | `int` | `20` | 비교에 사용할 최대 샘플 수 |
| `output_file` | `str` | `"compare_results.json"` | 비교 결과 JSON 파일명. `paths.output` 디렉토리에 저장됩니다 |

**제약조건**: `max_samples`는 1 이상이어야 합니다. 0 이하로 설정하면 설정 로드 시 오류가 발생합니다.

```yaml
compare:
  enabled: true
  base_model: "gemma:2b"
  finetuned_model: "my-project-model"
  metrics: ["bleu", "rouge"]
  max_samples: 20
  output_file: "compare_results.json"
```

---

## 16. gguf_export — GGUF 변환 설정

> 병합된 모델을 llama.cpp 호환 GGUF 양자화 형식으로 변환합니다.

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `enabled` | `bool` | `false` | GGUF 변환 기능 활성화 여부 |
| `quantization_type` | `str` | `"q4_k_m"` | 양자화 타입. `q4_0`, `q4_1`, `q4_k_m`, `q4_k_s`, `q5_0`, `q5_1`, `q5_k_m`, `q5_k_s`, `q8_0`, `f16`, `f32` 중 선택 |
| `llama_cpp_path` | `str` | `""` | llama.cpp 경로. 빈 문자열이면 시스템 PATH에서 탐색합니다 |

```yaml
gguf_export:
  enabled: true
  quantization_type: "q4_k_m"
  llama_cpp_path: "/path/to/llama.cpp"
```

---

## 17. incremental — 증분 학습 설정

> 문서 추가 시 기존 QA를 유지하면서 새 문서만 처리합니다. 해시 기반으로 변경을 감지합니다.

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `enabled` | `bool` | `false` | 증분 학습 기능 활성화 여부 |
| `hash_file` | `str` | `"document_hashes.json"` | 문서 해시를 저장할 파일명. `paths.output` 디렉토리에 저장됩니다 |
| `merge_strategy` | `"append"` \| `"replace"` | `"append"` | 기존 QA와 새 QA의 병합 전략. `"append"`는 추가, `"replace"`는 교체 |
| `resume_adapter` | `str` | `""` | 이전 학습의 어댑터 경로. 빈 문자열이면 새로 학습합니다 |

```yaml
incremental:
  enabled: true
  hash_file: "document_hashes.json"
  merge_strategy: "append"
  resume_adapter: "./output/checkpoints/adapter"
```

---

## 18. dialogue — 멀티턴 대화 설정

> QA 쌍을 멀티턴 대화 형식으로 확장합니다.

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `enabled` | `bool` | `false` | 멀티턴 대화 생성 기능 활성화 여부 |
| `min_turns` | `int` | `2` | 최소 대화 턴 수 (2 이상) |
| `max_turns` | `int` | `5` | 최대 대화 턴 수 |
| `include_single_qa` | `bool` | `true` | 단일 QA 쌍도 대화 데이터에 포함할지 여부 |

**제약조건**: `min_turns`는 반드시 `max_turns`보다 작거나 같아야 합니다. 또한 `min_turns`는 2 이상이어야 합니다. 이를 위반하면 설정 로드 시 오류가 발생합니다.

```yaml
dialogue:
  enabled: true
  min_turns: 2
  max_turns: 5
  include_single_qa: true
```

---

## 19. review — QA 리뷰 설정

> TUI에서 QA 쌍을 수동으로 검토하고 승인/거부/편집합니다.

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `enabled` | `bool` | `false` | QA 수동 리뷰 기능 활성화 여부 |
| `auto_open` | `bool` | `true` | 리뷰 완료 후 결과 파일을 자동으로 열지 여부 |
| `output_file` | `str` | `"qa_reviewed.json"` | 리뷰 결과 JSON 파일명. `paths.output` 디렉토리에 저장됩니다 |

```yaml
review:
  enabled: true
  auto_open: true
  output_file: "qa_reviewed.json"
```

---

## 20. dashboard — 대시보드 설정

> 파이프라인 진행 상태를 실시간 TUI로 모니터링합니다.

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `enabled` | `bool` | `false` | 대시보드 기능 활성화 여부 |
| `refresh_interval` | `float` | `2.0` | 대시보드 새로고침 간격 (초) |
| `theme` | `str` | `"dark"` | 대시보드 테마. `"dark"` 또는 `"light"` |

```yaml
dashboard:
  enabled: true
  refresh_interval: 2.0
  theme: "dark"
```

---

## 21. 설정 레시피

자주 사용되는 설정 패턴을 레시피 형태로 제공합니다. 각 레시피는 특정 상황에 맞게 최소한의 필드만 변경합니다.

각 설정을 언제 사용할지에 대한 맥락은 [사용 가이드](guide.md)를 참조하십시오.

### 빠른 시작 — 기본값으로 바로 실행

**상황**: 처음 slm-factory를 사용하며 기본 설정으로 빠르게 테스트하고 싶을 때.

`project.yaml`을 수정할 필요가 없습니다. `slm-factory init my-project`로 생성된 기본 템플릿을 그대로 사용하면 됩니다.

사전 준비:

```bash
ollama serve
ollama pull qwen3:8b
# documents/ 디렉토리에 PDF 또는 TXT 파일 추가
slm-factory tool wizard --config my-project/project.yaml
```

기본 설정은 영어 문서, Ollama 백엔드, Gemma 3 1B 모델을 사용합니다.

---

### 소량 문서 (5개 이하) — 과적합 방지를 위한 증강

**상황**: 문서가 5개 이하로 적어서 데이터가 부족하고 과적합이 우려될 때.

```yaml
# 데이터 증강으로 데이터셋 확장 (원본의 4배)
augment:
  enabled: true
  num_variants: 3
  max_concurrency: 4

# 저품질 QA 쌍 필터링
scoring:
  enabled: true
  threshold: 3.5
  max_concurrency: 4

# 과적합 방지를 위한 학습 설정
training:
  num_epochs: 10
  lora:
    dropout: 0.1
  early_stopping:
    enabled: true
    patience: 3
```

데이터 증강은 Teacher LLM 호출 비용을 증가시키므로 로컬 Ollama를 사용하는 것이 좋습니다. 소량 데이터에서는 품질 점수 평가도 함께 활성화하는 것을 권장합니다.

---

### VRAM 제한 (8GB GPU) — 메모리 최적화

**상황**: RTX 3060, RTX 4060 등 8GB VRAM GPU에서 학습할 때.

```yaml
# 경량 Student 모델
student:
  model: "google/gemma-3-1b-it"
  max_seq_length: 2048

# 메모리 최적화 학습 설정
training:
  batch_size: 2
  gradient_accumulation_steps: 8  # 실제 배치 크기는 2x8=16으로 유지
  quantization:
    enabled: true
    bits: 4
  bf16: true
```

양자화를 사용하면 VRAM 사용량이 약 50% 감소하지만 품질 저하는 거의 없습니다. `max_seq_length`를 2048로 줄이면 긴 문서는 잘리지만 대부분의 QA 쌍은 이 길이 내에 들어갑니다.

---

### vLLM/OpenAI 백엔드 — 외부 API 연동

**상황**: Ollama 대신 vLLM 서버 또는 OpenAI API를 Teacher 모델로 사용할 때.

vLLM 서버:

```yaml
teacher:
  backend: "openai"
  model: "meta-llama/Llama-3.1-8B-Instruct"
  api_base: "http://localhost:8000/v1"
  api_key: "dummy"  # vLLM은 키가 필요 없지만 필드는 채워야 합니다
  temperature: 0.3
  max_concurrency: 16  # vLLM은 높은 동시성을 지원합니다
```

OpenAI API:

```yaml
teacher:
  backend: "openai"
  model: "gpt-4o-mini"
  api_base: "https://api.openai.com"
  api_key: "sk-proj-..."
  temperature: 0.3
  max_concurrency: 8  # OpenAI는 rate limit을 고려하여 조절합니다
```

`api_key`는 환경 변수 `OPENAI_API_KEY`로 설정하는 것이 안전합니다.

---

## 22. 전체 설정 예시 (한국어 정책 문서 프로젝트)

한국어 회사 정책 문서를 처리하는 완전한 `project.yaml` 예시입니다.

```yaml
# 한국어 회사 정책 문서 처리 프로젝트
project:
  name: "company-policy-assistant"
  version: "1.0.0"
  language: "ko"

paths:
  documents: "./documents"
  output: "./output"

# PDF와 HWPX 문서 모두 파싱
parsing:
  formats: ["pdf", "hwpx"]
  pdf:
    extract_tables: true
  hwpx:
    apply_spacing: true  # 한국어 띄어쓰기 교정 (pip install slm-factory[korean] 필요)

# 로컬 Ollama 서버 사용
teacher:
  backend: "ollama"
  model: "qwen3:8b"  # 한국어 지원 우수
  api_base: "http://localhost:11434"
  temperature: 0.3
  timeout: 180
  max_context_chars: 12000
  max_concurrency: 4

# 한국어 질문 카테고리
questions:
  categories:
    개요:
      - "이 문서의 주요 목적은 무엇입니까?"
      - "이 정책이 해결하려는 문제는 무엇입니까?"
      - "대상 사용자는 누구입니까?"
      - "주요 변경 사항은 무엇입니까?"
      - "적용 범위와 제한 사항은 무엇입니까?"
    기술:
      - "주요 기술 사양은 무엇입니까?"
      - "사용된 기술이나 도구는 무엇입니까?"
      - "시스템 요구 사항은 무엇입니까?"
      - "기존 방식과 어떻게 다릅니까?"
      - "성능 지표는 무엇입니까?"
    구현:
      - "구현 일정은 어떻게 됩니까?"
      - "필요한 자원은 무엇입니까?"
      - "구현 절차는 무엇입니까?"
      - "예상되는 위험은 무엇입니까?"
      - "기대 효과는 무엇입니까?"
  system_prompt: >
    당신은 제공된 문서를 기반으로 질문에 답변하는 도움이 되는 어시스턴트입니다.
    문서 내용에만 근거하여 답변하십시오. 추측하거나 정보를 만들어내지 마십시오.
    간결하고 사실적으로 답변하십시오. 가능한 경우 구체적인 숫자, 날짜, 이름을 포함하십시오.
    문서에 관련 정보가 없으면 "문서에 해당 정보가 포함되어 있지 않습니다"라고 답변하십시오.
  output_format: "alpaca"

# 한국어 거부 패턴 추가
validation:
  enabled: true
  min_answer_length: 20
  max_answer_length: 2000
  remove_empty: true
  deduplicate: true
  reject_patterns:
    # 영어 패턴
    - "(?i)i don't know"
    - "(?i)not (available|provided|mentioned|found)"
    - "(?i)the document does not contain"
    # 한국어 패턴
    - "(?i)알 수 없"
    - "(?i)정보가 없"
    - "(?i)언급되지 않"
    - "(?i)포함되어 있지 않"
    - "(?i)제공되지 않"
  groundedness:
    enabled: true
    model: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    threshold: 0.35

# 품질 점수 평가 (필요 시 활성화)
scoring:
  enabled: false
  threshold: 3.0
  max_concurrency: 4

# 데이터 증강 (필요 시 활성화)
augment:
  enabled: false
  num_variants: 2
  max_concurrency: 4

# 데이터 분석
analyzer:
  enabled: true
  output_file: "data_analysis.json"

# 한국어 지원 우수한 Student 모델
student:
  model: "Qwen/Qwen3-1.7B"
  max_seq_length: 4096

# 12GB VRAM 기준 설정
training:
  lora:
    r: 16
    alpha: 32
    dropout: 0.05
    target_modules: "auto"
    use_rslora: false
  batch_size: 4
  gradient_accumulation_steps: 4
  learning_rate: 2.0e-5
  lr_scheduler: "cosine"
  warmup_ratio: 0.1
  num_epochs: 20
  early_stopping:
    enabled: true
    patience: 3
    threshold: 0.01
  optimizer: "adamw_torch_fused"
  bf16: true
  train_split: 0.9
  save_strategy: "epoch"
  quantization:
    enabled: false  # 12GB VRAM이면 양자화 불필요

# Ollama로 내보내기
export:
  merge_lora: true
  output_format: "safetensors"
  ollama:
    enabled: true
    model_name: "company-policy-assistant"
    system_prompt: "당신은 회사 정책에 대해 답변하는 전문 어시스턴트입니다. 정확하고 간결하게 답변하십시오."
    parameters:
      temperature: 0.5  # 정책 문서는 일관성 있는 답변이 중요
      top_p: 0.9
      num_ctx: 4096

# 나머지 기능은 기본값 (비활성화) 사용
eval:
  enabled: false

compare:
  enabled: false

gguf_export:
  enabled: false

incremental:
  enabled: false

dialogue:
  enabled: false

review:
  enabled: false

dashboard:
  enabled: false
```

주요 설정 포인트:

1. `parsing.formats: ["pdf", "hwpx"]` — PDF와 HWPX 모두 파싱합니다
2. `hwpx.apply_spacing: true` — 한국어 띄어쓰기를 자동 교정합니다
3. `teacher.model: "qwen3:8b"` — 한국어 지원이 우수한 모델을 선택했습니다
4. 질문과 시스템 프롬프트를 모두 한국어로 작성하여 한국어 답변을 유도합니다
5. 한국어 거부 패턴 5개를 추가했습니다
6. `groundedness.model`을 다국어 지원 모델로 변경했습니다
7. `student.model: "Qwen/Qwen3-1.7B"` — 한국어 지원이 양호한 모델입니다
8. `export.ollama.parameters.temperature: 0.5` — 정책 문서는 일관성이 중요하므로 낮췄습니다

---

## 23. 설정 검증 (slm-factory check)

설정 파일이 올바른지 확인하려면 `check` 명령을 사용합니다:

```bash
slm-factory check --config project.yaml
```

`--config`를 생략하면 현재 디렉토리와 상위 디렉토리(2단계)에서 `project.yaml`을 자동으로 검색합니다.

### 검증 항목

| 항목 | 설명 |
|------|------|
| 설정 파일 | YAML 로드 및 Pydantic 검증 |
| 문서 디렉토리 | 존재 여부 및 파일 유무 |
| 출력 디렉토리 | 쓰기 가능 여부 |
| Ollama 연결 | 서버 연결 상태 및 모델 존재 여부 (`backend: "ollama"`일 때만) |

### 예시 출력

```bash
$ slm-factory check --config project.yaml
✓ Config file: Valid
✓ Documents directory: Found (5 files)
✓ Output directory: Writable
✓ Ollama connection: Connected (model: qwen3:8b)

모든 점검 통과!
```

모든 항목이 통과하면 종료 코드 0, 실패하면 1을 반환합니다. 설정 파일에 오류가 있으면 (YAML 구문 오류, 잘못된 타입, 알 수 없는 필드 등) Pydantic 검증 단계에서 상세한 에러 메시지를 출력합니다.

설정 문제를 진단할 때는 `slm-factory -v check` 명령으로 상세 로그를 확인할 수 있습니다.

---

## 관련 문서

- [사용 가이드](guide.md) — 각 설정을 언제, 어떻게 사용할지에 대한 실전 안내
- [CLI 레퍼런스](cli-reference.md) — 모든 CLI 명령어와 옵션
- [아키텍처 가이드](architecture.md) — 설정 로딩 프로세스, 내부 구조, 설계 패턴
- [빠른 참조](quick-reference.md) — 자주 쓰는 설정 한눈에 보기
