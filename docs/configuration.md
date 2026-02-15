# 설정 레퍼런스 (project.yaml)

## 1. 개요

`project.yaml`은 slm-factory 파이프라인의 모든 동작을 제어하는 중앙 설정 파일입니다. 코드 수정 없이 문서 파싱, QA 생성, 검증, 학습, 내보내기 등 전체 워크플로우를 구성할 수 있습니다.

### 1.1 설정 파일의 역할

- **파이프라인 제어**: 문서 파싱부터 모델 내보내기까지 전 단계를 YAML 파일 하나로 제어합니다
- **타입 안전성**: Pydantic v2 기반 검증으로 잘못된 설정을 실행 전에 감지합니다
- **기본값 제공**: 모든 필드에 합리적인 기본값이 설정되어 있어 필요한 부분만 수정하면 됩니다

### 1.2 로딩 프로세스

설정 파일은 다음 순서로 로드되고 검증됩니다:

1. **YAML 파일 읽기**: `yaml.safe_load()`로 YAML 파일을 Python 딕셔너리로 변환합니다
2. **None 값 제거**: 최상위 키 중 값이 `None`인 항목은 자동으로 제거되어 기본값이 적용됩니다
3. **Pydantic 검증**: `SLMConfig.model_validate()`로 타입과 제약 조건을 검증합니다
4. **디렉토리 생성**: `paths.ensure_dirs()`로 필요한 디렉토리를 자동 생성합니다

### 1.3 기본 템플릿 생성

새 프로젝트를 시작할 때는 다음 명령으로 기본 템플릿을 생성합니다:

```bash
slm-factory init --name my-project
```

이 명령은 `my-project/project.yaml` 파일을 생성하며, 모든 기본값이 주석과 함께 포함되어 있습니다.

### 1.4 파일 위치

`project.yaml` 파일은 프로젝트 루트 디렉토리에 위치해야 합니다:

```
my-project/
├── project.yaml          # 설정 파일
├── documents/            # 입력 문서 디렉토리
└── output/               # 출력 디렉토리 (자동 생성)
```

---

## 2. project — 프로젝트 메타데이터

프로젝트의 기본 정보를 정의합니다. 이 정보는 내보낸 모델의 이름과 버전 관리에 사용됩니다.

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `name` | `str` | `"my-project"` | 프로젝트 식별자. 내보낸 모델 이름에 반영됩니다 (예: `my-project-model`) |
| `version` | `str` | `"1.0.0"` | 시맨틱 버전 (Semantic Versioning). 모델 버전 관리에 사용됩니다 |
| `language` | `str` | `"en"` | 문서 언어 코드. `"en"` (영어), `"ko"` (한국어), `"ja"` (일본어) 등 |

### 예시

```yaml
project:
  name: "company-policy-assistant"
  version: "2.1.0"
  language: "ko"
```

---

## 3. paths — 경로 설정

입력 문서와 출력 파일의 위치를 지정합니다. 지정된 디렉토리가 없으면 자동으로 생성됩니다.

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `documents` | `Path` | `"./documents"` | 입력 문서가 저장된 디렉토리. 파싱할 PDF, HWPX 등의 파일을 여기에 배치합니다 |
| `output` | `Path` | `"./output"` | 모든 출력 파일이 저장되는 디렉토리. QA 데이터셋, 학습된 모델, 체크포인트 등이 여기에 생성됩니다. `ensure_dirs()`로 자동 생성됩니다 |

### 예시

```yaml
paths:
  documents: "./data/source_docs"
  output: "./results"
```

---

## 4. parsing — 문서 파싱 설정

입력 문서를 텍스트로 변환하는 파싱 단계의 설정입니다.

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `formats` | `list[str]` | `["pdf"]` | 파싱할 문서 형식 목록. 여러 형식을 동시에 지정할 수 있습니다 |
| `pdf` | `PdfOptions` | (하위 참조) | PDF 파싱 옵션 |
| `hwpx` | `HwpxOptions` | (하위 참조) | HWPX 파싱 옵션 |

### 4.1 지원 형식

| 형식 | 확장자 | 파서 | 필요 패키지 |
|------|--------|------|-------------|
| `pdf` | `.pdf` | `PDFParser` | `pymupdf` (기본 포함) |
| `hwpx` | `.hwpx` | `HWPXParser` | `beautifulsoup4`, `lxml` (기본 포함), `pykospacing` (선택, 띄어쓰기 교정용) |
| `html` | `.html`, `.htm` | `HTMLParser` | `beautifulsoup4` (기본 포함) |
| `txt` | `.txt` | `TextParser` | 없음 |
| `md` | `.md` | `TextParser` | 없음 |

### 4.2 PDF 옵션 (`pdf`)

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `extract_tables` | `bool` | `true` | PDF 내 표를 마크다운 형식으로 추출합니다. `false`로 설정하면 표를 일반 텍스트로 처리합니다 |

### 4.3 HWPX 옵션 (`hwpx`)

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `apply_spacing` | `bool` | `true` | 한국어 띄어쓰기 교정을 적용합니다. `pykospacing` 패키지가 필요하며, `pip install slm-factory[korean]`으로 설치할 수 있습니다 |

### 4.4 여러 형식 동시 파싱 예시

```yaml
parsing:
  formats: ["pdf", "hwpx", "html", "txt"]
  pdf:
    extract_tables: true
  hwpx:
    apply_spacing: true
```

이 설정은 `documents/` 디렉토리 내 모든 PDF, HWPX, HTML, TXT 파일을 파싱합니다.

---

## 5. teacher — Teacher LLM 설정

QA 쌍을 생성하는 Teacher 모델의 설정입니다. Ollama 또는 OpenAI 호환 API를 사용할 수 있습니다.

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `backend` | `"ollama"` \| `"openai"` | `"ollama"` | 사용할 LLM 백엔드. `"ollama"`는 로컬 Ollama 서버, `"openai"`는 OpenAI 호환 API를 의미합니다 |
| `model` | `str` | `"qwen3:8b"` | 모델 이름 또는 ID. Ollama는 모델 태그 (예: `qwen3:8b`), OpenAI는 모델 ID (예: `gpt-4o-mini`)를 사용합니다 |
| `api_base` | `str` | `"http://localhost:11434"` | API 엔드포인트 URL. Ollama 기본값은 `http://localhost:11434`, OpenAI는 `https://api.openai.com`입니다 |
| `api_key` | `str \| None` | `null` | API 키. `backend: "openai"`일 때 필수입니다. Ollama는 불필요합니다 |
| `temperature` | `float` | `0.3` | 생성 온도. 낮을수록 일관성 있고 결정적인 답변, 높을수록 다양한 답변을 생성합니다 (0.0~1.0) |
| `timeout` | `int` | `180` | API 요청 타임아웃 (초). 긴 문서 처리 시 늘려야 할 수 있습니다 |
| `max_context_chars` | `int` | `12000` | 문서 내용 잘라내기 한계 (문자 수). 이 길이를 초과하는 문서는 앞부분만 사용됩니다 |
| `max_concurrency` | `int` | `4` | 비동기 QA 생성 시 최대 동시 요청 수. 높이면 빠르지만 서버 부하 증가 |

### 5.1 Ollama 백엔드 설정 예시

로컬 Ollama 서버를 사용하는 기본 설정입니다:

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

#### 권장 Ollama 모델

| 모델 | 장점 | 단점 | 용도 |
|------|------|------|------|
| `qwen3:8b` | 다국어 지원 우수, 빠른 속도 | 영어 전문 문서에서는 Llama보다 약간 낮은 품질 | 한국어/중국어 문서, 일반 용도 |
| `llama3.1:8b` | 영어 품질 우수, 안정적 | 한국어 지원 제한적 | 영어 문서, 기술 문서 |
| `gemma2:9b` | Google 모델, 균형 잡힌 성능 | 메모리 사용량 높음 | 고품질 QA 생성 |

### 5.2 OpenAI 호환 백엔드 설정 예시

OpenAI API, vLLM, LiteLLM, OpenRouter 등 `/v1/chat/completions` 엔드포인트를 제공하는 모든 서비스와 호환됩니다.

#### OpenAI API 사용

```yaml
teacher:
  backend: "openai"
  model: "gpt-4o-mini"
  api_base: "https://api.openai.com"
  api_key: "sk-proj-..."
  temperature: 0.3
```

#### vLLM 서버 사용

```yaml
teacher:
  backend: "openai"
  model: "meta-llama/Llama-3.1-8B-Instruct"
  api_base: "http://localhost:8000/v1"
  api_key: "dummy"  # vLLM은 키가 필요 없지만 필드는 채워야 함
  temperature: 0.3
```

#### LiteLLM 프록시 사용

```yaml
teacher:
  backend: "openai"
  model: "claude-3-5-sonnet-20241022"
  api_base: "http://localhost:4000"
  api_key: "sk-1234"
  temperature: 0.3
```

---

## 6. questions — 질문 설정

Teacher 모델이 문서에 대해 생성할 질문을 정의합니다. 카테고리별 질문 목록 또는 외부 파일로 지정할 수 있습니다.

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `categories` | `dict[str, list[str]]` | (하위 참조) | 카테고리별 질문 목록. 키는 카테고리 이름, 값은 질문 문자열 리스트입니다 |
| `file` | `Path \| None` | `null` | 외부 질문 파일 경로. 한 줄에 하나씩 질문을 작성합니다. 이 필드가 설정되면 `categories`는 무시됩니다 |
| `system_prompt` | `str` | (하위 참조) | Teacher 모델에게 전달되는 시스템 프롬프트. 답변 스타일과 제약 조건을 정의합니다 |
| `output_format` | `str` | `"alpaca"` | QA 데이터셋 출력 형식. 현재는 `"alpaca"` 형식만 지원합니다 |

### 6.1 기본 질문 카테고리

기본 템플릿에는 3개 카테고리에 총 15개 질문이 포함되어 있습니다:

#### overview (개요)

1. "What is the main purpose of this document?" (이 문서의 주요 목적은 무엇입니까?)
2. "What problem does this aim to solve?" (이 문서가 해결하려는 문제는 무엇입니까?)
3. "Who are the target users or beneficiaries?" (대상 사용자 또는 수혜자는 누구입니까?)
4. "What are the key changes or innovations described?" (설명된 주요 변경 사항이나 혁신은 무엇입니까?)
5. "What is the scope and limitations?" (범위와 제한 사항은 무엇입니까?)

#### technical (기술)

1. "What are the key technical details or specifications?" (주요 기술 세부 사항이나 사양은 무엇입니까?)
2. "What technologies, methods, or tools are described?" (어떤 기술, 방법 또는 도구가 설명되어 있습니까?)
3. "What are the system requirements?" (시스템 요구 사항은 무엇입니까?)
4. "How does this compare to existing alternatives?" (기존 대안과 비교하면 어떻습니까?)
5. "What are the performance metrics or benchmarks?" (성능 지표나 벤치마크는 무엇입니까?)

#### implementation (구현)

1. "What is the timeline or schedule?" (일정이나 스케줄은 어떻게 됩니까?)
2. "What resources (budget, personnel, infrastructure) are required?" (어떤 자원(예산, 인력, 인프라)이 필요합니까?)
3. "What are the step-by-step implementation procedures?" (단계별 구현 절차는 무엇입니까?)
4. "What risks or challenges are anticipated?" (예상되는 위험이나 과제는 무엇입니까?)
5. "What are the expected outcomes or deliverables?" (예상되는 결과물이나 산출물은 무엇입니까?)

### 6.2 기본 시스템 프롬프트

```
You are a helpful assistant that answers questions based strictly on the provided document. Answer only from the document content. Do not speculate or fabricate information. Be concise and factual. Include specific numbers, dates, and names when available. If the document does not contain relevant information, say "The document does not contain this information."
```

이 프롬프트는 Teacher 모델이 문서 내용에만 기반하여 답변하도록 제약합니다.

### 6.3 커스텀 질문 작성 가이드

#### 방법 1: YAML 내 카테고리 수정

기존 카테고리를 수정하거나 새 카테고리를 추가할 수 있습니다:

```yaml
questions:
  categories:
    overview:
      - "What is the main purpose of this document?"
      - "Who is the intended audience?"
    custom_category:
      - "What are the legal implications?"
      - "What are the compliance requirements?"
```

#### 방법 2: 외부 파일 사용

`questions.txt` 파일을 생성하고 한 줄에 하나씩 질문을 작성합니다:

```text
What is the main purpose of this document?
What are the key technical specifications?
What are the implementation steps?
What are the expected outcomes?
```

그런 다음 `project.yaml`에서 파일을 참조합니다:

```yaml
questions:
  file: "./questions.txt"
```

이 경우 `categories` 필드는 무시됩니다.

#### 한국어 프로젝트 질문 예시

한국어 문서를 처리할 때는 질문도 한국어로 작성하는 것이 좋습니다:

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
```

---

## 7. validation — QA 검증 설정

생성된 QA 쌍의 품질을 검증하고 필터링하는 규칙을 정의합니다.

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `enabled` | `bool` | `true` | 검증 기능 활성화 여부. `false`로 설정하면 모든 검증을 건너뜁니다 |
| `min_answer_length` | `int` | `20` | 답변 최소 길이 (문자 수). 이보다 짧은 답변은 거부됩니다 |
| `max_answer_length` | `int` | `2000` | 답변 최대 길이 (문자 수). 이보다 긴 답변은 거부됩니다 |
| `remove_empty` | `bool` | `true` | 빈 질문이나 답변을 제거합니다 |
| `deduplicate` | `bool` | `true` | 중복된 질문-답변 쌍을 제거합니다 |
| `reject_patterns` | `list[str]` | (하위 참조) | 답변에서 거부할 정규식 패턴 목록. 패턴이 매칭되면 해당 QA 쌍을 거부합니다 |
| `groundedness` | `GroundednessConfig` | (하위 참조) | 의미적 근거성 검증 설정 |

### 7.1 기본 거부 패턴

다음 3개 패턴이 기본으로 설정되어 있습니다:

| 패턴 | 설명 | 예시 |
|------|------|------|
| `"(?i)i don't know"` | "모릅니다" 류의 답변 거부 | "I don't know", "I do not know" |
| `"(?i)not (available\|provided\|mentioned\|found)"` | 정보 부재 표현 거부 | "not available", "not provided", "not mentioned", "not found" |
| `"(?i)the document does not contain"` | 명시적 정보 부재 거부 | "The document does not contain this information" |

`(?i)` 플래그는 대소문자를 구분하지 않습니다.

### 7.2 한국어 거부 패턴 추가

한국어 문서를 처리할 때는 한국어 패턴도 추가하는 것이 좋습니다:

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
```

### 7.3 groundedness — 의미적 검증

답변이 원본 문서 내용에 근거하고 있는지 의미적으로 검증합니다. 임베딩 모델을 사용하여 답변과 문서 간 코사인 유사도를 계산합니다.

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `enabled` | `bool` | `false` | 의미적 검증 활성화 여부. `pip install slm-factory[validation]`이 필요합니다 |
| `model` | `str` | `"all-MiniLM-L6-v2"` | 사용할 sentence-transformers 모델. 경량 모델이지만 다국어 지원이 제한적입니다 |
| `threshold` | `float` | `0.3` | 코사인 유사도 임계값 (0.0~1.0). 높을수록 엄격하게 검증합니다 |

#### 작동 방식

1. 답변을 임베딩 벡터로 변환합니다
2. 원본 문서를 청크로 나누고 각 청크를 임베딩합니다
3. 답변과 각 문서 청크 간 코사인 유사도를 계산합니다
4. 최대 유사도가 `threshold`를 초과하면 통과, 그렇지 않으면 거부합니다

#### 한국어 문서용 설정

한국어 문서에는 다국어 임베딩 모델을 사용하는 것이 좋습니다:

```yaml
validation:
  groundedness:
    enabled: true
    model: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    threshold: 0.35
```

---

## 8. scoring — 품질 점수 평가 설정

생성된 QA 쌍의 품질을 Teacher LLM으로 평가하고 낮은 점수의 QA 쌍을 필터링합니다.

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `enabled` | `bool` | `false` | 품질 점수 평가 기능 활성화 여부. `true`로 설정하면 Teacher LLM이 각 QA 쌍을 1~5점으로 평가합니다 |
| `threshold` | `float` | `3.0` | QA 쌍 최소 합격 점수 (1.0~5.0). 이 점수 미만의 QA 쌍은 데이터셋에서 제거됩니다 |
| `max_concurrency` | `int` | `4` | 비동기 점수 평가 시 최대 동시 요청 수. 높이면 빠르지만 서버 부하가 증가합니다 |

### 8.1 작동 방식

1. 검증을 통과한 QA 쌍을 Teacher LLM에게 전달합니다
2. Teacher LLM이 질문의 명확성, 답변의 정확성, 문서와의 관련성을 종합하여 1~5점으로 평가합니다
3. `threshold` 미만의 점수를 받은 QA 쌍은 데이터셋에서 제거됩니다
4. 최종적으로 고품질 QA 쌍만 학습 데이터로 사용됩니다

### 8.2 점수 기준

- **5점**: 매우 명확한 질문, 정확하고 완전한 답변, 문서 내용과 완벽히 일치
- **4점**: 명확한 질문, 정확한 답변, 문서 내용과 잘 일치
- **3점**: 적절한 질문, 대체로 정확한 답변, 문서 내용과 일치
- **2점**: 모호한 질문 또는 불완전한 답변, 문서 내용과 부분적으로 일치
- **1점**: 불명확한 질문 또는 부정확한 답변, 문서 내용과 불일치

### 8.3 예시

```yaml
scoring:
  enabled: true
  threshold: 3.5  # 3.5점 이상만 통과
  max_concurrency: 4
```

이 설정은 모든 QA 쌍을 평가하여 3.5점 미만의 QA 쌍을 제거합니다. 품질은 높아지지만 데이터셋 크기는 줄어들 수 있습니다.

---

## 9. augment — 데이터 증강 설정

기존 QA 쌍의 질문을 패러프레이즈하여 데이터셋을 확장합니다.

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `enabled` | `bool` | `false` | 데이터 증강 기능 활성화 여부. `true`로 설정하면 Teacher LLM이 질문을 다양하게 재작성합니다 |
| `num_variants` | `int` | `2` | 각 QA 쌍당 생성할 패러프레이즈 변형 수. 원본 포함 총 `num_variants + 1`개의 QA 쌍이 생성됩니다 |
| `max_concurrency` | `int` | `4` | 비동기 증강 시 최대 동시 요청 수. 높이면 빠르지만 서버 부하가 증가합니다 |

### 9.1 작동 방식

1. 검증 및 점수 평가를 통과한 QA 쌍을 Teacher LLM에게 전달합니다
2. Teacher LLM이 원본 질문의 의미를 유지하면서 다양한 표현으로 재작성합니다
3. 각 QA 쌍당 `num_variants`개의 변형 질문이 생성됩니다
4. 변형된 질문과 원본 답변을 조합하여 새로운 QA 쌍을 생성합니다
5. 최종 데이터셋 크기는 원본의 `(num_variants + 1)`배가 됩니다

### 9.2 패러프레이즈 예시

**원본 질문**: "이 문서의 주요 목적은 무엇입니까?"

**변형 1**: "이 문서가 작성된 주된 이유는 무엇인가요?"

**변형 2**: "이 문서를 통해 달성하고자 하는 목표는 무엇입니까?"

### 9.3 예시

```yaml
augment:
  enabled: true
  num_variants: 3  # 각 QA 쌍당 3개 변형 생성 (총 4배 증강)
  max_concurrency: 4
```

이 설정은 100개의 QA 쌍을 400개로 확장합니다. 데이터가 부족할 때 유용하지만, Teacher LLM 호출 비용이 증가합니다.

---

## 10. analyzer — 데이터 분석 설정

생성된 QA 데이터셋의 통계 정보를 분석하고 보고서를 생성합니다.

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `enabled` | `bool` | `true` | 데이터 분석 기능 활성화 여부. `true`로 설정하면 파이프라인 종료 시 분석 보고서를 생성합니다 |
| `output_file` | `str` | `"data_analysis.json"` | 분석 보고서 JSON 파일명. `paths.output` 디렉토리에 저장됩니다 |

### 10.1 작동 방식

1. 최종 QA 데이터셋을 로드합니다
2. 순수 통계 분석을 수행합니다 (LLM 호출 없음)
3. 데이터 분포, 길이 통계, 경고 사항을 JSON 보고서로 저장합니다

### 10.2 분석 항목

분석 보고서에는 다음 정보가 포함됩니다:

- **데이터셋 크기**: 전체 QA 쌍 수, 원본/증강 QA 쌍 수
- **길이 통계**: 질문/답변의 최소, 최대, 평균, 중앙값, 표준편차 (문자 수 기준)
- **카테고리 분포**: 질문 카테고리별 QA 쌍 수
- **문서별 분포**: 원본 문서별 QA 쌍 수
- **경고 사항**: 데이터셋이 너무 작거나, 답변 길이 편차가 크거나, 문서별 불균형이 심하거나, 카테고리가 1개뿐인 경우 경고

### 10.3 보고서 예시

```json
{
  "total_pairs": 450,
  "original_pairs": 300,
  "augmented_pairs": 150,
  "category_distribution": {
    "개요": 150,
    "기술": 150,
    "구현": 150
  },
  "source_doc_distribution": {
    "policy_2024.pdf": 200,
    "guide_v2.pdf": 250
  },
  "answer_length_stats": {
    "min": 23.0,
    "max": 1456.0,
    "mean": 187.5,
    "median": 150.0,
    "stdev": 200.3
  },
  "question_length_stats": {
    "min": 10.0,
    "max": 95.0,
    "mean": 42.5,
    "median": 38.0,
    "stdev": 15.2
  },
  "quality_score_stats": {},
  "warnings": [
    "답변 길이의 편차가 매우 큽니다. 일부 답변이 비정상적으로 길거나 짧을 수 있습니다."
  ]
}
```

### 10.4 예시

```yaml
analyzer:
  enabled: true
  output_file: "qa_analysis_report.json"
```

이 설정은 파이프라인 종료 시 `output/qa_analysis_report.json` 파일을 생성합니다.

---

## 11. student — Student 모델 설정

파인튜닝할 Student 모델을 지정합니다. HuggingFace Hub의 모든 causal language model을 사용할 수 있습니다.

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `model` | `str` | `"google/gemma-3-1b-it"` | HuggingFace 모델 ID. 반드시 causal LM이어야 합니다 (예: GPT, Llama, Gemma) |
| `max_seq_length` | `int` | `4096` | 학습 데이터의 최대 토큰 길이. 이보다 긴 시퀀스는 잘립니다 |

### 11.1 권장 모델

| 모델 | HuggingFace ID | 파라미터 | VRAM | 특징 |
|------|----------------|----------|------|------|
| Gemma 3 1B IT | `google/gemma-3-1b-it` | 1B | ~4GB | Google, 다국어 지원, 빠른 학습 |
| Gemma 3 4B IT | `google/gemma-3-4b-it` | 4B | ~10GB | Google, 고품질 출력, 균형 잡힌 성능 |
| Llama 3.2 1B | `meta-llama/Llama-3.2-1B-Instruct` | 1B | ~4GB | Meta, 영어 강점, 경량 |
| Llama 3.2 3B | `meta-llama/Llama-3.2-3B-Instruct` | 3B | ~8GB | Meta, 영어 강점, 중간 크기 |
| Phi-4 Mini | `microsoft/Phi-4-mini-instruct` | 3.8B | ~10GB | Microsoft, 코드 생성 강점, 추론 능력 우수 |
| Qwen3 1.7B | `Qwen/Qwen3-1.7B` | 1.7B | ~5GB | Alibaba, 한국어 지원 양호, 다국어 |

### 11.2 모델 선택 가이드

- **8GB VRAM 이하**: 1B 모델 + 양자화 (`quantization.enabled: true`)
- **12GB VRAM**: 1B~3B 모델, 양자화 없이 학습 가능
- **24GB VRAM**: 4B 모델까지 여유롭게 학습 가능
- **한국어 문서**: `Qwen/Qwen3-1.7B` 또는 `google/gemma-3-1b-it` 권장
- **영어 문서**: `meta-llama/Llama-3.2-1B-Instruct` 또는 `google/gemma-3-1b-it` 권장
- **코드 생성**: `microsoft/Phi-4-mini-instruct` 권장

---

## 12. training — 학습 설정

LoRA 파인튜닝의 하이퍼파라미터와 학습 전략을 정의합니다.

### 12.1 lora — LoRA 어댑터 설정

LoRA (Low-Rank Adaptation) 어댑터의 구조를 정의합니다.

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `r` | `int` | `16` | LoRA rank (랭크). 높을수록 표현력이 증가하지만 메모리 사용량도 증가합니다. 일반적으로 8~64 사이 값을 사용합니다 |
| `alpha` | `int` | `32` | LoRA scaling factor (스케일링 인자). 일반적으로 `r`의 2배 값을 사용합니다 |
| `dropout` | `float` | `0.05` | LoRA 레이어의 드롭아웃 비율. 과적합 방지를 위한 정규화 기법입니다 (0.0~1.0) |
| `target_modules` | `str \| list[str]` | `"auto"` | LoRA를 적용할 모듈 이름. `"auto"`는 자동 감지, 또는 `["q_proj", "v_proj", "k_proj", "o_proj"]` 같은 리스트로 명시할 수 있습니다 |
| `use_rslora` | `bool` | `false` | Rank-Stabilized LoRA 사용 여부. 학습 안정성을 높이지만 약간 느려집니다 |

#### 예시

```yaml
training:
  lora:
    r: 32                    # 더 높은 표현력
    alpha: 64                # r의 2배
    dropout: 0.1             # 더 강한 정규화
    target_modules: "auto"
    use_rslora: false
```

### 12.2 학습 하이퍼파라미터

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `batch_size` | `int` | `4` | 디바이스당 배치 크기. GPU 메모리에 따라 조정합니다 |
| `gradient_accumulation_steps` | `int` | `4` | 그래디언트 누적 스텝. 실제 배치 크기는 `batch_size × gradient_accumulation_steps` (기본값: 4×4=16)입니다 |
| `learning_rate` | `float` | `2e-5` | 학습률. LoRA 파인튜닝에는 일반적으로 `1e-5 ~ 5e-5` 범위를 사용합니다 |
| `lr_scheduler` | `str` | `"cosine"` | 학습률 스케줄러. `"cosine"`, `"linear"`, `"constant"` 등을 지원합니다 |
| `warmup_ratio` | `float` | `0.1` | 워밍업 비율. 전체 학습 스텝의 10%를 워밍업에 사용합니다 (0.0~1.0) |
| `num_epochs` | `int` | `20` | 최대 에포크 수. 조기 종료가 활성화되면 이보다 일찍 멈출 수 있습니다 |
| `optimizer` | `str` | `"adamw_torch_fused"` | 옵티마이저. `"adamw_torch_fused"`는 PyTorch의 fused AdamW로 가장 빠릅니다 |
| `bf16` | `bool` | `true` | bfloat16 혼합 정밀도 학습 사용 여부. Ampere 이상 GPU (RTX 30xx, A100 등)에서만 지원됩니다 |
| `train_split` | `float` | `0.9` | 학습 데이터 비율. 0.9는 90% 학습, 10% 검증을 의미합니다 (0.0~1.0) |
| `save_strategy` | `str` | `"epoch"` | 체크포인트 저장 전략. `"epoch"`는 에포크마다, `"steps"`는 일정 스텝마다 저장합니다 |

### 12.3 GPU 메모리별 권장 설정

#### 8GB VRAM (RTX 3060, RTX 4060)

```yaml
training:
  batch_size: 2
  gradient_accumulation_steps: 8
  quantization:
    enabled: true
    bits: 4
student:
  model: "google/gemma-3-1b-it"
```

#### 12GB VRAM (RTX 3060 12GB, RTX 4070)

```yaml
training:
  batch_size: 4
  gradient_accumulation_steps: 4
  quantization:
    enabled: false
student:
  model: "google/gemma-3-1b-it"
```

#### 24GB VRAM (RTX 3090, RTX 4090, A5000)

```yaml
training:
  batch_size: 8
  gradient_accumulation_steps: 2
  quantization:
    enabled: false
student:
  model: "google/gemma-3-4b-it"
```

### 12.4 early_stopping — 조기 종료

검증 손실이 개선되지 않으면 학습을 조기에 종료합니다.

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `enabled` | `bool` | `true` | 조기 종료 활성화 여부 |
| `patience` | `int` | `3` | 개선이 없어도 기다릴 에포크 수. 3 에포크 동안 개선이 없으면 학습을 중단합니다 |
| `threshold` | `float` | `0.01` | 개선으로 인정할 최소 변화량. 검증 손실이 이전 최저값보다 `threshold` 이상 낮아져야 개선으로 간주합니다 |

#### 작동 방식

1. 매 에포크마다 검증 손실 (`eval_loss`)을 모니터링합니다
2. 검증 손실이 이전 최저값보다 `threshold` 이상 낮아지면 개선으로 간주하고 카운터를 리셋합니다
3. 개선이 없으면 카운터를 증가시킵니다
4. 카운터가 `patience`에 도달하면 학습을 중단합니다

#### 예시

```yaml
training:
  num_epochs: 50
  early_stopping:
    enabled: true
    patience: 5
    threshold: 0.005
```

이 설정은 최대 50 에포크까지 학습하지만, 5 에포크 동안 검증 손실이 0.005 이상 개선되지 않으면 조기 종료합니다.

### 12.5 quantization — 양자화

4비트 양자화를 사용하여 메모리 사용량을 크게 줄입니다.

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `enabled` | `bool` | `false` | 4비트 양자화 활성화 여부 |
| `bits` | `int` | `4` | 양자화 비트 수. 현재는 4비트만 지원합니다 |

#### 작동 방식

- BitsAndBytesConfig를 사용하여 NF4 (Normal Float 4) 양자화를 적용합니다
- 모델 가중치를 4비트로 압축하고, 계산은 bfloat16으로 수행합니다
- VRAM 사용량을 약 50~60% 줄일 수 있습니다

#### 장단점

**장점:**
- VRAM 사용량 대폭 감소 (8GB GPU에서도 3B 모델 학습 가능)
- 학습 속도는 거의 동일

**단점:**
- 약간의 품질 저하 가능 (일반적으로 미미함)
- CPU 오프로드 시 학습 속도 저하

#### 예시

```yaml
training:
  quantization:
    enabled: true
    bits: 4
```

---

## 13. export — 모델 내보내기 설정

학습된 모델을 내보내는 방식을 정의합니다.

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `merge_lora` | `bool` | `true` | LoRA 어댑터를 베이스 모델에 병합할지 여부. `true`면 독립 실행 가능한 모델을 생성하지만 크기가 큽니다. `false`면 어댑터만 저장하여 크기가 작지만 베이스 모델이 별도로 필요합니다 |
| `output_format` | `str` | `"safetensors"` | 모델 저장 형식. `"safetensors"` (권장) 또는 `"pytorch"` |
| `ollama` | `OllamaExportConfig` | (하위 참조) | Ollama 내보내기 설정 |

### 13.1 ollama — Ollama 내보내기

학습된 모델을 Ollama 형식으로 내보내고 로컬 Ollama 서버에 등록합니다.

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `enabled` | `bool` | `true` | Ollama 내보내기 활성화 여부. `true`면 Modelfile을 생성하고 `ollama create` 명령을 실행합니다 |
| `model_name` | `str` | `"my-project-model"` | Ollama에 등록할 모델 이름. `project.name` 값을 사용하는 것이 좋습니다 |
| `system_prompt` | `str` | `"You are a helpful domain-specific assistant."` | Ollama 모델의 시스템 프롬프트. 모델의 역할과 동작을 정의합니다 |
| `parameters` | `dict[str, Any]` | (하위 참조) | Ollama 런타임 파라미터 |

#### parameters (Ollama 파라미터)

| 키 | 타입 | 기본값 | 설명 |
|-----|------|--------|------|
| `temperature` | `float` | `0.7` | 생성 온도 (0.0~1.0) |
| `top_p` | `float` | `0.9` | Nucleus sampling 임계값 (0.0~1.0) |
| `num_ctx` | `int` | `4096` | 컨텍스트 윈도우 크기 (토큰 수) |

#### 생성되는 Modelfile 예시

```
FROM ./output/merged_model
SYSTEM """
You are a helpful domain-specific assistant.
"""
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER num_ctx 4096
```

이 Modelfile은 `output/` 디렉토리에 생성되며, `ollama create` 명령으로 자동 등록됩니다.

#### 한국어 모델 예시

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

## 14. 전체 설정 예시 — 한국어 프로젝트

한국어 문서를 처리하는 완전한 프로젝트 설정 예시입니다.

```yaml
# 한국어 회사 정책 문서 처리 프로젝트
project:
  name: "company-policy-assistant"
  version: "1.0.0"
  language: "ko"  # 한국어 문서

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

  # 한국어 시스템 프롬프트
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
    # 한국어 패턴 추가
    - "(?i)알 수 없"
    - "(?i)정보가 없"
    - "(?i)언급되지 않"
    - "(?i)포함되어 있지 않"
    - "(?i)제공되지 않"
  groundedness:
    enabled: true
    model: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # 다국어 임베딩 모델
    threshold: 0.35

# 품질 점수 평가 (선택적)
scoring:
  enabled: false
  threshold: 3.0
  max_concurrency: 4

# 데이터 증강 (선택적)
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
  model: "Qwen/Qwen3-1.7B"  # 한국어 지원 양호
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
```

### 주요 변경 사항 설명

1. **문서 형식**: PDF와 HWPX 모두 파싱하도록 설정했습니다
2. **띄어쓰기 교정**: `hwpx.apply_spacing: true`로 한국어 띄어쓰기를 자동 교정합니다
3. **Teacher 모델**: `qwen3:8b`는 한국어 지원이 우수합니다
4. **질문**: 모든 질문을 한국어로 작성했습니다
5. **시스템 프롬프트**: 한국어로 작성하여 한국어 답변을 유도합니다
6. **거부 패턴**: 한국어 패턴 5개를 추가했습니다
7. **임베딩 모델**: 다국어 지원 모델로 변경했습니다
8. **품질 점수 평가**: `scoring.enabled: false`로 비활성화했습니다 (필요시 활성화 가능)
9. **데이터 증강**: `augment.enabled: false`로 비활성화했습니다 (필요시 활성화 가능)
10. **데이터 분석**: `analyzer.enabled: true`로 활성화하여 QA 데이터셋 통계를 자동 생성합니다
11. **Student 모델**: `Qwen/Qwen3-1.7B`는 한국어 지원이 양호합니다
12. **Ollama 프롬프트**: 한국어로 작성했습니다
13. **Temperature**: 정책 문서는 일관성이 중요하므로 0.5로 낮췄습니다

---

## 부록: 설정 파일 검증

설정 파일이 올바른지 확인하려면, 가장 가벼운 파이프라인 단계인 `parse`를 실행하여 설정 로딩이 정상적으로 수행되는지 확인할 수 있습니다:

```bash
slm-factory parse --config project.yaml
```

설정 파일에 오류가 있으면(YAML 구문 오류, 잘못된 타입, 알 수 없는 필드 등) Pydantic 검증 단계에서 상세한 에러 메시지를 출력합니다.

---

## 관련 문서

- [README](../README.md) — 프로젝트 소개, 설치, 빠른 시작, CLI 레퍼런스
- [아키텍처 가이드](architecture.md) — 내부 구조, 설계 패턴, 데이터 흐름
- [모듈별 상세 문서](modules.md) — 각 모듈의 클래스, 함수, 확장 방법
