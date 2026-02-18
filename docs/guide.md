# 사용 가이드

> slm-factory 설치부터 모델 배포까지, 단계별로 안내합니다.

---

## 목차

- [1. 시작하기 전에](#1-시작하기-전에)
- [2. 첫 번째 프로젝트 (튜토리얼)](#2-첫-번째-프로젝트-튜토리얼)
- [3. 수동 파이프라인 실행](#3-수동-파이프라인-실행)
- [4. 활용 시나리오](#4-활용-시나리오)
- [5. 데이터 품질 관리](#5-데이터-품질-관리)
- [6. 모델 평가 및 배포](#6-모델-평가-및-배포)
- [7. 트러블슈팅](#7-트러블슈팅)
- [관련 문서](#관련-문서)

---

## 1. 시작하기 전에

### 시스템 요구사항

slm-factory를 실행하려면 다음 환경이 필요합니다.

| 항목 | 최소 요구사항 | 권장 사양 |
|------|-------------|---------|
| **Python** | 3.11 이상 | 3.11 또는 3.12 |
| **GPU** | CPU 가능 (매우 느림) | CUDA 지원 GPU, VRAM 8GB 이상 |
| **Ollama** | 1.0 이상 | 최신 버전 |
| **디스크** | 5GB 이상 | 20GB 이상 (모델 여러 개 보관 시) |
| **RAM** | 8GB | 16GB 이상 |

GPU 없이도 학습이 가능하지만, GPU 대비 10~100배 느립니다. 테스트 목적이 아니라면 GPU 환경을 권장합니다.

---

### Ollama 설치 및 설정

slm-factory는 Teacher LLM으로 Ollama를 기본 사용합니다. 파이프라인 실행 전에 Ollama를 준비해야 합니다.

**1단계: Ollama 설치**

[ollama.com](https://ollama.com)에서 운영체제에 맞는 설치 파일을 다운로드합니다.

```bash
# Linux (curl 설치)
curl -fsSL https://ollama.com/install.sh | sh

# macOS: ollama.com에서 .dmg 파일 다운로드
# Windows: ollama.com에서 .exe 파일 다운로드
```

**2단계: Ollama 서버 실행**

별도 터미널 창에서 서버를 실행합니다. 파이프라인이 실행되는 동안 계속 켜두어야 합니다.

```bash
ollama serve
```

**3단계: Teacher 모델 다운로드**

한국어와 영어를 모두 지원하는 `qwen3:8b`를 권장합니다.

```bash
# 권장 Teacher 모델 (한국어/영어 모두 지원)
ollama pull qwen3:8b

# 영어 전용 대안
ollama pull llama3.1:8b

# 고품질 대안 (더 느림)
ollama pull gemma2:9b
```

다운로드 완료 후 정상 동작을 확인합니다.

```bash
ollama run qwen3:8b "안녕하세요"
```

---

### slm-factory 설치

```bash
# 1. 저장소 클론
git clone https://github.com/DevDnA/slm-factory.git
cd slm-factory

# 2. 가상환경 생성 및 활성화
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. 전체 의존성 설치
pip install -e ".[all]"

# 4. 설치 확인
slm-factory version
```

> **주의**: 가상환경 활성화는 필수입니다. 시스템 Python에 직접 설치하면 `externally-managed-environment` 오류가 발생합니다 (PEP 668).

`pip install -e ".[all]"` 명령은 PDF/HTML/TXT/HWPX/DOCX 파싱, 한국어 띄어쓰기 교정, 임베딩 기반 검증, 테스트 도구, Shell 자동완성을 포함한 모든 기능을 설치합니다.

---

### Shell 자동완성 설정

Tab 키로 명령어와 옵션을 자동완성할 수 있습니다.

```bash
# 자동완성 설치 (bash/zsh/fish/PowerShell 자동 감지)
slm-factory --install-completion

# 셸 재시작 후 적용됩니다. 즉시 적용하려면:
source ~/.bashrc    # bash
source ~/.zshrc     # zsh
```

자동완성이 작동하지 않으면 현재 셸의 스크립트를 직접 확인합니다.

```bash
slm-factory --show-completion
```

---

## 2. 첫 번째 프로젝트 (튜토리얼)

### 프로젝트 생성

`init` 명령으로 프로젝트 디렉토리와 기본 설정 파일을 생성합니다.

```bash
slm-factory init my-first-project
```

다음 구조가 생성됩니다.

```
my-first-project/
├── documents/      # 학습할 문서를 여기에 넣습니다
├── output/         # 파이프라인 결과물이 저장됩니다
└── project.yaml    # 프로젝트 설정 파일
```

---

### 문서 준비

`documents/` 디렉토리에 학습할 문서를 복사합니다. PDF, HWPX, HTML, TXT, MD, DOCX 형식을 지원합니다.

```bash
# PDF 문서 복사
cp /path/to/your/documents/*.pdf my-first-project/documents/

# 또는 여러 형식 혼합
cp /path/to/docs/*.pdf /path/to/docs/*.txt my-first-project/documents/
```

문서가 많을수록 더 많은 QA 쌍이 생성되어 학습 품질이 높아집니다. 최소 100개 이상의 QA 쌍 생성을 위해 충분한 문서를 준비하십시오.

---

### 환경 점검

wizard 실행 전에 `check` 명령으로 설정과 환경을 사전 점검합니다.

```bash
slm-factory check --config my-first-project/project.yaml
```

점검 항목은 다음과 같습니다.

- 설정 파일 로드 및 유효성 검사
- 문서 디렉토리 존재 여부 및 파일 유무
- 출력 디렉토리 쓰기 권한
- Ollama 서버 연결 상태
- Teacher 모델 사용 가능 여부

모든 항목이 통과되면 `모든 점검 통과!` 메시지와 함께 wizard 실행 명령을 안내합니다.

---

### Wizard 실행

`tool wizard`는 처음 사용자에게 권장하는 대화형 파이프라인입니다. 각 단계를 확인하며 진행하고, 선택 단계는 건너뛸 수 있습니다.

```bash
slm-factory tool wizard --config my-first-project/project.yaml
```

wizard는 다음 12단계를 순서대로 안내합니다.

| # | 단계 | 필수/선택 | 설명 |
|---|------|:---------:|------|
| 1 | 설정 파일 로드 | **필수** | `project.yaml`을 로드하고 Teacher/Student 모델을 확인합니다 |
| 2 | 문서 선택 | **필수** | 전체 또는 개별 문서를 선택합니다 |
| 3 | 문서 파싱 | **필수** | 선택한 문서에서 텍스트와 표를 자동 추출합니다 |
| 4 | QA 쌍 생성 | **필수** | Teacher LLM이 문서 기반 질문-답변 쌍을 생성합니다 |
| 5 | QA 검증 | **필수** | 규칙 기반 및 임베딩 기반으로 저품질 QA를 필터링합니다 |
| 6 | 품질 점수 평가 | 선택 | Teacher LLM이 각 QA를 1~5점으로 평가하여 추가 필터링합니다 |
| 7 | 데이터 증강 | 선택 | 질문을 다양한 표현으로 변형하여 학습 데이터를 늘립니다 |
| 8 | LoRA 학습 | **필수** | Student 모델에 LoRA 어댑터를 적용하여 파인튜닝합니다 |
| 9 | 모델 내보내기 | **필수** | LoRA 어댑터를 기본 모델에 병합하고 Ollama Modelfile을 생성합니다 |
| 10 | 멀티턴 대화 생성 | 선택 | QA 쌍을 멀티턴 대화 형식으로 확장합니다 |
| 11 | GGUF 변환 | 선택 | llama.cpp 호환 GGUF 양자화 형식으로 변환합니다 |
| 12 | 모델 평가 | 선택 | BLEU/ROUGE 메트릭으로 학습 결과를 자동 평가합니다 |

선택 단계를 건너뛰면 나중에 개별 실행할 수 있는 CLI 명령어를 안내합니다.

---

### 결과 확인

wizard 완료 후 `status` 명령으로 각 단계의 결과물을 확인합니다.

```bash
slm-factory status --config my-first-project/project.yaml
```

출력 예시:

```
┌──────────────────────────────────────────────────────────┐
  단계       파일                      상태    건수
  parse      parsed_documents.json     존재    5개 문서
  generate   qa_alpaca.json            존재    150개 쌍
  score      qa_scored.json            없음    -
  augment    qa_augmented.json         없음    -
  analyze    data_analysis.json        존재    1개 항목
  convert    training_data.jsonl       존재    150개 줄
  train      checkpoints/adapter/      존재    디렉토리
  export     merged_model/             존재    디렉토리
└──────────────────────────────────────────────────────────┘
모든 단계가 완료되었습니다
```

---

### 모델 테스트

내보내기가 완료되면 Ollama에 모델을 등록하고 바로 테스트할 수 있습니다.

```bash
# 병합된 모델 디렉토리로 이동
cd my-first-project/output/merged_model

# Ollama에 모델 등록
ollama create my-first-project-model -f Modelfile

# 대화 테스트
ollama run my-first-project-model
```

`ollama run` 실행 후 `>>>` 프롬프트에서 질문을 입력하면 학습된 도메인 지식으로 답변합니다.

---

## 3. 수동 파이프라인 실행

wizard 없이 각 단계를 직접 제어하려면 `run` 명령을 사용합니다. 자동화 스크립트나 CI/CD 환경에 적합합니다.

### 설정 파일 편집

`project.yaml`을 열어 주요 항목을 수정합니다. 전체 설정 옵션은 [설정 레퍼런스](configuration.md)를 참조하십시오.

```yaml
project:
  name: "my-project"
  language: "ko"              # "ko" 또는 "en"

teacher:
  backend: "ollama"           # "ollama" 또는 "openai"
  model: "qwen3:8b"           # Teacher 모델 이름
  api_base: "http://localhost:11434"
  temperature: 0.3

student:
  model: "google/gemma-3-1b-it"   # HuggingFace 모델 ID

export:
  ollama:
    model_name: "my-project-model"  # Ollama에 등록할 이름
    system_prompt: "당신은 전문 도우미입니다."
```

---

### 전체 파이프라인

설정 파일 하나로 문서 파싱부터 모델 배포까지 전체 파이프라인을 한 번에 실행합니다.

```bash
slm-factory run --config project.yaml
```

---

### 단계별 실행

`--until` 옵션으로 특정 단계까지만 실행하고 중단할 수 있습니다. 결과를 확인하며 단계적으로 진행할 때 유용합니다.

```bash
# 문서 파싱만 실행
slm-factory run --until parse --config project.yaml

# 파싱 + QA 생성
slm-factory run --until generate --config project.yaml

# + QA 검증
slm-factory run --until validate --config project.yaml

# + 품질 점수 평가 (scoring.enabled: true 필요)
slm-factory run --until score --config project.yaml

# + 데이터 증강 (augment.enabled: true 필요)
slm-factory run --until augment --config project.yaml

# 학습만 실행 (QA 데이터가 이미 있을 때)
slm-factory train --config project.yaml

# 내보내기만 실행 (학습이 완료된 후)
slm-factory export --config project.yaml
```

---

### 파이프라인 재개

중간에 중단된 파이프라인은 `--resume` 옵션으로 이어서 실행합니다. 중간 저장 파일을 자동으로 감지하여 가장 최근 완료 단계부터 재개합니다.

```bash
slm-factory run --resume --config project.yaml
```

재개 지점은 다음 순서로 탐색합니다.

1. `qa_augmented.json` 존재 시 → analyze 단계부터
2. `qa_scored.json` 존재 시 → augment 단계부터
3. `qa_alpaca.json` 존재 시 → validate 단계부터
4. `parsed_documents.json` 존재 시 → generate 단계부터
5. 없으면 처음부터 실행

모든 명령어 옵션은 [CLI 레퍼런스](cli-reference.md)를 참조하십시오.

---

## 4. 활용 시나리오

### 한국어 정책 문서 (HWPX) → 정책 전문 모델

한국 정부 정책 문서(HWPX 형식)를 학습하여 정책 질의응답 모델을 생성합니다. `apply_spacing: true`로 한국어 띄어쓰기 교정을 활성화하면 파싱 품질이 향상됩니다.

**설정 파일** (`policy-project/project.yaml`):

```yaml
project:
  name: "policy-assistant"
  language: "ko"

paths:
  documents: "./documents"
  output: "./output"

parsing:
  formats: ["hwpx"]
  hwpx:
    apply_spacing: true          # 한국어 띄어쓰기 교정 활성화

teacher:
  backend: "ollama"
  model: "qwen3:8b"
  api_base: "http://localhost:11434"
  temperature: 0.3
  timeout: 180
  max_context_chars: 12000

questions:
  system_prompt: >
    당신은 제공된 문서를 기반으로 질문에 답변하는 전문가입니다.
    문서 내용에만 근거하여 정확하고 상세하게 답변하십시오.
    구체적인 숫자, 날짜, 이름을 포함하십시오.
    문서에 정보가 없으면 "문서에 해당 정보가 포함되어 있지 않습니다"라고 답변하십시오.
  categories:
    policy_overview:
      - "이 정책의 주요 목적은 무엇입니까?"
      - "정책 대상자는 누구입니까?"
      - "정책의 시행 기간은 언제입니까?"
      - "이 정책의 법적 근거는 무엇입니까?"
    policy_details:
      - "지원 내용과 규모는 어떻게 됩니까?"
      - "신청 자격 요건은 무엇입니까?"
      - "신청 절차는 어떻게 됩니까?"
      - "지원 금액 또는 혜택의 상한선은 얼마입니까?"
    policy_faq:
      - "이 정책에서 제외되는 대상은 누구입니까?"
      - "신청 기한은 언제까지입니까?"
      - "담당 기관은 어디입니까?"

validation:
  enabled: true
  min_answer_length: 20
  max_answer_length: 2000
  groundedness:
    enabled: true
    threshold: 0.3

scoring:
  enabled: true
  threshold: 3.5

student:
  model: "google/gemma-3-1b-it"
  max_seq_length: 4096

training:
  num_epochs: 20
  batch_size: 4
  learning_rate: 2e-5
  quantization:
    enabled: true
    bits: 4

export:
  ollama:
    model_name: "policy-assistant-ko"
    system_prompt: "당신은 한국 정부 정책 전문 상담 도우미입니다. 정확하고 친절하게 답변하십시오."
```

**실행**:

```bash
# 환경 점검
slm-factory check --config policy-project/project.yaml

# 전체 파이프라인 실행
slm-factory run --config policy-project/project.yaml

# 모델 배포 및 테스트
cd policy-project/output/merged_model
ollama create policy-assistant-ko -f Modelfile
ollama run policy-assistant-ko
```

---

### 영문 기술 문서 (PDF) → API 문서 전문 모델

소프트웨어 API 문서(PDF)를 학습하여 개발자 지원 모델을 생성합니다. 기술 문서는 낮은 temperature와 의미적 검증을 권장합니다.

**설정 파일** (`tech-docs/project.yaml`):

```yaml
project:
  name: "api-assistant"
  language: "en"

paths:
  documents: "./documents"
  output: "./output"

parsing:
  formats: ["pdf"]
  pdf:
    extract_tables: true         # 표 추출 활성화

teacher:
  backend: "ollama"
  model: "qwen3:8b"
  api_base: "http://localhost:11434"
  temperature: 0.2              # 기술 문서는 낮은 temperature 권장
  timeout: 180
  max_context_chars: 15000

questions:
  system_prompt: >
    You are an expert assistant that answers questions based on the provided documentation.
    Answer only based on the document content. Do not speculate or invent information.
    Be concise, accurate, and include specific details such as parameter names, types, and examples.
    If the information is not in the document, say "This information is not available in the documentation."
  categories:
    api_basics:
      - "What is the purpose of this API?"
      - "What are the authentication requirements?"
      - "What is the base URL for API requests?"
      - "What data formats does this API support?"
    api_usage:
      - "What are the available endpoints?"
      - "What parameters does this endpoint accept?"
      - "What is the expected response format?"
      - "What are common error codes and their meanings?"
      - "What are the rate limits for this API?"
    examples:
      - "Provide a code example for this functionality."
      - "What are best practices for using this API?"
      - "How do I handle authentication in a request?"

validation:
  enabled: true
  min_answer_length: 30
  max_answer_length: 3000
  groundedness:
    enabled: true               # 의미적 검증 활성화
    threshold: 0.3

scoring:
  enabled: true
  threshold: 3.0

augment:
  enabled: true
  num_variants: 2

student:
  model: "google/gemma-3-1b-it"
  max_seq_length: 4096

training:
  num_epochs: 15
  batch_size: 4
  learning_rate: 1.5e-5
  quantization:
    enabled: true
    bits: 4

export:
  ollama:
    model_name: "api-assistant"
    system_prompt: "You are a helpful API documentation assistant. Provide accurate, concise answers based on the documentation."
```

**실행**:

```bash
# 환경 점검
slm-factory check --config tech-docs/project.yaml

# 전체 파이프라인 실행
slm-factory run --config tech-docs/project.yaml

# 모델 배포
cd tech-docs/output/merged_model
ollama create api-assistant -f Modelfile
ollama run api-assistant
```

---

### 기존 QA 데이터로 학습만 실행

이미 준비된 QA 데이터셋이 있다면 파싱과 생성 단계를 건너뛰고 학습만 실행할 수 있습니다. 외부 데이터셋을 사용하거나 하이퍼파라미터를 반복 조정할 때 유용합니다.

**JSONL 데이터 형식** (`custom_qa.jsonl`):

```jsonl
{"messages": [{"role": "user", "content": "이 정책의 주요 목적은 무엇입니까?"}, {"role": "assistant", "content": "이 정책의 주요 목적은 중소기업의 디지털 전환을 지원하는 것입니다..."}]}
{"messages": [{"role": "user", "content": "신청 자격 요건은 무엇입니까?"}, {"role": "assistant", "content": "신청 자격은 상시 근로자 50인 미만의 중소기업으로..."}]}
```

**실행**:

```bash
slm-factory train --config project.yaml --data ./custom_qa.jsonl
```

---

### 증분 학습 (문서 추가 시)

기존 프로젝트에 새 문서를 추가할 때 전체 파이프라인을 재실행하지 않아도 됩니다. `tool update`는 해시 기반으로 변경된 문서만 감지하여 새 QA를 생성하고 기존 QA와 병합합니다.

```bash
# 새 문서를 documents/ 디렉토리에 추가
cp /path/to/new-documents/*.pdf my-project/documents/

# 변경된 문서만 처리하여 QA 업데이트
slm-factory tool update --config my-project/project.yaml

# 업데이트된 데이터로 재학습
slm-factory train --config my-project/project.yaml
```

---

## 5. 데이터 품질 관리

### 품질 점수 평가 (scoring)

Teacher LLM이 생성된 각 QA 쌍을 1~5점으로 평가하여 저품질 데이터를 자동으로 필터링합니다. 점수가 `threshold` 미만인 QA는 학습 데이터에서 제외됩니다.

**설정**:

```yaml
scoring:
  enabled: true
  threshold: 3.5        # 3.5점 이상만 통과 (기본값)
  max_concurrency: 4    # 동시 평가 요청 수
```

**동작**: Teacher LLM이 각 QA 쌍에 대해 "이 답변이 질문에 얼마나 정확하고 유용한가?"를 1~5점으로 평가합니다. 결과는 `output/qa_scored.json`에 저장됩니다.

품질 점수 평가는 추가 Teacher LLM 호출이 필요하므로 QA 수에 비례하여 시간이 더 소요됩니다.

---

### 데이터 증강 (augment)

Teacher LLM이 기존 질문을 다양한 표현으로 변형하여 학습 데이터를 늘립니다. 같은 의미의 질문을 여러 방식으로 표현하면 모델의 일반화 성능이 향상됩니다.

**설정**:

```yaml
augment:
  enabled: true
  num_variants: 2       # 질문당 생성할 변형 수 (기본값 2)
  max_concurrency: 4
```

**패러프레이즈 예시**:

| 원본 질문 | 증강된 변형 |
|---------|-----------|
| "이 정책의 주요 목적은 무엇입니까?" | "이 정책이 추구하는 핵심 목표는 무엇인가요?" |
| "이 정책의 주요 목적은 무엇입니까?" | "해당 정책을 시행하는 이유가 무엇입니까?" |

증강 결과는 `output/qa_augmented.json`에 저장됩니다.

---

### QA 수동 리뷰 (tool review)

자동 검증을 통과한 QA 쌍을 TUI에서 직접 확인하고 승인/거부/편집할 수 있습니다. 중요한 도메인 모델을 만들 때 최종 품질을 보장하는 데 유용합니다.

```bash
slm-factory tool review --config project.yaml
```

TUI에서 각 QA 카드를 확인하며 다음 작업을 수행합니다.

- **승인 (A)**: 해당 QA를 학습 데이터에 포함합니다
- **거부 (R)**: 해당 QA를 제외합니다
- **편집 (E)**: 질문 또는 답변을 직접 수정합니다

리뷰 결과는 `output/qa_reviewed.json`에 저장됩니다.

---

### 데이터 분석 리포트 (analyzer)

파이프라인 실행 중 자동으로 생성되는 분석 보고서입니다. `output/data_analysis.json`에 저장됩니다.

**분석 항목**:

- 카테고리별 QA 분포 (불균형 경고 포함)
- 문서별 QA 분포
- 질문 길이 통계 (평균, 최소, 최대)
- 답변 길이 통계
- 데이터 품질 경고 (너무 짧은 답변, 중복 질문 등)

분석 보고서를 활용하여 질문 카테고리를 조정하거나 부족한 문서를 추가하면 학습 데이터 품질을 개선할 수 있습니다.

---

## 6. 모델 평가 및 배포

### 자동 평가 (eval run)

학습된 모델을 BLEU/ROUGE 메트릭으로 자동 평가합니다.

- **BLEU**: 생성된 답변과 참조 답변의 n-gram 일치율을 측정합니다. 0~1 범위이며 높을수록 좋습니다.
- **ROUGE**: 재현율 기반 메트릭으로, 참조 답변의 내용이 생성된 답변에 얼마나 포함되는지 측정합니다.

```bash
slm-factory eval run --model my-project-model --config project.yaml
```

결과는 `output/eval_results.json`에 저장됩니다.

**결과 해석**: BLEU 0.3 이상, ROUGE-L 0.4 이상이면 일반적으로 양호한 수준입니다. 점수가 낮으면 학습 데이터 품질 개선이나 에포크 수 조정을 고려하십시오.

---

### 모델 비교 (eval compare)

파인튜닝 전후 모델의 답변을 나란히 비교하여 학습 효과를 확인합니다.

```bash
slm-factory eval compare \
  --base-model gemma:2b \
  --ft my-project-model \
  --config project.yaml
```

결과는 `output/compare_results.json`에 저장됩니다. 각 질문에 대한 Base 모델과 Fine-tuned 모델의 답변을 나란히 확인할 수 있습니다.

---

### GGUF 변환 (tool gguf)

병합된 모델을 llama.cpp 호환 GGUF 양자화 형식으로 변환합니다. GGUF 형식은 CPU에서도 빠르게 실행되며, 다양한 llama.cpp 기반 도구와 호환됩니다.

```bash
slm-factory tool gguf --config project.yaml
```

변환에는 llama.cpp의 convert 스크립트가 사용됩니다. 변환된 GGUF 파일은 `output/merged_model/` 디렉토리에 저장됩니다.

---

### Ollama 배포

학습이 완료된 모델을 Ollama에 등록하고 즉시 사용할 수 있습니다.

```bash
# 병합된 모델 디렉토리로 이동
cd output/merged_model

# Ollama에 모델 등록
ollama create my-project-model -f Modelfile

# 대화 테스트
ollama run my-project-model

# API 서버로 사용 (포트 11434)
ollama serve
# 다른 터미널에서: curl http://localhost:11434/api/chat -d '{"model":"my-project-model","messages":[{"role":"user","content":"안녕하세요"}]}'
```

Ollama에 등록된 모델은 `ollama list`로 확인하고, `ollama rm my-project-model`로 삭제할 수 있습니다.

---

## 7. 트러블슈팅

### Ollama 연결 실패

**증상**:
```
Error: Failed to connect to Ollama at http://localhost:11434
RuntimeError: Cannot connect to Ollama. Ollama가 실행 중인지 확인하세요.
```

**해결 방법**:

1. Ollama 서버가 실행 중인지 확인합니다.
   ```bash
   ollama serve
   ```

2. Ollama가 정상 응답하는지 테스트합니다.
   ```bash
   curl http://localhost:11434/api/tags
   ```

3. `project.yaml`의 `api_base` 포트가 올바른지 확인합니다.
   ```yaml
   teacher:
     api_base: "http://localhost:11434"
   ```

4. 방화벽이 11434 포트를 차단하는지 확인합니다.
   ```bash
   # Linux
   sudo ufw allow 11434
   ```

---

### GPU 메모리 부족 (CUDA OOM)

**증상**:
```
RuntimeError: CUDA out of memory. Tried to allocate X.XX GiB
torch.cuda.OutOfMemoryError
```

**해결 방법**:

방법 1: 양자화 활성화 (VRAM 사용량 약 50% 감소)
```yaml
training:
  quantization:
    enabled: true
    bits: 4
```

방법 2: 배치 크기 감소
```yaml
training:
  batch_size: 2                    # 기본값 4에서 감소
  gradient_accumulation_steps: 8   # 총 배치 크기 유지 (2×8=16)
```

방법 3: 시퀀스 길이 감소
```yaml
student:
  max_seq_length: 2048             # 기본값 4096에서 감소
```

방법 4: 더 작은 Student 모델 사용
```yaml
student:
  model: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
```

방법 5: CPU 학습 (느리지만 메모리 제약 없음)
```yaml
training:
  bf16: false                      # CPU는 bfloat16 미지원
  batch_size: 1
  gradient_accumulation_steps: 16
```

---

### HWPX 파싱 실패

**증상**:
```
Error: Failed to parse HWPX file - section0.xml not found
```

**해결 방법**:

- HWPX 파일이 손상되지 않았는지 확인합니다. 한글 프로그램에서 직접 열어보십시오.
- 파일이 암호화되어 있지 않은지 확인합니다.
- 파일 확장자가 `.hwpx`인지 확인합니다. `.hwp` 형식은 지원하지 않습니다.
- 최신 한글 버전에서 "다른 이름으로 저장 → HWPX"로 다시 저장해보십시오.

---

### pykospacing 설치 오류

**증상**:
```
ERROR: Could not install packages due to an OSError
```

**해결 방법**:

방법 1: Python 버전 확인 (3.11 이상 필요)
```bash
python --version
```

방법 2: Git 설치 확인 (pykospacing은 Git 저장소에서 설치됨)
```bash
git --version
```

방법 3: 수동 설치 시도
```bash
pip install git+https://github.com/haven-jeon/PyKoSpacing.git
```

방법 4: 한국어 띄어쓰기 교정 비활성화
```yaml
parsing:
  hwpx:
    apply_spacing: false
```

---

### 학습 데이터 부족 경고

**증상**:
```
Warning: Only 15 QA pairs generated. Recommend at least 100 for effective training.
```

**해결 방법**:

방법 1: 더 많은 문서 추가
```bash
cp /path/to/more/documents/*.pdf ./documents/
```

방법 2: 질문 카테고리 확장
```yaml
questions:
  categories:
    overview: [...]
    technical: [...]
    additional:
      - "이 문서의 핵심 내용을 요약하면 무엇입니까?"
      - "주요 제한 사항은 무엇입니까?"
      - "관련 법령 또는 규정은 무엇입니까?"
```

방법 3: Teacher 모델의 컨텍스트 크기 증가
```yaml
teacher:
  max_context_chars: 20000         # 기본값 12000에서 증가
```

---

### 빈 QA 응답 생성

**증상**: 대부분의 QA 쌍이 "The document does not contain this information." 또는 "문서에 해당 정보가 포함되어 있지 않습니다"로 생성됩니다.

**해결 방법**:

방법 1: Teacher 모델 타임아웃 증가
```yaml
teacher:
  timeout: 300                     # 기본값 180초에서 증가
```

방법 2: 다른 Teacher 모델 시도
```yaml
teacher:
  model: "llama3.1:8b"             # 또는 "gemma2:9b"
```

방법 3: 질문을 문서 내용에 맞게 조정합니다. 문서를 먼저 읽고 실제로 답변 가능한 질문으로 수정하십시오. 너무 일반적이거나 추상적인 질문은 피하십시오.

방법 4: System prompt 조정
```yaml
questions:
  system_prompt: >
    Answer the question based on the document.
    Provide detailed answers with specific information.
    If the exact answer is not in the document, provide related information from the document.
```

---

### 한국어 문서에서 영어 QA 생성

**증상**: 한국어 문서를 파싱했지만 질문과 답변이 영어로 생성됩니다.

**해결 방법**:

방법 1: 언어 설정 변경
```yaml
project:
  language: "ko"
```

방법 2: 한국어 질문 카테고리 사용
```yaml
questions:
  categories:
    개요:
      - "이 문서의 주요 목적은 무엇입니까?"
      - "대상 사용자는 누구입니까?"
    기술:
      - "주요 기술 사양은 무엇입니까?"
```

방법 3: 한국어 system_prompt 명시
```yaml
questions:
  system_prompt: >
    당신은 제공된 문서를 기반으로 질문에 답변하는 도움이 되는 어시스턴트입니다.
    반드시 한국어로 답변하십시오.
    문서 내용에만 근거하여 답변하십시오.
```

한국어 지원이 우수한 Teacher 모델은 `qwen3:8b`입니다. `llama3.1:8b`는 한국어 지원이 제한적입니다.

---

### Python 버전 문제

**증상**: 설치 중 Python 버전 관련 오류가 발생하거나 패키지 설치가 실패합니다.

**해결 방법**:

현재 Python 버전을 확인합니다.
```bash
python --version
python3 --version
```

출력이 `Python 3.11.x` 이상이어야 합니다. 이전 버전이라면 업그레이드하십시오.

```bash
# Ubuntu/Debian
sudo apt update && sudo apt install python3.11

# macOS (Homebrew)
brew install python@3.11
```

여러 Python 버전이 설치된 경우 명시적으로 지정합니다.
```bash
python3.11 -m venv .venv
python3.11 -m pip install -e ".[all]"
```

---

### 학습 중단 시 복구

**증상**: 학습 중 Ctrl+C로 중단하거나 시스템이 종료되었습니다.

**해결 방법**:

체크포인트가 자동 저장되어 있으므로 `--resume` 옵션으로 이어서 실행합니다.

```bash
# wizard로 재개 (자동으로 중단 지점 감지)
slm-factory tool wizard --resume --config project.yaml

# 또는 학습 데이터가 있으면 학습만 재실행
slm-factory train --config project.yaml --data output/training_data.jsonl
```

체크포인트 위치를 직접 확인하려면:
```bash
ls output/checkpoints/
# checkpoint-100, checkpoint-200 등 디렉토리 확인
```

---

### 모델 배포 방법

**증상**: 학습이 완료되었지만 모델을 어떻게 사용하는지 모릅니다.

**해결 방법**:

1. Ollama가 설치되어 있는지 확인합니다.
   ```bash
   ollama --version
   ```

2. 병합된 모델 디렉토리에서 Ollama에 등록합니다.
   ```bash
   cd output/merged_model
   ollama create my-project-model -f Modelfile
   ```

3. 대화를 시작합니다.
   ```bash
   ollama run my-project-model
   ```

4. 등록된 모델 목록을 확인합니다.
   ```bash
   ollama list
   ```

---

## 관련 문서

| 문서 | 내용 |
|------|------|
| [빠른 참조](quick-reference.md) | 자주 쓰는 명령어 한눈에 보기 |
| [CLI 레퍼런스](cli-reference.md) | 모든 명령어와 옵션 상세 설명 |
| [설정 레퍼런스](configuration.md) | `project.yaml` 전체 설정 옵션 |
| [아키텍처 가이드](architecture.md) | 내부 구조와 설계 원칙 |
| [개발 가이드](development.md) | 기여 방법, 테스트, 모듈 확장 |
