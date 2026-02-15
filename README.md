# slm-factory

도메인 문서를 학습하여 특화된 소형 언어모델(SLM)을 자동 생성하는 Teacher-Student 지식 증류 프레임워크

---

## 목차

- [1. 소개](#1-소개)
- [2. 파이프라인 흐름도](#2-파이프라인-흐름도)
- [3. 주요 기능](#3-주요-기능)
- [4. 기술 스택](#4-기술-스택)
- [5. 시스템 요구사항](#5-시스템-요구사항)
- [6. 설치](#6-설치)
- [7. 빠른 시작 (wizard)](#7-빠른-시작-wizard)
- [8. 고급: 수동 파이프라인 실행](#8-고급-수동-파이프라인-실행)
- [9. CLI 명령어 레퍼런스](#9-cli-명령어-레퍼런스)
- [10. 출력 파일 구조](#10-출력-파일-구조)
- [11. 활용 예시](#11-활용-예시)
- [12. 트러블슈팅](#12-트러블슈팅)
- [13. 프로젝트 구조](#13-프로젝트-구조)
- [14. 관련 문서](#14-관련-문서)
- [15. 라이선스](#15-라이선스)


---

## 1. 소개

**slm-factory**는 도메인 문서를 학습하여 특화된 소형 언어모델(SLM)을 자동 생성하는 Teacher-Student 지식 증류 프레임워크입니다.

### 지식 증류(Knowledge Distillation)란?

지식 증류는 대형 언어모델(Teacher)의 지식을 소형 모델(Student)에게 전달하는 기술입니다. slm-factory는 이 과정을 다음과 같이 자동화합니다:

1. **Teacher 모델의 역할**: 도메인 문서를 읽고 질문에 대한 답변을 생성합니다. 예를 들어, Qwen3 8B와 같은 대형 모델이 정책 문서를 읽고 "이 정책의 주요 목적은 무엇인가?"라는 질문에 상세한 답변을 제공합니다.

2. **Student 모델의 학습**: Teacher가 생성한 질문-답변 쌍을 학습 데이터로 활용하여 소형 모델(예: Gemma 1B)을 파인튜닝합니다. Student 모델은 Teacher의 답변 패턴과 도메인 지식을 학습하게 됩니다.

3. **결과물**: 원본 문서의 도메인 지식을 내재화한 경량 모델이 생성됩니다. 이 모델은 Teacher보다 훨씬 작지만, 특정 도메인에서는 유사한 수준의 성능을 발휘할 수 있습니다.

### 왜 소형 언어모델(SLM)인가?

- **비용 효율성**: 대형 모델 대비 추론 비용이 10배 이상 저렴합니다
- **속도**: 응답 생성 속도가 빠르고 실시간 서비스에 적합합니다
- **프라이버시**: 로컬 환경에서 실행 가능하여 민감한 도메인 데이터를 외부로 전송하지 않습니다
- **도메인 특화**: 특정 분야에 집중하여 범용 모델보다 높은 정확도를 달성할 수 있습니다

### slm-factory의 자동화

slm-factory는 "도메인 문서 → 파인튜닝된 SLM" 전환 과정을 완전히 자동화합니다. 사용자는 문서를 제공하고 설정 파일을 편집하는 것만으로, 문서 파싱부터 QA 생성, 검증, 학습, 배포까지 전체 파이프라인을 한 번의 명령으로 실행할 수 있습니다.

---

## 2. 파이프라인 흐름도

```
┌─────────────┐      ┌──────────────┐      ┌──────────────┐      ┌──────────────┐
  Documents    ───▶    Step 1        ───▶    Step 2        ───▶    Step 3
  PDF/HWPX/            Parse                 Generate              Validate
  HTML/TXT             문서 파싱             QA 쌍 생성            QA 검증
└─────────────┘      └──────────────┘      └──────────────┘      └──────────────┘
                              ▼                      ▼                      ▼
                       ParsedDocument        QA Pairs (Alpaca)    Filtered QA Pairs

┌──────────────┐      ┌──────────────┐      ┌──────────────┐
  Step 3a       ───▶    Step 3b       ───▶    Step 3c
  Score                 Augment               Analyze
  품질 평가             데이터 증강           통계 분석
└──────────────┘      └──────────────┘      └──────────────┘
         ▼                      ▼                      ▼
  Scored QA Pairs       Augmented Pairs       data_analysis.json

┌──────────────┐      ┌──────────────┐      ┌──────────────┐
  Step 4        ───▶    Step 5        ───▶    Step 6
  Convert               Train                 Export
  채팅 템플릿           LoRA 학습             모델 배포
└──────────────┘      └──────────────┘      └──────────────┘
         ▼                      ▼                      ▼
  training_data.jsonl    LoRA Adapter     Merged Model + Ollama Modelfile
```

### 단계별 입출력

1. **parse**: PDF/HWPX/HTML/TXT → `ParsedDocument`
2. **generate**: `ParsedDocument` + Teacher LLM → QA 쌍
3. **validate**: QA 쌍 → 필터링된 QA 쌍 (규칙 + 임베딩 검증)
4. **score**: QA 쌍 → 품질 점수 평가된 QA 쌍 (Teacher LLM 1~5점 평가, threshold 필터링)
5. **augment**: QA 쌍 → 증강된 QA 쌍 (Teacher LLM 질문 패러프레이즈)
6. **analyze**: QA 쌍 → 분석 보고서 (통계 분석, 데이터 품질 경고)
7. **convert**: Alpaca JSON → 채팅 템플릿 JSONL
8. **train**: 채팅 데이터 → LoRA 어댑터
9. **export**: LoRA 어댑터 → 병합 모델 + Ollama Modelfile

---

## 3. 주요 기능

- **다중 형식 파싱**: PDF, HWPX(한글), HTML, TXT/MD, DOCX(Word) 문서를 자동으로 파싱하여 텍스트와 표를 추출합니다
- **유연한 Teacher LLM 백엔드**: Ollama(로컬 실행) 또는 OpenAI 호환 API를 Teacher 모델로 사용할 수 있습니다
- **규칙 기반 + 임베딩 기반 QA 검증**: 생성된 QA 쌍의 품질을 자동으로 검증하고 필터링합니다
- **Teacher LLM 기반 품질 점수 평가**: 생성된 QA 쌍을 Teacher LLM이 1~5점으로 평가하여 저품질 데이터를 자동으로 필터링합니다
- **질문 패러프레이즈 데이터 증강**: Teacher LLM을 활용하여 질문을 다양한 표현으로 변형하여 학습 데이터를 증강합니다
- **자동 데이터 분석 보고서**: 카테고리 분포, 길이 통계, 데이터 불균형 경고 등을 포함한 분석 보고서를 자동 생성합니다
- **자동 채팅 템플릿 변환**: HuggingFace의 모든 대화형 모델을 지원하며, 각 모델의 채팅 템플릿을 자동으로 적용합니다
- **LoRA 파인튜닝 + 조기 종료**: 효율적인 LoRA 학습과 조기 종료 기능으로 과적합을 방지합니다
- **원클릭 Ollama 배포**: 학습된 모델을 Ollama Modelfile로 자동 변환하여 즉시 배포할 수 있습니다
- **DOCX(Word) 파싱 지원**: python-docx를 사용하여 Word 문서의 텍스트, 표, 메타데이터를 자동 추출합니다
- **설정 검증 명령어**: `slm-factory check` 명령으로 설정 파일, 문서 디렉토리, Ollama 연결을 사전 점검합니다
- **파이프라인 재개**: `--resume` 옵션으로 중간 저장 파일에서 중단된 단계부터 재실행할 수 있습니다
- **Rich 진행률 표시**: QA 생성, 품질 평가, 데이터 증강 시 실시간 진행 바를 표시합니다
- **모듈 직접 실행**: `python -m slm_factory`로 패키지를 직접 실행할 수 있습니다
- **편의 CLI 도구**: `status`로 진행 상태 확인, `clean`으로 중간 파일 정리, `convert`/`export` 단독 실행, `--verbose`/`--quiet` 로그 레벨 조절을 지원합니다
- **대화형 파이프라인 (권장)**: `wizard` 명령 하나로 문서 선택부터 모델 배포까지 단계별로 안내합니다. 처음 사용자는 wizard만 실행하면 됩니다
- **자동 모델 평가**: 학습된 모델을 BLEU/ROUGE 메트릭으로 자동 평가합니다
- **모델 비교 (Before/After)**: Base 모델과 Fine-tuned 모델의 답변을 나란히 비교합니다
- **GGUF 변환**: llama.cpp 호환 GGUF 양자화 형식으로 모델을 변환합니다
- **증분 학습**: 문서 추가 시 기존 QA를 유지하면서 새 문서만 처리합니다
- **멀티턴 대화 생성**: QA 쌍을 멀티턴 대화 데이터로 확장합니다
- **QA 수동 리뷰 (TUI)**: 생성된 QA 쌍을 TUI에서 승인/거부/편집합니다
- **파이프라인 대시보드 (TUI)**: 파이프라인 진행 상태를 실시간 TUI로 모니터링합니다

---

## 4. 기술 스택

| 카테고리 | 패키지 | 버전 | 역할 |
|---------|--------|------|------|
| **Core** | typer | >=0.9.0 | CLI 프레임워크 |
| | pydantic | >=2.0 | 설정 검증 |
| | pyyaml | >=6.0 | YAML 파싱 |
| | rich | >=13.0 | 터미널 출력 |
| | httpx | >=0.25.0 | HTTP 클라이언트 |
| **Parsing** | pymupdf | >=1.24.0 | PDF 파싱 |
| | beautifulsoup4 | >=4.12 | HTML 파싱 |
| | lxml | >=5.0 | XML 파싱 (HWPX) |
| **ML/Training** | torch | >=2.1.0 | 딥러닝 프레임워크 |
| | transformers | >=4.40.0 | 모델 로딩 및 학습 |
| | datasets | >=2.18.0 | 데이터셋 관리 |
| | peft | >=0.10.0 | LoRA 어댑터 |
| | trl | >=0.8.0 | SFTTrainer |
| | accelerate | >=0.28.0 | 분산 학습 |
| | bitsandbytes | >=0.43.0 | 양자화 |
| **Evaluation** | evaluate | >=0.4.0 | BLEU/ROUGE 메트릭 |
| **TUI** | textual | >=0.40.0 | TUI 프레임워크 |
| **Optional** | pykospacing | - | 한국어 띄어쓰기 교정 |
| | sentence-transformers | >=2.6.0 | 의미적 유사도 검증 |
| | pdfplumber | >=0.11.0 | 대체 PDF 파서 |
| | python-docx | - | DOCX(Word) 파싱 |
| | pytest | >=8.0 | 테스트 프레임워크 |

---

## 5. 시스템 요구사항

- **Python**: 3.11 이상
- **GPU**: CUDA 지원 GPU 권장 (VRAM 8GB 이상, CPU 학습도 가능하나 매우 느림)
- **Ollama**: Teacher LLM으로 Ollama를 사용하는 경우 설치 필요 ([ollama.com](https://ollama.com))
- **디스크 공간**: 약 5GB 이상 (모델 다운로드 및 학습 체크포인트 저장)

---

## 6. 설치

```bash
git clone https://github.com/DevDnA/slm-factory.git
cd slm-factory
pip install -e ".[all]"
slm-factory --install-completion
```

모든 기능이 포함됩니다: PDF/HTML/TXT/HWPX/DOCX 파싱, 한국어 띄어쓰기 교정, 임베딩 기반 검증, 테스트 도구, Shell 자동완성.

### Shell 자동완성

Tab 키로 명령어와 옵션을 자동완성할 수 있습니다:

```bash
# 자동완성 설치 (bash/zsh/fish/powershell 자동 감지)
slm-factory --install-completion

# 셸 재시작 후 적용됩니다
# 또는 즉시 적용:
source ~/.bashrc   # bash
source ~/.zshrc    # zsh
```

자동완성이 작동하지 않으면:

```bash
# 현재 셸의 자동완성 스크립트 확인
slm-factory --show-completion
```

---

## 7. 빠른 시작 (wizard)

3단계로 첫 번째 도메인 특화 모델을 생성할 수 있습니다:

```bash
# 1. 프로젝트 생성
slm-factory init --name my-project

# 2. 학습할 문서를 넣기
cp /path/to/your/documents/*.pdf my-project/documents/

# 3. wizard 실행 — 이후는 안내에 따라 진행
slm-factory wizard --config my-project/project.yaml
```

wizard가 다음을 순서대로 안내합니다:

1. 설정 파일 확인
2. 문서 선택 (전체 또는 개별)
3. 문서 파싱
4. QA 쌍 생성 (확인 후 진행)
5. QA 검증
6. 품질 점수 평가 (선택)
7. 데이터 증강 (선택)
8. LoRA 학습 (확인 후 진행)
9. 모델 내보내기 (확인 후 진행)
10. 멀티턴 대화 생성 (선택)
11. GGUF 변환 (선택)
12. 모델 평가 (선택)

각 단계에서 건너뛰기를 선택하면, 나중에 실행할 명령어를 알려줍니다.

> **사전 준비**: wizard 실행 전 Ollama 서버와 Teacher 모델이 필요합니다:
> ```bash
> ollama serve          # 별도 터미널에서 실행
> ollama pull qwen3:8b  # Teacher 모델 다운로드
> ```

완료 후 생성된 모델을 테스트합니다:

```bash
cd my-project/output/merged_model
ollama create my-project-model -f Modelfile
ollama run my-project-model
```

---

## 8. 고급: 수동 파이프라인 실행

wizard 없이 각 단계를 직접 제어하려면 `run` 명령을 사용합니다.

### 설정 편집

`my-project/project.yaml` 파일을 열어 필요한 설정을 수정합니다:

```yaml
teacher:
  model: "qwen3:8b"              # Teacher 모델 선택
  
student:
  model: "google/gemma-3-1b-it"  # Student 모델 선택

export:
  ollama:
    model_name: "my-project-model"  # 배포할 모델 이름
```

### 전체 파이프라인 한 번에 실행

```bash
slm-factory run --config my-project/project.yaml
```

### 단계별 실행

개별 단계만 실행할 수도 있습니다:

```bash
slm-factory parse --config my-project/project.yaml      # 문서 파싱만
slm-factory generate --config my-project/project.yaml    # 파싱 + QA 생성
slm-factory validate --config my-project/project.yaml    # + 검증
slm-factory train --config my-project/project.yaml       # 학습
slm-factory export --config my-project/project.yaml      # 모델 내보내기
```

> **Tip**: `slm-factory check --config my-project/project.yaml`으로 설정과 환경을 사전 점검할 수 있습니다.

---

## 9. CLI 명령어 레퍼런스

### CLI 명령어 레퍼런스

모든 명령어에 `--verbose` (`-v`) 또는 `--quiet` (`-q`) 전역 옵션을 사용할 수 있습니다.

```
┌─────────────────────────────────────────────────────────────────────────┐
  Options
  --verbose  -v         디버그 로그를 표시합니다
  --quiet    -q         경고와 에러만 표시합니다
└─────────────────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────────────┐
  시작하기
  wizard                대화형 파이프라인 (권장)
  init                  새 프로젝트 초기화
  check                 설정 및 환경 사전 점검
└─────────────────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────────────┐
  파이프라인
  run                   전체 파이프라인 실행
  parse                 문서 파싱만 실행
  generate              파싱 + QA 생성
  validate              파싱 + 생성 + 검증
  score                 파싱 + 생성 + 검증 + 품질 평가
  augment               파싱 + 생성 + 검증 + 평가 + 증강
  analyze               파싱 + 생성 + 검증 + 평가 + 증강 + 분석
  train                 학습 단계 실행
  convert               QA 데이터를 훈련용 JSONL로 변환
  export                훈련된 모델 내보내기 (LoRA 병합)
  update                변경된 문서만 증분 처리
  generate-dialogue     멀티턴 대화 데이터 생성
  export-gguf           GGUF 양자화 형식으로 변환
└─────────────────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────────────┐
  평가
  eval                  학습된 모델 평가 (BLEU/ROUGE)
  compare               Base vs Fine-tuned 모델 비교
└─────────────────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────────────┐
  유틸리티
  status                파이프라인 진행 상태 확인
  clean                 중간 생성 파일 정리
  review                QA 수동 리뷰 TUI
  dashboard             파이프라인 대시보드 TUI
  version               버전 정보 출력
└─────────────────────────────────────────────────────────────────────────┘
```

---

### `wizard` - 대화형 파이프라인 (권장)

전체 파이프라인을 단계별로 안내하며 대화형으로 실행합니다. 각 단계에서 사용자의 확인을 받고, 선택적 단계(품질 평가, 데이터 증강)는 건너뛸 수 있습니다.

**사용법**:
```bash
slm-factory wizard [--config <설정파일경로>] [--resume]
```

**옵션**:
- `--config` (선택, 기본값: `project.yaml`): 프로젝트 설정 파일 경로
- `--resume` / `-r` (선택): 이전 실행의 중간 결과에서 재개합니다

**진행 단계**:
1. 설정 파일 로드 (자동 탐색 또는 직접 입력)
2. 문서 선택 (전체 또는 개별 선택)
3. 문서 파싱 (자동 진행)
4. QA 쌍 생성 (확인 후 진행)
5. QA 검증 (자동 진행)
6. 품질 점수 평가 (선택)
7. 데이터 증강 (선택)
8. LoRA 학습 (확인 후 진행)
9. 모델 내보내기 (확인 후 진행)
10. 멀티턴 대화 생성 (선택)
11. GGUF 변환 (선택)
12. 모델 평가 (선택)

각 단계에서 "건너뜀"을 선택하면 해당 단계의 결과물 경로와 나중에 실행할 명령어를 안내합니다.

---

### `init` - 프로젝트 초기화

새로운 slm-factory 프로젝트를 생성합니다.

**사용법**:
```bash
slm-factory init --name <프로젝트명> [--path <부모디렉토리>]
```

**옵션**:
- `--name` (필수): 생성할 프로젝트 이름
- `--path` (선택, 기본값: `.`): 프로젝트를 생성할 부모 디렉토리

**예시**:
```bash
slm-factory init --name policy-assistant
```

**출력**:
```
✓ 프로젝트 'policy-assistant'가 생성되었습니다: ./policy-assistant

프로젝트 구조:
  ./policy-assistant/
  ./policy-assistant/documents/
  ./policy-assistant/output/
  ./policy-assistant/project.yaml

사전 준비:
  1. ./policy-assistant/documents 디렉토리에 학습할 문서(PDF, TXT 등)를 추가하세요
  2. 별도 터미널에서 Ollama를 실행하세요: ollama serve
  3. Teacher 모델을 다운로드하세요: ollama pull qwen3:8b

실행:
  4. 환경 점검: slm-factory check --config ./policy-assistant/project.yaml
  5. wizard 실행: slm-factory wizard --config ./policy-assistant/project.yaml
```

기존 프로젝트에 `project.yaml`이 있으면 덮어쓰기 확인 프롬프트가 표시됩니다.

---

### `run` - 전체 파이프라인 실행

문서 파싱부터 모델 배포까지 전체 파이프라인을 순차적으로 실행합니다.

**사용법**:
```bash
slm-factory run [--config <설정파일경로>] [--resume]
```

**옵션**:
- `--config` (선택, 기본값: `project.yaml`): 프로젝트 설정 파일 경로
- `--resume` / `-r` (선택): 중간 저장 파일에서 재개합니다

**예시**:
```bash
slm-factory run --config ./my-project/project.yaml

# 중단된 지점에서 재개
slm-factory run --config ./my-project/project.yaml --resume
```

**실행 단계**:
1. 문서 파싱 (parse)
2. QA 쌍 생성 (generate)
3. QA 검증 (validate)
4. 품질 점수 평가 (score) — scoring.enabled 시
5. 데이터 증강 (augment) — augment.enabled 시
6. 데이터 분석 (analyze) — analyzer.enabled 시
7. 학습 데이터 변환 (convert)
8. LoRA 학습 (train)
9. 모델 병합 및 배포 (export)

---

### `parse` - 문서 파싱

문서 파싱 단계만 실행하여 결과를 확인합니다.

**사용법**:
```bash
slm-factory parse [--config <설정파일경로>]
```

**예시**:
```bash
slm-factory parse --config project.yaml
```

**출력**:
```
Parsed 5 documents
```

**생성 파일**: `output/parsed_documents.json` (디버깅 및 재개용)

---

### `generate` - QA 생성

문서 파싱 후 Teacher LLM을 사용하여 QA 쌍을 생성합니다.

**사용법**:
```bash
slm-factory generate [--config <설정파일경로>]
```

**예시**:
```bash
slm-factory generate --config project.yaml
```

**출력**:
```
Generated 150 QA pairs from 5 documents
```

**생성 파일**: `output/qa_alpaca.json` (Alpaca 형식 QA 쌍)

---

### `validate` - QA 검증

파싱, 생성 후 QA 쌍을 검증하고 필터링합니다.

**사용법**:
```bash
slm-factory validate [--config <설정파일경로>]
```

**예시**:
```bash
slm-factory validate --config project.yaml
```

**출력**:
```
Validation complete: 142 accepted, 8 rejected (out of 150 generated)
```

검증 기준:
- 최소/최대 답변 길이
- 빈 답변 제거
- 중복 제거
- 거부 패턴 매칭 (예: "I don't know")
- 선택적 의미적 groundedness 체크

---

### `score` - 품질 점수 평가

문서 파싱, QA 생성, 검증 후 Teacher LLM을 사용하여 QA 쌍의 품질을 1~5점으로 평가합니다.

**사용법**:
```bash
slm-factory score [--config <설정파일경로>] [--resume]
```

**옵션**:
- `--config` (선택, 기본값: `project.yaml`): 프로젝트 설정 파일 경로
- `--resume` / `-r` (선택): 중간 저장 파일에서 재개합니다

**예시**:
```bash
slm-factory score --config project.yaml
```

**출력**:
```
Score complete: 120 passed, 22 filtered (out of 142 validated)
```

Note: scoring.enabled가 false(기본값)이면 점수 평가를 건너뜁니다. project.yaml에서 scoring.enabled: true로 설정하십시오.

---

### `augment` - 데이터 증강

점수 평가까지 완료된 QA 쌍을 Teacher LLM으로 질문 패러프레이즈하여 증강합니다.

**사용법**:
```bash
slm-factory augment [--config <설정파일경로>] [--resume]
```

**옵션**:
- `--config` (선택, 기본값: `project.yaml`): 프로젝트 설정 파일 경로
- `--resume` / `-r` (선택): 중간 저장 파일에서 재개합니다

**예시**:
```bash
slm-factory augment --config project.yaml
```

**출력**:
```
Augmentation complete: 120 → 360 (240 augmented pairs added)
```

Note: augment.enabled가 false(기본값)이면 증강을 건너뜁니다.

---

### `analyze` - 데이터 분석

전체 전처리 파이프라인 실행 후 데이터 통계 분석을 수행합니다.

**사용법**:
```bash
slm-factory analyze [--config <설정파일경로>] [--resume]
```

**옵션**:
- `--config` (선택, 기본값: `project.yaml`): 프로젝트 설정 파일 경로
- `--resume` / `-r` (선택): 중간 저장 파일에서 재개합니다

**예시**:
```bash
slm-factory analyze --config project.yaml
```

**출력**: 분석 요약 테이블 (Rich 콘솔) + data_analysis.json 보고서

---

### `train` - 학습

LoRA 파인튜닝을 실행합니다. 기존 학습 데이터를 사용하거나 전체 파이프라인을 실행할 수 있습니다.

**사용법**:
```bash
slm-factory train [--config <설정파일경로>] [--data <학습데이터경로>] [--resume]
```

**옵션**:
- `--config` (선택, 기본값: `project.yaml`): 프로젝트 설정 파일
- `--data` (선택): 기존 `training_data.jsonl` 파일 경로. 지정하지 않으면 전체 파이프라인 실행
- `--resume` / `-r` (선택): 중간 저장 파일에서 재개합니다

**예시 1**: 전체 파이프라인 실행 후 학습
```bash
slm-factory train --config project.yaml
```

**예시 2**: 기존 학습 데이터로 학습만 실행
```bash
slm-factory train --config project.yaml --data ./custom_qa_data.jsonl
```

**출력**:
```
Training complete! Adapter saved to: ./output/checkpoints/adapter
```

---

### `status` - 진행 상태 확인

파이프라인의 진행 상태를 확인합니다. 각 단계의 중간 파일 존재 여부와 건수를 표시합니다.

**사용법**: `slm-factory status [--config <설정파일경로>]`

**출력 예시**:
```
┌──────────────────────────────────────────────────────┐
  단계       파일                    상태   건수

  parse      parsed_documents.json   존재   5개 문서
  generate   qa_alpaca.json          존재   150개 쌍
  score      qa_scored.json          없음   -
  ...        ...                     ...    ...
└──────────────────────────────────────────────────────┘
  다음 --resume 실행 시 validate부터 재개됩니다
```

---

### `clean` - 파일 정리

중간 생성 파일을 정리합니다.

**사용법**: `slm-factory clean [--config <설정파일경로>] [--all]`

**옵션**:
- `--all`: 모든 출력 파일 삭제 (기본: 중간 파일만 — qa_scored.json, qa_augmented.json, data_analysis.json)

삭제 전 확인 프롬프트가 표시됩니다.

---

### `convert` - 데이터 변환

QA 데이터를 훈련용 JSONL 형식으로 변환합니다. 파이프라인 전체를 재실행하지 않고 변환만 수행합니다.

**사용법**: `slm-factory convert [--config <설정파일경로>] [--data <QA데이터경로>]`

**옵션**:
- `--data`: QA 데이터 파일 경로 (미지정 시 output에서 자동 감지: qa_augmented.json → qa_scored.json → qa_alpaca.json)

---

### `export` - 모델 내보내기

훈련된 모델을 내보냅니다 (LoRA 병합 + Ollama Modelfile). 파이프라인 전체를 재실행하지 않고 내보내기만 수행합니다.

**사용법**: `slm-factory export [--config <설정파일경로>] [--adapter <어댑터경로>]`

**옵션**:
- `--adapter`: 어댑터 디렉토리 경로 (미지정 시 output/checkpoints/adapter/ 사용)

---

### `check` - 설정 검증

프로젝트 설정과 실행 환경을 사전 점검합니다.

**사용법**: `slm-factory check --config project.yaml`

점검 항목:
- 설정 파일 로드 및 Pydantic 검증
- 문서 디렉토리 존재 및 파일 유무
- 출력 디렉토리 쓰기 권한
- Ollama 서버 연결 (backend=ollama일 때)
- Teacher 모델 사용 가능 여부

모든 항목 통과 시 "모든 점검 통과!" 메시지와 wizard 실행 명령을 안내하며 종료 코드 0을 반환합니다.

실패 항목이 있으면 일반적인 해결 방법(문서 추가, Ollama 실행, 모델 다운로드)을 안내하며 종료 코드 1을 반환합니다.

---

### `eval` - 모델 평가

학습된 모델을 BLEU/ROUGE 메트릭으로 평가합니다.

**사용법**:
```bash
slm-factory eval --model <모델이름> [--config <설정파일>] [--data <QA데이터>]
```

**옵션**:
- `--model` (필수): 평가할 Ollama 모델 이름
- `--config` (선택, 기본값: `project.yaml`): 프로젝트 설정 파일
- `--data` (선택): QA 데이터 파일 경로 (미지정 시 자동 감지)

**예시**:
```bash
slm-factory eval --model my-project-model --config project.yaml
```

**출력**: `output/eval_results.json` (BLEU/ROUGE 점수)

---

### `compare` - 모델 비교

Base 모델과 Fine-tuned 모델의 답변을 나란히 비교합니다.

**사용법**:
```bash
slm-factory compare --base-model <기준모델> --finetuned-model <파인튜닝모델> [--config <설정파일>] [--data <QA데이터>]
```

**옵션**:
- `--base-model` (필수): 비교 기준 모델 (Ollama)
- `--finetuned-model` (필수): 파인튜닝된 모델 (Ollama)
- `--config` (선택, 기본값: `project.yaml`): 프로젝트 설정 파일
- `--data` (선택): QA 데이터 파일 경로

**예시**:
```bash
slm-factory compare --base-model gemma:2b --finetuned-model my-project-model --config project.yaml
```

**출력**: `output/compare_results.json` (나란히 비교 결과)

---

### `export-gguf` - GGUF 변환

병합된 모델을 llama.cpp 호환 GGUF 양자화 형식으로 변환합니다.

**사용법**:
```bash
slm-factory export-gguf [--config <설정파일>] [--model-dir <모델경로>]
```

**옵션**:
- `--config` (선택, 기본값: `project.yaml`): 프로젝트 설정 파일
- `--model-dir` (선택, 기본값: `output/merged_model`): 병합된 모델 디렉토리

**예시**:
```bash
slm-factory export-gguf --config project.yaml
```

llama.cpp의 convert 스크립트를 사용하여 GGUF 양자화 형식으로 변환합니다.

---

### `update` - 증분 업데이트

변경된 문서만 감지하여 새 QA를 생성하고 기존 QA와 병합합니다.

**사용법**:
```bash
slm-factory update [--config <설정파일>]
```

**옵션**:
- `--config` (선택, 기본값: `project.yaml`): 프로젝트 설정 파일

**예시**:
```bash
slm-factory update --config project.yaml
```

해시 기반으로 변경 감지하여 기존 QA를 유지하면서 새 문서만 처리합니다.

---

### `generate-dialogue` - 대화 생성

QA 쌍을 멀티턴 대화 형식으로 확장합니다.

**사용법**:
```bash
slm-factory generate-dialogue [--config <설정파일>] [--data <QA데이터>]
```

**옵션**:
- `--config` (선택, 기본값: `project.yaml`): 프로젝트 설정 파일
- `--data` (선택): QA 데이터 파일 경로

**예시**:
```bash
slm-factory generate-dialogue --config project.yaml
```

**출력**: `output/dialogues.json` (멀티턴 대화 데이터)

---

### `review` - QA 리뷰

TUI에서 QA 쌍을 하나씩 확인하며 승인/거부/편집합니다.

**사용법**:
```bash
slm-factory review [--config <설정파일>] [--data <QA데이터>]
```

**옵션**:
- `--config` (선택, 기본값: `project.yaml`): 프로젝트 설정 파일
- `--data` (선택): QA 데이터 파일 경로

**예시**:
```bash
slm-factory review --config project.yaml
```

**출력**: `output/qa_reviewed.json` (수동 리뷰된 QA 쌍)

---

### `dashboard` - 대시보드

파이프라인 진행 상태를 실시간 TUI로 모니터링합니다.

**사용법**:
```bash
slm-factory dashboard [--config <설정파일>]
```

**옵션**:
- `--config` (선택, 기본값: `project.yaml`): 프로젝트 설정 파일

**예시**:
```bash
slm-factory dashboard --config project.yaml
```

각 단계의 파일 존재 여부, 건수, 최근 수정 시각을 표시합니다.

---

### `version` - 버전 정보

slm-factory의 현재 버전을 출력합니다.

**사용법**:
```bash
slm-factory version
```

**출력**:
```
slm-factory 0.1.0
```

---

## 10. 출력 파일 구조

파이프라인 실행 후 `output/` 디렉토리에 다음 파일들이 생성됩니다:

```
output/
├── parsed_documents.json       # 파싱된 문서 (디버깅 및 재개용)
├── qa_alpaca.json             # 생성된 QA 쌍 (Alpaca 형식)
├── qa_scored.json             # 점수 평가된 QA 쌍 (재개용)
├── qa_augmented.json          # 증강된 QA 쌍 (재개용)
├── qa_reviewed.json           # 수동 리뷰된 QA 쌍
├── data_analysis.json         # 데이터 분석 보고서
├── dialogues.json             # 멀티턴 대화 데이터
├── eval_results.json          # 모델 평가 결과 (BLEU/ROUGE)
├── compare_results.json       # 모델 비교 결과 (Before/After)
├── training_data.jsonl        # 채팅 템플릿 적용된 학습 데이터
├── checkpoints/
    └── adapter/               # LoRA 어댑터 가중치
        ├── adapter_config.json
        ├── adapter_model.safetensors
        └── ...
└── merged_model/              # 병합된 최종 모델
    ├── config.json
    ├── model.safetensors
    ├── tokenizer.json
    ├── tokenizer_config.json
    └── Modelfile              # Ollama 배포 파일
```

### 파일 설명

- **`parsed_documents.json`**: 원본 문서에서 추출한 텍스트, 표, 메타데이터를 JSON 형식으로 저장합니다. 파이프라인 재개 시 파싱 단계를 건너뛸 수 있습니다.

- **`qa_alpaca.json`**: Teacher LLM이 생성한 질문-답변 쌍을 Alpaca 형식으로 저장합니다. 각 항목은 `instruction`, `input`, `output` 필드를 포함합니다.

- **`qa_scored.json`**: 품질 점수 평가를 통과한 QA 쌍입니다. `--resume` 옵션으로 augment 단계부터 재개할 수 있습니다.

- **`qa_augmented.json`**: 데이터 증강이 완료된 QA 쌍입니다. `--resume` 옵션으로 analyze 단계부터 재개할 수 있습니다.

- **`qa_reviewed.json`**: TUI에서 수동 리뷰를 거친 QA 쌍입니다. 승인된 QA만 포함되며, 편집된 내용이 반영됩니다.

- **`data_analysis.json`**: QA 데이터의 통계 분석 보고서입니다. 카테고리 분포, 문서별 분포, 답변/질문 길이 통계, 데이터 품질 경고를 포함합니다.

- **`dialogues.json`**: QA 쌍을 멀티턴 대화 형식으로 확장한 데이터입니다. 대화형 모델 학습에 활용할 수 있습니다.

- **`eval_results.json`**: 학습된 모델의 BLEU/ROUGE 점수 평가 결과입니다. 각 메트릭별 점수와 평균 점수를 포함합니다.

- **`compare_results.json`**: Base 모델과 Fine-tuned 모델의 답변을 나란히 비교한 결과입니다. 각 질문에 대한 두 모델의 답변을 포함합니다.

- **`training_data.jsonl`**: Student 모델의 채팅 템플릿이 적용된 학습 데이터입니다. 각 줄은 `{"text": "..."}` 형식의 JSON 객체이며, `text` 필드에 채팅 템플릿이 적용된 전체 대화 문자열이 포함됩니다.

- **`checkpoints/adapter/`**: LoRA 학습 중 저장된 어댑터 가중치입니다. PEFT 형식으로 저장되며, 원본 모델과 결합하여 사용할 수 있습니다.

- **`merged_model/`**: LoRA 어댑터가 원본 Student 모델과 병합된 최종 모델입니다. HuggingFace 형식으로 저장되며, `Modelfile`을 통해 Ollama에 즉시 배포할 수 있습니다.

---

## 11. 활용 예시

### 예시 1: 한국어 정책 문서(HWPX) → 정책 전문 모델

한국 정부 정책 문서(HWPX 형식)를 학습하여 정책 질의응답 모델을 생성합니다.

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
  temperature: 0.3

questions:
  categories:
    policy_overview:
      - "이 정책의 주요 목적은 무엇입니까?"
      - "정책 대상자는 누구입니까?"
      - "정책의 시행 기간은 언제입니까?"
    policy_details:
      - "지원 내용과 규모는 어떻게 됩니까?"
      - "신청 자격 요건은 무엇입니까?"
      - "신청 절차는 어떻게 됩니까?"

student:
  model: "google/gemma-3-1b-it"

export:
  ollama:
    model_name: "policy-assistant-ko"
    system_prompt: "당신은 한국 정부 정책 전문 상담 도우미입니다."
```

**실행**:
```bash
# 파이프라인 실행
slm-factory run --config policy-project/project.yaml

# 모델 배포 및 테스트
cd policy-project/output/merged_model
ollama create policy-assistant-ko -f Modelfile
ollama run policy-assistant-ko
```

---

### 예시 2: 영문 기술 문서(PDF) → 기술 문서 전문 모델

소프트웨어 API 문서(PDF)를 학습하여 개발자 지원 모델을 생성합니다.

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
    extract_tables: true

teacher:
  backend: "ollama"
  model: "qwen3:8b"
  temperature: 0.2              # 기술 문서는 낮은 temperature 권장

questions:
  categories:
    api_basics:
      - "What is the purpose of this API?"
      - "What are the authentication requirements?"
      - "What is the base URL for API requests?"
    api_usage:
      - "What are the available endpoints?"
      - "What parameters does this endpoint accept?"
      - "What is the expected response format?"
      - "What are common error codes and their meanings?"
    examples:
      - "Provide a code example for this functionality."
      - "What are best practices for using this API?"

validation:
  enabled: true
  groundedness:
    enabled: true               # 의미적 검증 활성화
    threshold: 0.3

student:
  model: "google/gemma-3-1b-it"

training:
  num_epochs: 15
  learning_rate: 1.5e-5

export:
  ollama:
    model_name: "api-assistant"
    system_prompt: "You are a helpful API documentation assistant."
```

**실행**:
```bash
# 파이프라인 실행
slm-factory run --config tech-docs/project.yaml
```

---

### 예시 3: 기존 QA 데이터로 학습만 실행

이미 준비된 QA 데이터셋이 있는 경우, 파싱과 생성 단계를 건너뛰고 학습만 실행할 수 있습니다.

**QA 데이터 형식** (`custom_qa.jsonl`):
```jsonl
{"messages": [{"role": "user", "content": "What is Python?"}, {"role": "assistant", "content": "Python is a high-level programming language..."}]}
{"messages": [{"role": "user", "content": "How do I install packages?"}, {"role": "assistant", "content": "Use pip install <package-name>..."}]}
```

**실행**:
```bash
slm-factory train --config project.yaml --data ./custom_qa.jsonl
```

이 방식은 다음과 같은 경우에 유용합니다:
- 외부에서 준비한 QA 데이터셋을 사용하는 경우
- 여러 번 학습 하이퍼파라미터를 조정하며 실험하는 경우
- 파싱과 생성 단계가 이미 완료된 경우

---

## 12. 트러블슈팅

### 1. Ollama 연결 실패

**증상**:
```
Error: Failed to connect to Ollama at http://localhost:11434
```

**해결 방법**:
- Ollama 서버가 실행 중인지 확인하십시오:
  ```bash
  ollama serve
  ```
- `project.yaml`의 `teacher.api_base` 설정이 올바른지 확인하십시오:
  ```yaml
  teacher:
    api_base: "http://localhost:11434"  # Ollama 기본 포트
  ```
- 방화벽이나 네트워크 설정이 11434 포트를 차단하지 않는지 확인하십시오

---

### 2. GPU 메모리 부족 (CUDA Out of Memory)

**증상**:
```
RuntimeError: CUDA out of memory. Tried to allocate X.XX GiB
```

**해결 방법**:

**방법 1**: 양자화 활성화 (메모리 사용량 50% 감소)
```yaml
training:
  quantization:
    enabled: true
    bits: 4
```

**방법 2**: 배치 크기 감소
```yaml
training:
  batch_size: 2                    # 기본값 4에서 감소
  gradient_accumulation_steps: 8   # 총 배치 크기 유지
```

**방법 3**: 더 작은 Student 모델 사용
```yaml
student:
  model: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # 1B 모델
```

**방법 4**: CPU 학습 (느리지만 메모리 제약 없음)
```yaml
training:
  bf16: false                      # CPU는 bf16 미지원
```

---

### 3. HWPX 파싱 실패

**증상**:
```
Error: Failed to parse HWPX file - section0.xml not found
```

**해결 방법**:
- HWPX 파일이 손상되지 않았는지 확인하십시오 (한글에서 열어보기)
- HWPX 파일이 암호화되어 있지 않은지 확인하십시오
- 파일 확장자가 `.hwpx`인지 확인하십시오 (`.hwp`는 지원하지 않음)
- 최신 한글 버전에서 HWPX로 다시 저장해보십시오

---

### 4. pykospacing 설치 오류

**증상**:
```
ERROR: Could not install packages due to an OSError
```

**해결 방법**:

**방법 1**: Python 버전 확인 (3.11 이상 필요)
```bash
python --version
```

**방법 2**: Git이 설치되어 있는지 확인 (pykospacing은 Git 저장소에서 설치)
```bash
git --version
```

**방법 3**: 수동 설치 시도
```bash
pip install git+https://github.com/haven-jeon/PyKoSpacing.git
```

**방법 4**: 한국어 띄어쓰기 교정 비활성화
```yaml
parsing:
  hwpx:
    apply_spacing: false
```

---

### 5. 학습 데이터 부족 경고

**증상**:
```
Warning: Only 15 QA pairs generated. Recommend at least 100 for effective training.
```

**해결 방법**:

**방법 1**: 더 많은 문서 추가
```bash
cp /path/to/more/documents/*.pdf ./documents/
```

**방법 2**: 질문 카테고리 확장
```yaml
questions:
  categories:
    overview: [...]
    technical: [...]
    implementation: [...]
    additional:                    # 새 카테고리 추가
      - "What are the benefits?"
      - "What are the limitations?"
```

**방법 3**: Teacher 모델의 컨텍스트 크기 증가
```yaml
teacher:
  max_context_chars: 20000         # 기본값 12000에서 증가
```

---

### 6. 빈 QA 응답 생성

**증상**:
대부분의 QA 쌍이 "The document does not contain this information."으로 생성됨

**해결 방법**:

**방법 1**: Teacher 모델 타임아웃 증가
```yaml
teacher:
  timeout: 300                     # 기본값 180초에서 증가
```

**방법 2**: 다른 Teacher 모델 시도
```yaml
teacher:
  model: "llama3.1:8b"             # 또는 "mistral:7b"
```

**방법 3**: 질문을 문서 내용에 맞게 조정
- 문서를 먼저 읽고 실제로 답변 가능한 질문으로 수정하십시오
- 너무 일반적이거나 추상적인 질문은 피하십시오

**방법 4**: System prompt 조정
```yaml
questions:
  system_prompt: >
    Answer the question based on the document. 
    Provide detailed answers with specific information.
    If the exact answer is not in the document, provide related information.
```

---

## 13. 프로젝트 구조

```
slm-factory/
├── src/
    └── slm_factory/
        ├── __init__.py              # 패키지 초기화 및 버전 정보
        ├── __main__.py              # python -m slm_factory 진입점
        ├── cli.py                   # CLI 진입점 및 명령어 정의
        ├── config.py                # Pydantic 기반 설정 스키마
        ├── models.py                # 공유 데이터 모델 (QAPair, ParsedDocument)
        ├── pipeline.py              # 파이프라인 오케스트레이터
        ├── scorer.py                # QA 품질 점수 평가 (Teacher LLM)
        ├── augmenter.py             # QA 데이터 증강 (질문 패러프레이즈)
        ├── analyzer.py              # 학습 데이터 통계 분석
        ├── evaluator.py             # 모델 자동 평가 (BLEU/ROUGE)
        ├── comparator.py            # 모델 비교 (Before/After)
        ├── incremental.py           # 증분 학습 추적
        ├── converter.py             # 채팅 템플릿 변환기
        ├── utils.py                 # 유틸리티 및 로깅 설정
        ├── tui/
            ├── __init__.py          # TUI 패키지
            ├── widgets.py           # TUI 위젯 (QACard, StatusBar)
            ├── reviewer.py          # QA 수동 리뷰 TUI
            └── dashboard.py         # 파이프라인 대시보드 TUI
        ├── parsers/
            ├── __init__.py          # 파서 레지스트리
            ├── base.py              # 파서 기본 클래스
            ├── pdf.py               # PDF 파서 (PyMuPDF)
            ├── hwpx.py              # HWPX 파서 (한글 문서)
            ├── html.py              # HTML 파서 (BeautifulSoup)
            ├── text.py              # TXT/MD 파서
            └── docx.py              # DOCX 파서 (python-docx)
        ├── teacher/
            ├── __init__.py          # Teacher LLM 팩토리
            ├── base.py              # Teacher 기본 클래스
            ├── ollama.py            # Ollama 백엔드
            ├── openai_compat.py     # OpenAI 호환 API 백엔드
            ├── qa_generator.py      # QA 쌍 생성 로직
            └── dialogue_generator.py  # 멀티턴 대화 생성
        ├── validator/
            ├── __init__.py          # 검증 모듈 초기화
            ├── rules.py             # 규칙 기반 검증 (길이, 패턴 등)
            └── similarity.py        # 임베딩 기반 groundedness 체크
        ├── trainer/
            ├── __init__.py          # 학습 모듈 초기화
            └── lora_trainer.py      # LoRA 파인튜닝 (SFTTrainer, DataLoader 포함)
        └── exporter/
            ├── __init__.py          # 내보내기 모듈 초기화
            ├── hf_export.py         # HuggingFace 모델 병합
            ├── ollama_export.py     # Ollama Modelfile 생성
            └── gguf_export.py       # GGUF 양자화 변환
├── templates/
    └── project.yaml                 # 기본 프로젝트 템플릿
├── tests/
    └── __init__.py                  # 테스트 패키지
├── docs/
    ├── architecture.md              # 아키텍처 가이드
    ├── configuration.md             # 설정 레퍼런스
    └── modules.md                   # 모듈별 상세 문서
├── pyproject.toml                   # 프로젝트 메타데이터 및 의존성
└── README.md                        # 이 문서
```

---

## 14. 관련 문서

프로젝트에 대한 더 자세한 정보는 다음 문서를 참조하십시오:

- **[아키텍처 가이드](docs/architecture.md)**: slm-factory의 내부 구조와 설계 원칙을 설명합니다
- **[설정 레퍼런스](docs/configuration.md)**: `project.yaml`의 모든 설정 옵션에 대한 상세 설명을 제공합니다
- **[모듈별 상세 문서](docs/modules.md)**: 각 모듈(파서, Teacher, 검증기 등)의 API와 확장 방법을 안내합니다
- **[자주 묻는 질문 (FAQ)](docs/faq.md)**: Ollama 연결, GPU 메모리, QA 품질 등 자주 발생하는 문제의 해결 방법을 안내합니다

---

## 15. 라이선스

이 프로젝트의 라이선스는 추후 결정됩니다.

---

**slm-factory**로 도메인 특화 언어모델을 쉽고 빠르게 구축하십시오!
