# 빠른 참조 (Quick Reference)

> slm-factory의 핵심 명령어와 워크플로우를 한눈에 확인하세요. 도메인 문서를 SLM으로 변환하는 전체 파이프라인을 빠르게 참조하려는 활성 사용자를 위한 치트시트입니다.

---

## 설치

```bash
git clone https://github.com/DevDnA/slm-factory.git
cd slm-factory
python3 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[all]"
slm-factory --install-completion
```

---

## 사전 준비

Ollama를 Teacher LLM으로 사용하는 경우, wizard 실행 전 아래 두 명령을 먼저 실행합니다.

```bash
ollama serve           # 별도 터미널에서 실행 (백그라운드 유지)
ollama pull qwen3:8b   # Teacher 모델 다운로드 (최초 1회)
```

---

## 프로젝트 시작

```bash
# 1. 프로젝트 초기화
slm-factory init my-project

# 2. 학습할 문서 복사
cp /path/to/documents/*.pdf my-project/documents/

# 3. 환경 점검 (선택 권장)
slm-factory check --config my-project/project.yaml

# 4. 대화형 파이프라인 실행
slm-factory tool wizard --config my-project/project.yaml
```

---

## CLI 명령어 요약

모든 명령어에 `--verbose` (`-v`) 또는 `--quiet` (`-q`) 전역 옵션을 사용할 수 있습니다.

### 🚀 시작하기

| 명령어 | 설명 | 주요 옵션 |
|--------|------|-----------|
| `slm-factory init <name>` | 새 프로젝트 초기화 | `--path <디렉토리>` |
| `slm-factory check` | 설정 및 환경 사전 점검 | `--config <파일>` |

### ⚙️ 파이프라인

| 명령어 | 설명 | 주요 옵션 |
|--------|------|-----------|
| `slm-factory run` | 전체 파이프라인 실행 | `--config`, `--until <단계>`, `--resume` / `-r` |
| `slm-factory train` | LoRA 학습 실행 | `--config`, `--data <jsonl>`, `--resume` / `-r` |
| `slm-factory export` | 모델 내보내기 (LoRA 병합 + Modelfile) | `--config`, `--adapter <경로>` |

### 📊 평가

| 명령어 | 설명 | 주요 옵션 |
|--------|------|-----------|
| `slm-factory eval run` | BLEU/ROUGE 평가 | `--model <이름>`, `--config`, `--data` |
| `slm-factory eval compare` | Base vs Fine-tuned 비교 | `--base-model <이름>`, `--ft <이름>`, `--config`, `--data` |

### 🔧 도구

| 명령어 | 설명 | 주요 옵션 |
|--------|------|-----------|
| `slm-factory tool wizard` | 대화형 파이프라인 (권장) | `--config`, `--resume` / `-r` |
| `slm-factory tool review` | QA 수동 리뷰 TUI | `--config`, `--data` |
| `slm-factory tool dashboard` | 파이프라인 대시보드 TUI | `--config` |
| `slm-factory tool convert` | QA → JSONL 변환 | `--config`, `--data` |
| `slm-factory tool dialogue` | 멀티턴 대화 생성 | `--config`, `--data` |
| `slm-factory tool gguf` | GGUF 양자화 변환 | `--config`, `--model-dir` |
| `slm-factory tool update` | 증분 업데이트 (변경 문서만) | `--config` |

### ℹ️ 정보

| 명령어 | 설명 | 주요 옵션 |
|--------|------|-----------|
| `slm-factory status` | 파이프라인 진행 상태 확인 | `--config` |
| `slm-factory clean` | 중간 파일 정리 | `--config`, `--all` |
| `slm-factory version` | 버전 정보 출력 | |

> 각 명령어의 전체 옵션은 [CLI 레퍼런스](cli-reference.md)를 참조하십시오.

---

## 파이프라인 단계

`slm-factory run` 실행 시 아래 순서로 진행됩니다.

1. **parse** (필수) — PDF/HWPX/HTML/TXT/DOCX 파싱 → `output/parsed_documents.json`
2. **generate** (필수) — Teacher LLM으로 QA 쌍 생성 → `output/qa_alpaca.json`
3. **validate** (필수) — 규칙 + 임베딩 기반 QA 검증 및 필터링 → `qa_alpaca.json` 갱신
4. **score** (선택, `scoring.enabled: true`) — Teacher LLM 1~5점 품질 평가 → `output/qa_scored.json`
5. **augment** (선택, `augment.enabled: true`) — 질문 패러프레이즈 데이터 증강 → `output/qa_augmented.json`
6. **analyze** (선택, `analyzer.enabled: true`) — 통계 분석 보고서 생성 → `output/data_analysis.json`
7. **convert** (필수) — 채팅 템플릿 적용 JSONL 변환 → `output/training_data.jsonl`
8. **train** (필수) — LoRA 파인튜닝 → `output/checkpoints/adapter/`
9. **export** (필수) — 모델 병합 + Ollama Modelfile 생성 → `output/merged_model/`

---

## 자주 쓰는 워크플로우

### 1. 전체 자동 실행 (wizard)

처음 사용자에게 권장합니다. 각 단계를 확인하며 대화형으로 진행합니다.

```bash
slm-factory tool wizard --config my-project/project.yaml
```

### 2. 수동 전체 실행 (run)

설정을 직접 제어하고 싶을 때 사용합니다.

```bash
slm-factory run --config my-project/project.yaml
```

### 3. 단계별 실행 (run --until)

특정 단계까지만 실행하고 결과를 확인한 후 다음 단계로 진행합니다.

```bash
slm-factory run --until parse    --config my-project/project.yaml
slm-factory run --until generate --config my-project/project.yaml
slm-factory run --until validate --config my-project/project.yaml
slm-factory run --until score    --config my-project/project.yaml
slm-factory run --until augment  --config my-project/project.yaml
slm-factory train                --config my-project/project.yaml
slm-factory export               --config my-project/project.yaml
```

### 4. 기존 데이터로 학습만 (train --data)

이미 준비된 `training_data.jsonl`이 있거나 하이퍼파라미터를 반복 실험할 때 사용합니다.

```bash
slm-factory train --config my-project/project.yaml --data ./output/training_data.jsonl
```

### 5. 중단 후 재개 (--resume)

파이프라인이 중간에 중단된 경우, 중간 저장 파일에서 자동으로 재개합니다.

```bash
slm-factory run   --config my-project/project.yaml --resume
slm-factory train --config my-project/project.yaml --resume
slm-factory tool wizard --config my-project/project.yaml --resume
```

### 6. 평가 및 비교 (eval run / eval compare)

학습 완료 후 모델 성능을 측정하고 Base 모델과 비교합니다.

```bash
# BLEU/ROUGE 평가
slm-factory eval run --model my-project-model --config my-project/project.yaml

# Base vs Fine-tuned 비교
slm-factory eval compare \
  --base-model gemma:2b \
  --ft my-project-model \
  --config my-project/project.yaml
```

---

## 출력 파일 구조

```
output/
├── parsed_documents.json       # 파싱된 문서 텍스트 및 메타데이터
├── qa_alpaca.json              # Teacher LLM이 생성한 QA 쌍 (Alpaca 형식)
├── qa_scored.json              # 품질 점수 평가를 통과한 QA 쌍
├── qa_augmented.json           # 데이터 증강이 완료된 QA 쌍
├── qa_reviewed.json            # TUI 수동 리뷰를 거친 QA 쌍
├── data_analysis.json          # 카테고리 분포, 길이 통계, 품질 경고
├── dialogues.json              # 멀티턴 대화 형식으로 확장된 데이터
├── eval_results.json           # BLEU/ROUGE 평가 점수
├── compare_results.json        # Base vs Fine-tuned 비교 결과
├── training_data.jsonl         # 채팅 템플릿 적용된 최종 학습 데이터
├── checkpoints/
│   └── adapter/                # LoRA 어댑터 가중치 (PEFT 형식)
│       ├── adapter_config.json
│       └── adapter_model.safetensors
├── *.gguf                      # GGUF 양자화 변환 결과 (선택)
└── merged_model/               # 병합된 최종 모델 (HuggingFace 형식)
    ├── config.json
    ├── model.safetensors
    ├── tokenizer.json
    └── Modelfile               # Ollama 배포용 파일
```

---

## 설정 파일 골격

`project.yaml`의 최소 필수 설정입니다. 나머지는 기본값으로 동작합니다.

```yaml
project:
  name: "my-project"
  language: "ko"                        # "ko", "en", "ja" 등

paths:
  documents: "./documents"
  output: "./output"

teacher:
  backend: "ollama"                     # "ollama" 또는 "openai"
  model: "qwen3:8b"
  api_base: "http://localhost:11434"
  temperature: 0.3

student:
  model: "google/gemma-3-1b-it"

export:
  ollama:
    model_name: "my-project-model"
    system_prompt: "당신은 도메인 전문 도우미입니다."
```

> 전체 설정 옵션(scoring, augment, validation, training, LoRA 등)은 [설정 레퍼런스](configuration.md)를 참조하십시오.

---

## 트러블슈팅 빠른 해결

| 증상 | 원인 | 해결 |
|------|------|------|
| `Failed to connect to Ollama at http://localhost:11434` | Ollama 서버 미실행 | `ollama serve` 실행 후 재시도 |
| `CUDA out of memory` | GPU VRAM 부족 | `training.batch_size: 2`, `quantization.bits: 4` 설정 |
| `Failed to parse HWPX file - section0.xml not found` | HWPX 파일 손상 또는 `.hwp` 형식 | 한글에서 `.hwpx`로 다시 저장 (`.hwp` 미지원) |
| `Could not install packages due to an OSError` (pykospacing) | Python 버전 또는 Git 미설치 | `python --version` 확인 (3.11+), `git --version` 확인 |
| `Only N QA pairs generated. Recommend at least 100` | 문서 부족 또는 질문 카테고리 부족 | 문서 추가 또는 `questions.categories` 확장 |
| 대부분의 답변이 "The document does not contain..." | Teacher 모델 타임아웃 또는 질문 부적합 | `teacher.timeout: 300`, 질문을 문서 내용에 맞게 수정 |
| `model not found` (Ollama) | Teacher 모델 미다운로드 | `ollama pull qwen3:8b` |
| `externally-managed-environment` (pip) | 가상환경 미활성화 | `source .venv/bin/activate` 후 재설치 |

> 상세 해결 방법은 [사용 가이드](guide.md#7-트러블슈팅)를 참조하십시오.

---

## 관련 문서

| 문서 | 내용 | 대상 |
|------|------|------|
| [사용 가이드](guide.md) | 설치부터 배포까지 단계별 상세 안내 | 처음 사용자 |
| [CLI 레퍼런스](cli-reference.md) | 모든 명령어와 옵션의 완전한 설명 | 명령어 확인 |
| [설정 레퍼런스](configuration.md) | `project.yaml`의 모든 설정 옵션 상세 설명 | 설정 커스터마이징 |
| [아키텍처 가이드](architecture.md) | 내부 구조, 모듈 설계, 확장 방법 | 기여자, 고급 사용자 |
| [개발 가이드](development.md) | 개발 환경 설정, 테스트, 기여 방법 | 기여자 |
