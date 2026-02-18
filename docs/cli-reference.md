# CLI 명령어 레퍼런스

> slm-factory의 모든 명령어와 옵션을 정리한 공식 레퍼런스입니다.
> 빠른 조회는 [quick-reference.md](quick-reference.md)를, 단계별 사용법은 [guide.md](guide.md)를 참조하십시오.

---

## 전역 옵션

모든 명령어 앞에 사용할 수 있는 옵션입니다.

```
slm-factory [전역 옵션] <명령어> [옵션]
```

| 플래그 | 단축키 | 설명 |
|--------|--------|------|
| `--verbose` | `-v` | DEBUG 레벨 로그를 출력합니다. 문제 진단 시 사용합니다. |
| `--quiet` | `-q` | WARNING 이상의 로그만 출력합니다. 스크립트 자동화에 적합합니다. |
| `--help` | | 도움말을 출력합니다. |
| `--version` | | 버전 정보를 출력합니다. |
| `--install-completion` | | 현재 셸에 자동완성을 설치합니다. |
| `--show-completion` | | 자동완성 스크립트를 출력합니다. |

`--verbose`와 `--quiet`는 동시에 사용할 수 없습니다. 두 옵션 모두 지정하면 `--verbose`가 우선합니다.

---

## 명령어 목록

| 명령어 | 그룹 | 설명 |
|--------|------|------|
| `init` | 🚀 시작하기 | 새 프로젝트 디렉토리와 설정 파일을 생성합니다 |
| `check` | 🚀 시작하기 | 설정 파일, 문서 디렉토리, Ollama 연결을 사전 점검합니다 |
| `run` | ⚙️ 파이프라인 | 전체 파이프라인 또는 지정 단계까지 실행합니다 |
| `train` | ⚙️ 파이프라인 | LoRA 파인튜닝을 실행합니다 |
| `export` | ⚙️ 파이프라인 | 훈련된 어댑터를 병합하고 Ollama Modelfile을 생성합니다 |
| `eval run` | 📊 평가 | 학습된 모델을 BLEU/ROUGE 메트릭으로 평가합니다 |
| `eval compare` | 📊 평가 | Base 모델과 Fine-tuned 모델의 답변을 나란히 비교합니다 |
| `tool wizard` | 🔧 도구 | 대화형 파이프라인을 단계별로 안내합니다 (권장) |
| `tool review` | 🔧 도구 | QA 쌍을 TUI에서 수동으로 승인/거부/편집합니다 |
| `tool dashboard` | 🔧 도구 | 파이프라인 진행 상태를 실시간 TUI로 모니터링합니다 |
| `tool convert` | 🔧 도구 | QA 데이터를 훈련용 JSONL 형식으로 변환합니다 |
| `tool dialogue` | 🔧 도구 | QA 쌍을 멀티턴 대화 데이터로 확장합니다 |
| `tool gguf` | 🔧 도구 | 병합된 모델을 GGUF 양자화 형식으로 변환합니다 |
| `tool update` | 🔧 도구 | 변경된 문서만 감지하여 증분 처리합니다 |
| `status` | ℹ️ 정보 | 각 파이프라인 단계의 진행 상태를 표시합니다 |
| `clean` | ℹ️ 정보 | 중간 생성 파일을 정리합니다 |
| `version` | ℹ️ 정보 | slm-factory 버전을 출력합니다 |

---

## 🚀 시작하기

### `init`

새로운 slm-factory 프로젝트를 초기화합니다. 프로젝트 디렉토리, `documents/`, `output/` 하위 디렉토리, 기본 `project.yaml` 설정 파일을 생성합니다.

**사용법**

```
slm-factory init <name> [OPTIONS]
```

**인수**

| 인수 | 필수 여부 | 설명 |
|------|:---------:|------|
| `name` | 필수 | 생성할 프로젝트 이름. 디렉토리명과 설정 파일의 `project.name`에 사용됩니다. |

**옵션**

| 플래그 | 단축키 | 타입 | 기본값 | 설명 |
|--------|--------|------|--------|------|
| `--path` | | `TEXT` | `.` | 프로젝트를 생성할 상위 디렉토리입니다. |

**예시**

```bash
# 현재 디렉토리에 my-project 생성
slm-factory init my-project

# 특정 경로에 생성
slm-factory init policy-assistant --path /workspace/projects
```

**출력**

```
✓ 프로젝트 'my-project'가 생성되었습니다: ./my-project

프로젝트 구조:
  ./my-project/
  ./my-project/documents/
  ./my-project/output/
  ./my-project/project.yaml

사전 준비:
  1. ./my-project/documents 디렉토리에 학습할 문서(PDF, TXT 등)를 추가하세요
  2. 별도 터미널에서 Ollama를 실행하세요: ollama serve
  3. Teacher 모델을 다운로드하세요: ollama pull qwen3:8b

실행:
  4. 환경 점검: slm-factory check --config ./my-project/project.yaml
  5. wizard 실행: slm-factory tool wizard --config ./my-project/project.yaml
```

**참고**

- 지정 경로에 `project.yaml`이 이미 존재하면 덮어쓰기 여부를 확인하는 프롬프트가 표시됩니다.
- 생성된 `project.yaml`의 `project.name`과 `export.ollama.model_name`에 `name` 인수가 자동으로 반영됩니다.
- 설정 파일의 모든 옵션은 [설정 레퍼런스](configuration.md)를 참조하십시오.

---

### `check`

프로젝트 설정과 실행 환경을 사전 점검합니다. 파이프라인 실행 전에 실행하여 문제를 미리 확인하십시오.

**사용법**

```
slm-factory check [OPTIONS]
```

**옵션**

| 플래그 | 단축키 | 타입 | 기본값 | 설명 |
|--------|--------|------|--------|------|
| `--config` | | `TEXT` | `project.yaml` | 프로젝트 설정 파일 경로입니다. 현재 디렉토리부터 상위까지 자동 탐색합니다. |

**점검 항목**

| 항목 | 설명 |
|------|------|
| 설정 파일 | `project.yaml` 로드 및 Pydantic 스키마 검증 |
| 문서 디렉토리 | `paths.documents` 디렉토리 존재 여부 및 파일 유무 확인 |
| 출력 디렉토리 | `paths.output` 디렉토리 쓰기 권한 확인 |
| Ollama 연결 | `teacher.backend == "ollama"`일 때 서버 연결 상태 확인 |
| 모델 사용 가능 | `teacher.model`이 Ollama에 다운로드되어 있는지 확인 |

**종료 코드**

| 코드 | 의미 |
|------|------|
| `0` | 모든 항목 통과 |
| `1` | 하나 이상의 항목 실패 또는 경고 |

**예시**

```bash
slm-factory check --config my-project/project.yaml
```

**출력 예시 (전체 통과)**

```
                slm-factory 환경 점검
┌──────────────┬────────┬──────────────────────────────┐
│ 항목         │ 상태   │ 상세                         │
├──────────────┼────────┼──────────────────────────────┤
│ 설정 파일    │ OK     │ ./my-project/project.yaml    │
│ 문서 디렉토리│ OK     │ 3개 파일 (./documents)       │
│ 출력 디렉토리│ OK     │ 쓰기 가능 (./output)         │
│ Ollama 연결  │ OK     │ v0.3.12 (http://localhost:11434) │
│ 모델 사용 가능│ OK    │ qwen3:8b                     │
└──────────────┴────────┴──────────────────────────────┘

모든 점검 통과!
  wizard 실행: slm-factory tool wizard --config ./my-project/project.yaml
```

---

## ⚙️ 파이프라인

### `run`

문서 파싱부터 모델 배포까지 전체 파이프라인을 실행합니다. `--until` 옵션으로 특정 단계까지만 실행하고 중단할 수 있습니다.

**사용법**

```
slm-factory run [OPTIONS]
```

**옵션**

| 플래그 | 단축키 | 타입 | 기본값 | 설명 |
|--------|--------|------|--------|------|
| `--config` | | `TEXT` | `project.yaml` | 프로젝트 설정 파일 경로입니다. 현재 디렉토리부터 상위까지 자동 탐색합니다. |
| `--resume` | `-r` | `FLAG` | `False` | 중간 저장 파일에서 재개합니다. |
| `--until` | | `ENUM` | `None` | 지정된 단계까지만 실행합니다. 생략하면 전체 파이프라인을 실행합니다. |

**`--until` 단계 값**

`PipelineStep` 열거형 값을 사용합니다. 각 단계는 이전 단계의 출력을 입력으로 받습니다.

| 값 | 필수/선택 | 설명 | 생성 파일 |
|----|:---------:|------|-----------|
| `parse` | 필수 | 문서를 파싱하여 텍스트와 표를 추출합니다 | `output/parsed_documents.json` |
| `generate` | 필수 | Teacher LLM으로 QA 쌍을 생성합니다 | `output/qa_alpaca.json` |
| `validate` | 필수 | 규칙 기반 및 임베딩 기반으로 QA 쌍을 검증합니다 | `output/qa_alpaca.json` (갱신) |
| `score` | 선택 | Teacher LLM이 QA 쌍을 1~5점으로 평가하고 저품질을 제거합니다 | `output/qa_scored.json` |
| `augment` | 선택 | 질문을 다양한 표현으로 변형하여 데이터를 증강합니다 | `output/qa_augmented.json` |
| `analyze` | 선택 | 카테고리 분포, 길이 통계 등 데이터 분석 보고서를 생성합니다 | `output/data_analysis.json` |

`--until`을 생략하면 위 6단계에 이어 `convert`, `train`, `export`까지 전체 파이프라인을 실행합니다.

**`--resume` 동작 방식**

출력 디렉토리에서 가장 최근 중간 저장 파일을 탐색하여 해당 단계부터 재개합니다.

| 발견된 파일 | 재개 단계 |
|-------------|-----------|
| `qa_augmented.json` | `analyze`부터 |
| `qa_scored.json` | `augment`부터 |
| `qa_alpaca.json` | `validate`부터 |
| `parsed_documents.json` | `generate`부터 |
| 없음 | 처음부터 |

**예시**

```bash
# 전체 파이프라인 실행
slm-factory run --config my-project/project.yaml

# 문서 파싱만 실행
slm-factory run --until parse --config my-project/project.yaml

# QA 생성까지 실행
slm-factory run --until generate --config my-project/project.yaml

# 검증까지 실행
slm-factory run --until validate --config my-project/project.yaml

# 품질 평가까지 실행
slm-factory run --until score --config my-project/project.yaml

# 중단된 지점에서 재개
slm-factory run --resume --config my-project/project.yaml

# 특정 단계까지 재개하며 실행
slm-factory run --until augment --resume --config my-project/project.yaml
```

**출력 예시 (전체 파이프라인)**

```
slm-factory — 전체 파이프라인 시작 중...

  ✓ 5개 문서 파싱 완료
  ✓ 150개 QA 쌍 생성 완료
  ✓ 검증 완료: 132개 수락, 18개 거부
  ✓ 점수 평가: 120개 통과, 12개 제거
  ✓ 데이터 증강: 120개 → 360개
  ✓ 360개 QA 쌍 분석 완료

파이프라인 완료! 모델 저장 위치: ./my-project/output/merged_model
```

---

### `train`

LoRA 파인튜닝을 실행합니다. `--data`로 기존 학습 데이터를 직접 지정하거나, 생략하면 파싱부터 변환까지 전체 전처리를 자동으로 실행합니다.

**사용법**

```
slm-factory train [OPTIONS]
```

**옵션**

| 플래그 | 단축키 | 타입 | 기본값 | 설명 |
|--------|--------|------|--------|------|
| `--config` | | `TEXT` | `project.yaml` | 프로젝트 설정 파일 경로입니다. 현재 디렉토리부터 상위까지 자동 탐색합니다. |
| `--data` | | `TEXT` | `None` | 사전 생성된 `training_data.jsonl` 파일 경로입니다. 지정하면 전처리 단계를 건너뜁니다. |
| `--resume` | `-r` | `FLAG` | `False` | 중간 저장 파일에서 재개합니다. |

**동작 방식**

- `--data` 지정: 해당 JSONL 파일로 바로 학습을 시작합니다.
- `--resume` 지정 (data 없음): 중간 저장 파일을 탐색하여 해당 단계부터 전처리를 재개한 뒤 학습합니다.
- 옵션 없음: 파싱부터 변환까지 전체 전처리를 실행한 뒤 학습합니다.

**예시**

```bash
# 전체 전처리 후 학습
slm-factory train --config my-project/project.yaml

# 기존 학습 데이터로 학습만 실행
slm-factory train --config my-project/project.yaml --data ./my-project/output/training_data.jsonl

# 외부에서 준비한 데이터로 학습
slm-factory train --config my-project/project.yaml --data ./custom_data.jsonl

# 중단된 전처리부터 재개하여 학습
slm-factory train --resume --config my-project/project.yaml
```

**출력**

```
훈련 완료! 어댑터 저장 위치: ./my-project/output/checkpoints/adapter
```

**참고**

- 학습 하이퍼파라미터(`num_epochs`, `batch_size`, `learning_rate` 등)는 `project.yaml`의 `training` 섹션에서 설정합니다. [설정 레퍼런스](configuration.md)를 참조하십시오.
- GPU 메모리 부족 시 해결 방법은 [사용 가이드](guide.md)를 참조하십시오.

---

### `export`

훈련된 LoRA 어댑터를 기본 모델에 병합하고 Ollama Modelfile을 생성합니다. 파이프라인 전체를 재실행하지 않고 내보내기만 단독으로 수행합니다.

**사용법**

```
slm-factory export [OPTIONS]
```

**옵션**

| 플래그 | 단축키 | 타입 | 기본값 | 설명 |
|--------|--------|------|--------|------|
| `--config` | | `TEXT` | `project.yaml` | 프로젝트 설정 파일 경로입니다. 현재 디렉토리부터 상위까지 자동 탐색합니다. |
| `--adapter` | | `TEXT` | `None` | 어댑터 디렉토리 경로입니다. 미지정 시 `output/checkpoints/adapter/`를 사용합니다. |

**예시**

```bash
# 기본 어댑터 경로로 내보내기
slm-factory export --config my-project/project.yaml

# 특정 어댑터 경로 지정
slm-factory export --config my-project/project.yaml --adapter ./my-project/output/checkpoints/adapter
```

**출력**

```
내보내기 완료! 모델 저장 위치: ./my-project/output/merged_model
```

생성 파일:
- `output/merged_model/` — HuggingFace 형식의 병합된 모델
- `output/merged_model/Modelfile` — Ollama 배포용 Modelfile

**참고**

- Ollama 배포 설정(`model_name`, `system_prompt` 등)은 `project.yaml`의 `export.ollama` 섹션에서 설정합니다. [설정 레퍼런스](configuration.md)를 참조하십시오.

---

## 📊 평가

### `eval run`

학습된 모델을 QA 데이터로 평가합니다. BLEU와 ROUGE 메트릭을 계산하여 결과를 저장합니다.

**사용법**

```
slm-factory eval run [OPTIONS]
```

**옵션**

| 플래그 | 단축키 | 타입 | 기본값 | 설명 |
|--------|--------|------|--------|------|
| `--model` | | `TEXT` | 필수 | 평가할 Ollama 모델 이름입니다. |
| `--config` | | `TEXT` | `project.yaml` | 프로젝트 설정 파일 경로입니다. 현재 디렉토리부터 상위까지 자동 탐색합니다. |
| `--data` | | `TEXT` | `None` | QA 데이터 파일 경로입니다. 미지정 시 출력 디렉토리에서 자동 감지합니다. |

`--data` 미지정 시 자동 감지 우선순위: `qa_augmented.json` → `qa_scored.json` → `qa_alpaca.json`

**예시**

```bash
# 기본 설정으로 평가
slm-factory eval run --model my-project-model --config my-project/project.yaml

# 특정 QA 데이터로 평가
slm-factory eval run --model my-project-model --config my-project/project.yaml --data ./my-project/output/qa_alpaca.json
```

**출력**

터미널에 BLEU/ROUGE 점수 요약을 출력하고, `output/eval_results.json`에 상세 결과를 저장합니다.

```
평가 완료! 결과: ./my-project/output/eval_results.json (50건)
```

**참고**

- 평가 대상 모델은 Ollama에 등록된 모델이어야 합니다. `ollama list`로 확인하십시오.
- 평가 설정(`output_file` 등)은 `project.yaml`의 `eval` 섹션에서 설정합니다. [설정 레퍼런스](configuration.md)를 참조하십시오.

---

### `eval compare`

Base 모델과 Fine-tuned 모델의 답변을 동일한 질문으로 비교합니다. 파인튜닝 전후의 성능 차이를 확인할 때 사용합니다.

**사용법**

```
slm-factory eval compare [OPTIONS]
```

**옵션**

| 플래그 | 단축키 | 타입 | 기본값 | 설명 |
|--------|--------|------|--------|------|
| `--base-model` | | `TEXT` | 필수 | 비교 기준 모델 이름 (Ollama)입니다. |
| `--ft` | | `TEXT` | 필수 | 파인튜닝된 모델 이름 (Ollama)입니다. |
| `--config` | | `TEXT` | `project.yaml` | 프로젝트 설정 파일 경로입니다. 현재 디렉토리부터 상위까지 자동 탐색합니다. |
| `--data` | | `TEXT` | `None` | QA 데이터 파일 경로입니다. 미지정 시 출력 디렉토리에서 자동 감지합니다. |

`--data` 미지정 시 자동 감지 우선순위: `qa_augmented.json` → `qa_scored.json` → `qa_alpaca.json`

**예시**

```bash
# Base 모델과 파인튜닝 모델 비교
slm-factory eval compare \
  --base-model gemma:2b \
  --ft my-project-model \
  --config my-project/project.yaml

# 특정 QA 데이터로 비교
slm-factory eval compare \
  --base-model gemma:2b \
  --ft my-project-model \
  --config my-project/project.yaml \
  --data ./my-project/output/qa_alpaca.json
```

**출력**

터미널에 비교 요약을 출력하고, `output/compare_results.json`에 각 질문별 두 모델의 답변을 저장합니다.

```
비교 완료! 결과: ./my-project/output/compare_results.json (50건)
```

---

## 🔧 도구

### `tool wizard`

전체 파이프라인을 단계별로 안내하는 대화형 인터페이스입니다. 각 단계에서 진행 여부를 확인하며, 선택 단계는 건너뛸 수 있습니다. 처음 사용자에게 권장합니다.

**사용법**

```
slm-factory tool wizard [OPTIONS]
```

**옵션**

| 플래그 | 단축키 | 타입 | 기본값 | 설명 |
|--------|--------|------|--------|------|
| `--config` | | `TEXT` | `project.yaml` | 프로젝트 설정 파일 경로입니다. 현재 디렉토리부터 상위까지 자동 탐색합니다. |
| `--resume` | `-r` | `FLAG` | `False` | 이전 실행의 중간 결과에서 재개합니다. |

**12단계 진행 순서**

| # | 단계 | 필수/선택 | 설명 |
|---|------|:---------:|------|
| 1 | 설정 파일 로드 | 필수 | `project.yaml`을 자동 탐색하거나 경로를 직접 입력합니다 |
| 2 | 문서 선택 | 필수 | `documents/` 디렉토리의 파일 목록을 표시하고 전체 또는 개별 선택합니다 |
| 3 | 문서 파싱 | 필수 | 선택한 문서를 자동으로 파싱합니다 |
| 4 | QA 쌍 생성 | 필수 | Teacher LLM으로 질문-답변 쌍을 생성합니다. 확인 후 진행합니다. |
| 5 | QA 검증 | 필수 | 규칙 기반 및 임베딩 기반 검증을 자동으로 실행합니다 |
| 6 | 품질 점수 평가 | 선택 | Teacher LLM이 QA 쌍을 1~5점으로 평가합니다. 건너뛸 수 있습니다. |
| 7 | 데이터 증강 | 선택 | 질문을 다양한 표현으로 변형하여 데이터를 늘립니다. 건너뛸 수 있습니다. |
| 8 | LoRA 학습 | 필수 | Student 모델에 LoRA 어댑터를 적용하여 파인튜닝합니다. 확인 후 진행합니다. |
| 9 | 모델 내보내기 | 필수 | LoRA 어댑터를 병합하고 Ollama Modelfile을 생성합니다. 확인 후 진행합니다. |
| 10 | 멀티턴 대화 생성 | 선택 | QA 쌍을 멀티턴 대화 형식으로 확장합니다. 건너뛸 수 있습니다. |
| 11 | GGUF 변환 | 선택 | 모델을 llama.cpp 호환 GGUF 형식으로 변환합니다. 건너뛸 수 있습니다. |
| 12 | 모델 평가 | 선택 | BLEU/ROUGE 메트릭으로 학습된 모델을 평가합니다. 건너뛸 수 있습니다. |

선택 단계를 건너뛰면 해당 단계를 나중에 실행할 수 있는 CLI 명령어를 안내합니다.

**예시**

```bash
# 처음 실행
slm-factory tool wizard --config my-project/project.yaml

# 이전 실행에서 재개
slm-factory tool wizard --resume --config my-project/project.yaml
```

**참고**

- wizard 실행 전 Ollama 서버와 Teacher 모델이 필요합니다. 자세한 사전 준비는 [사용 가이드](guide.md)를 참조하십시오.
- `--resume` 사용 시 출력 디렉토리의 중간 파일을 자동으로 감지하여 해당 단계부터 재개합니다.

---

### `tool review`

생성된 QA 쌍을 TUI(텍스트 기반 UI)에서 하나씩 확인하며 승인, 거부, 편집합니다. 데이터 품질을 수동으로 관리할 때 사용합니다.

**사용법**

```
slm-factory tool review [OPTIONS]
```

**옵션**

| 플래그 | 단축키 | 타입 | 기본값 | 설명 |
|--------|--------|------|--------|------|
| `--config` | | `TEXT` | `project.yaml` | 프로젝트 설정 파일 경로입니다. 현재 디렉토리부터 상위까지 자동 탐색합니다. |
| `--data` | | `TEXT` | `None` | QA 데이터 파일 경로입니다. 미지정 시 출력 디렉토리에서 자동 감지합니다. |

`--data` 미지정 시 자동 감지 우선순위: `qa_reviewed.json` → `qa_augmented.json` → `qa_scored.json` → `qa_alpaca.json`

**예시**

```bash
# 자동 감지된 QA 데이터로 리뷰 시작
slm-factory tool review --config my-project/project.yaml

# 특정 파일로 리뷰 시작
slm-factory tool review --config my-project/project.yaml --data ./my-project/output/qa_alpaca.json
```

**출력**

TUI 종료 후 리뷰 결과를 출력합니다.

```
리뷰 완료! 승인: 98, 거부: 12, 대기: 5
```

생성 파일: `output/qa_reviewed.json` (승인된 QA 쌍만 포함)

---

### `tool dashboard`

파이프라인 각 단계의 진행 상태를 실시간으로 모니터링하는 TUI 대시보드를 실행합니다.

**사용법**

```
slm-factory tool dashboard [OPTIONS]
```

**옵션**

| 플래그 | 단축키 | 타입 | 기본값 | 설명 |
|--------|--------|------|--------|------|
| `--config` | | `TEXT` | `project.yaml` | 프로젝트 설정 파일 경로입니다. 현재 디렉토리부터 상위까지 자동 탐색합니다. |

**예시**

```bash
slm-factory tool dashboard --config my-project/project.yaml
```

대시보드는 각 단계의 출력 파일 존재 여부, 항목 수, 최근 수정 시각을 표시하며 설정된 주기(`dashboard.refresh_interval`)로 자동 갱신합니다.

---

### `tool convert`

QA 데이터를 Student 모델의 채팅 템플릿이 적용된 훈련용 JSONL 형식으로 변환합니다. 파이프라인 전체를 재실행하지 않고 변환만 단독으로 수행합니다.

**사용법**

```
slm-factory tool convert [OPTIONS]
```

**옵션**

| 플래그 | 단축키 | 타입 | 기본값 | 설명 |
|--------|--------|------|--------|------|
| `--config` | | `TEXT` | `project.yaml` | 프로젝트 설정 파일 경로입니다. 현재 디렉토리부터 상위까지 자동 탐색합니다. |
| `--data` | | `TEXT` | `None` | QA 데이터 파일 경로 (`qa_alpaca.json` 또는 `qa_augmented.json`). 미지정 시 자동 감지합니다. |

**`--data` 미지정 시 자동 감지 우선순위**

출력 디렉토리에서 다음 순서로 파일을 탐색합니다.

1. `qa_augmented.json`
2. `qa_scored.json`
3. `qa_alpaca.json`

**예시**

```bash
# 자동 감지된 QA 데이터로 변환
slm-factory tool convert --config my-project/project.yaml

# 특정 파일로 변환
slm-factory tool convert --config my-project/project.yaml --data ./my-project/output/qa_augmented.json
```

**출력**

```
변환 완료! 훈련 데이터: ./my-project/output/training_data.jsonl (360개 쌍)
```

생성 파일: `output/training_data.jsonl`

---

### `tool dialogue`

QA 쌍을 멀티턴 대화 형식으로 확장합니다. Teacher LLM이 단일 질문-답변을 자연스러운 다중 턴 대화로 변환합니다.

**사용법**

```
slm-factory tool dialogue [OPTIONS]
```

**옵션**

| 플래그 | 단축키 | 타입 | 기본값 | 설명 |
|--------|--------|------|--------|------|
| `--config` | | `TEXT` | `project.yaml` | 프로젝트 설정 파일 경로입니다. 현재 디렉토리부터 상위까지 자동 탐색합니다. |
| `--data` | | `TEXT` | `None` | QA 데이터 파일 경로 (`qa_alpaca.json` 또는 `qa_augmented.json`). 미지정 시 자동 감지합니다. |

`--data` 미지정 시 자동 감지 우선순위: `qa_augmented.json` → `qa_scored.json` → `qa_alpaca.json`

**예시**

```bash
# 자동 감지된 QA 데이터로 대화 생성
slm-factory tool dialogue --config my-project/project.yaml

# 특정 파일로 대화 생성
slm-factory tool dialogue --config my-project/project.yaml --data ./my-project/output/qa_alpaca.json
```

**출력**

```
대화 생성 완료! 120개 대화 생성됨 → ./my-project/output/dialogues.json
```

생성 파일: `output/dialogues.json`

**참고**

- 대화 생성에는 Ollama 서버와 Teacher 모델이 필요합니다.
- 대화 설정(`turns`, `style` 등)은 `project.yaml`의 `dialogue` 섹션에서 설정합니다. [설정 레퍼런스](configuration.md)를 참조하십시오.

---

### `tool gguf`

병합된 모델을 llama.cpp 호환 GGUF 양자화 형식으로 변환합니다. llama.cpp의 변환 스크립트를 내부적으로 사용합니다.

**사용법**

```
slm-factory tool gguf [OPTIONS]
```

**옵션**

| 플래그 | 단축키 | 타입 | 기본값 | 설명 |
|--------|--------|------|--------|------|
| `--config` | | `TEXT` | `project.yaml` | 프로젝트 설정 파일 경로입니다. 현재 디렉토리부터 상위까지 자동 탐색합니다. |
| `--model-dir` | | `TEXT` | `None` | 병합된 모델 디렉토리 경로입니다. 미지정 시 `output/merged_model`을 사용합니다. |

**예시**

```bash
# 기본 경로의 모델을 GGUF로 변환
slm-factory tool gguf --config my-project/project.yaml

# 특정 모델 디렉토리 지정
slm-factory tool gguf --config my-project/project.yaml --model-dir ./my-project/output/merged_model
```

**출력**

```
GGUF 변환 완료! 파일: ./my-project/output/merged_model.gguf
```

**참고**

- GGUF 변환 전에 `export` 명령으로 모델을 먼저 병합해야 합니다.
- 양자화 타입은 `project.yaml`의 `gguf_export.quantization_type`에서 설정합니다. [설정 레퍼런스](configuration.md)를 참조하십시오.

---

### `tool update`

문서 디렉토리를 스캔하여 변경된 파일만 감지하고, 새 QA를 생성하여 기존 QA와 병합합니다. 문서를 추가하거나 수정했을 때 전체 파이프라인을 재실행하지 않고 증분 처리할 수 있습니다.

**사용법**

```
slm-factory tool update [OPTIONS]
```

**옵션**

| 플래그 | 단축키 | 타입 | 기본값 | 설명 |
|--------|--------|------|--------|------|
| `--config` | | `TEXT` | `project.yaml` | 프로젝트 설정 파일 경로입니다. 현재 디렉토리부터 상위까지 자동 탐색합니다. |

**동작 방식**

1. `IncrementalTracker`가 각 문서 파일의 해시값을 계산합니다.
2. 이전 실행 시 저장된 해시값과 비교하여 변경된 파일을 감지합니다.
3. 변경된 파일만 파싱하고 QA를 생성합니다.
4. `project.yaml`의 `incremental.merge_strategy` 설정에 따라 기존 `qa_alpaca.json`과 병합합니다.

**예시**

```bash
# 변경된 문서만 증분 처리
slm-factory tool update --config my-project/project.yaml
```

**출력**

```
증분 업데이트: 2개 변경 문서 감지

증분 업데이트 완료! 변경 문서: 2개, 새 QA: 40개, 전체 QA: 190개 (전략: append)
```

변경된 문서가 없으면 "변경된 문서가 없습니다" 메시지를 출력하고 종료합니다.

**참고**

- 병합 전략(`merge_strategy`)은 `project.yaml`의 `incremental` 섹션에서 설정합니다. [설정 레퍼런스](configuration.md)를 참조하십시오.

---

## ℹ️ 정보

### `status`

파이프라인 각 단계의 진행 상태를 확인합니다. 출력 디렉토리의 파일 존재 여부와 항목 수를 테이블로 표시합니다.

**사용법**

```
slm-factory status [OPTIONS]
```

**옵션**

| 플래그 | 단축키 | 타입 | 기본값 | 설명 |
|--------|--------|------|--------|------|
| `--config` | | `TEXT` | `project.yaml` | 프로젝트 설정 파일 경로입니다. 현재 디렉토리부터 상위까지 자동 탐색합니다. |

**예시**

```bash
slm-factory status --config my-project/project.yaml
```

**출력 예시**

```
              파이프라인 진행 상태
┌──────────┬──────────────────────────┬────────┬──────────────┐
│ 단계     │ 파일                     │ 상태   │ 건수         │
├──────────┼──────────────────────────┼────────┼──────────────┤
│ parse    │ parsed_documents.json    │ 존재   │ 5개 문서     │
│ generate │ qa_alpaca.json           │ 존재   │ 150개 쌍     │
│ score    │ qa_scored.json           │ 없음   │ -            │
│ augment  │ qa_augmented.json        │ 없음   │ -            │
│ analyze  │ data_analysis.json       │ 없음   │ -            │
│ convert  │ training_data.jsonl      │ 없음   │ -            │
│ train    │ checkpoints/adapter/     │ 없음   │ -            │
│ export   │ merged_model/            │ 없음   │ -            │
└──────────┴──────────────────────────┴────────┴──────────────┘

다음 --resume 실행 시 validate부터 재개됩니다
```

모든 단계가 완료된 경우 "모든 단계가 완료되었습니다" 메시지를 출력합니다.

---

### `clean`

출력 디렉토리의 중간 생성 파일을 정리합니다. 삭제 전 확인 프롬프트가 표시됩니다.

**사용법**

```
slm-factory clean [OPTIONS]
```

**옵션**

| 플래그 | 단축키 | 타입 | 기본값 | 설명 |
|--------|--------|------|--------|------|
| `--config` | | `TEXT` | `project.yaml` | 프로젝트 설정 파일 경로입니다. 현재 디렉토리부터 상위까지 자동 탐색합니다. |
| `--all` | | `FLAG` | `False` | 출력 디렉토리의 모든 파일을 삭제합니다. |

**기본 동작 vs `--all` 동작**

| 동작 | 삭제 대상 |
|------|-----------|
| 기본 (`--all` 없음) | `qa_scored.json`, `qa_augmented.json`, `data_analysis.json` |
| `--all` | 출력 디렉토리(`output/`)의 모든 파일과 디렉토리 |

**예시**

```bash
# 중간 파일만 정리
slm-factory clean --config my-project/project.yaml

# 모든 출력 파일 정리 (학습 결과 포함)
slm-factory clean --all --config my-project/project.yaml
```

**출력**

```
삭제 대상:
  ./my-project/output/qa_scored.json
  ./my-project/output/qa_augmented.json
  ./my-project/output/data_analysis.json

삭제하시겠습니까? [y/N]:

          삭제 결과
┌──────────────────────────────────┬────────┐
│ 파일                             │ 상태   │
├──────────────────────────────────┼────────┤
│ ./my-project/output/qa_scored... │ 삭제됨 │
│ ...                              │ 삭제됨 │
└──────────────────────────────────┴────────┘

3개 항목 삭제 완료
```

---

### `version`

slm-factory의 현재 버전을 출력합니다.

**사용법**

```
slm-factory version
```

**출력**

```
slm-factory 0.1.0
```

---

## 출력 파일 구조

파이프라인 실행 후 `output/` 디렉토리에 생성되는 파일 목록입니다.

```
output/
├── parsed_documents.json       # parse 단계 출력
├── qa_alpaca.json              # generate + validate 단계 출력
├── qa_scored.json              # score 단계 출력 (선택)
├── qa_augmented.json           # augment 단계 출력 (선택)
├── qa_reviewed.json            # tool review 출력 (선택)
├── data_analysis.json          # analyze 단계 출력 (선택)
├── dialogues.json              # tool dialogue 출력 (선택)
├── eval_results.json           # eval run 출력 (선택)
├── compare_results.json        # eval compare 출력 (선택)
├── training_data.jsonl         # convert 단계 출력
├── checkpoints/
│   └── adapter/                # train 단계 출력
│       ├── adapter_config.json
│       ├── adapter_model.safetensors
│       └── ...
└── merged_model/               # export 단계 출력
    ├── config.json
    ├── model.safetensors
    ├── tokenizer.json
    ├── tokenizer_config.json
    └── Modelfile
```

**파일별 설명**

| 파일 | 생성 명령 | 설명 |
|------|-----------|------|
| `parsed_documents.json` | `run --until parse` | 원본 문서에서 추출한 텍스트, 표, 메타데이터입니다. `--resume` 시 파싱 단계를 건너뜁니다. |
| `qa_alpaca.json` | `run --until generate` | Teacher LLM이 생성한 QA 쌍입니다. Alpaca 형식(`instruction`, `input`, `output`)으로 저장됩니다. |
| `qa_scored.json` | `run --until score` | 품질 점수 평가를 통과한 QA 쌍입니다. `--resume` 시 `augment`부터 재개합니다. |
| `qa_augmented.json` | `run --until augment` | 데이터 증강이 완료된 QA 쌍입니다. `--resume` 시 `analyze`부터 재개합니다. |
| `qa_reviewed.json` | `tool review` | TUI에서 수동 리뷰를 거친 QA 쌍입니다. 승인된 항목만 포함됩니다. |
| `data_analysis.json` | `run --until analyze` | 카테고리 분포, 길이 통계, 데이터 품질 경고를 포함한 분석 보고서입니다. |
| `dialogues.json` | `tool dialogue` | QA 쌍을 멀티턴 대화 형식으로 확장한 데이터입니다. |
| `eval_results.json` | `eval run` | BLEU/ROUGE 메트릭별 점수와 평균 점수를 포함한 평가 결과입니다. |
| `compare_results.json` | `eval compare` | 각 질문에 대한 Base 모델과 Fine-tuned 모델의 답변을 나란히 기록한 비교 결과입니다. |
| `training_data.jsonl` | `tool convert` | Student 모델의 채팅 템플릿이 적용된 학습 데이터입니다. 각 줄은 `{"text": "..."}` 형식입니다. |
| `checkpoints/adapter/` | `train` | PEFT 형식의 LoRA 어댑터 가중치입니다. `export` 명령으로 기본 모델과 병합합니다. |
| `merged_model/` | `export` | LoRA 어댑터가 병합된 최종 모델입니다. `Modelfile`로 Ollama에 즉시 배포할 수 있습니다. |

---

*관련 문서: [설정 레퍼런스](configuration.md) · [사용 가이드](guide.md) · [아키텍처](architecture.md)*
