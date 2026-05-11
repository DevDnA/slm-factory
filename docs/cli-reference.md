# CLI 레퍼런스

> `rf` 명령어와 옵션 정리. 처음 쓰시는 분은 [Quick Start](index.html)부터 보세요.

`rf`는 저장소 루트의 `uv run --project <repo> rag-factory [args]` wrapper입니다. venv 활성화 불필요. 어디서든 실행 가능.

## 전역 옵션

| 옵션 | 설명 |
|---|---|
| `-v`, `--verbose` | 디버그 로그 |
| `-q`, `--quiet` | 경고·에러만 |
| `-V`, `--version` | 버전 표시 |
| `--install-completion` | 현재 셸에 자동완성 설치 |
| `--help` | 도움말 |

## 자주 쓰는 명령 (RAG 위주)

| 명령어 | 역할 |
|---|---|
| [`rf init <name>`](#rf-init) | 프로젝트 템플릿 복사 |
| [`rf rag`](#rf-rag) | RAG 인덱스 빌드 + 웹 채팅 (30초) — **주력 명령** |
| [`rf check`](#rf-check) | 문서·설정·환경 사전 점검 |
| [`rf status`](#rf-status) | 진행 상태 |
| [`rf clean`](#rf-clean) | 산출물 정리 |

## 부수 명령 (Fine-tuning, 잘 안 됨)

> 1B Student 모델 파인튜닝은 소규모 데이터(<100개 QA)에서 거의 100% 과적합되어 실제 환경에서 잘 동작하지 않습니다. RAG가 도메인 답변의 대부분을 해결하므로 평소엔 `rf rag`만 쓰면 됩니다. 아래는 실험용·고급 사용자용.

| 명령어 | 역할 |
|---|---|
| `rf tune` | 풀 파이프라인(파싱→QA생성→학습→RAG) |
| `rf train` | 학습 단계만 |
| `rf export` | LoRA 병합 + Ollama Modelfile |
| `rf eval` | BLEU/ROUGE 평가 |
| `rf tool ...` | 디버깅·외부 평가용 유틸리티 |

설정 파일(`project.yaml`)은 현재 디렉터리에서 상위로 자동 탐색합니다. 명시하려면 `--config <path>`.

## rf init

새 프로젝트 디렉터리를 템플릿에서 복사.

```bash
rf init my-project              # ./my-project/ 생성
rf init my-project --force      # 이미 있으면 덮어쓰기
```

생성:

```
my-project/
├─ project.yaml      # 모든 설정 (수정해서 사용)
├─ documents/        # 도메인 문서를 여기에
└─ output/           # 산출물 (자동 생성, gitignored)
```

## rf rag

RAG 인덱스를 빌드(필요 시)하고 FastAPI 서버 + 웹 채팅을 시작.

```bash
cd my-project
rf rag                          # 인덱스 + 서버 + 브라우저 채팅
rf rag --no-chat                # 인덱스만 빌드 후 종료
rf rag --config ../other.yaml   # 다른 설정 파일
```

| 옵션 | 기본 | 설명 |
|---|---|---|
| `--config` | `project.yaml` | 설정 파일 (상위 자동 탐색) |
| `--chat`/`--no-chat` | `chat` | 인덱스 후 채팅 자동 시작 |

서버 주소: `http://localhost:8000` (포트는 `rag.port`로 변경).

엔드포인트:
- `/` — opencode-ai 스타일 추론 UI 웹 채팅
- `/auto` — SSE 스트리밍, 라우팅 자동 (chitchat/general/simple/agent)
- `/v1/chat/completions` — OpenAI 호환, OpenWebUI 등과 연동
- `/health/ready` — 헬스 체크 (200 = 인덱스 로드 완료)

## rf check

실행 전 사전 점검.

```bash
rf check                        # 표준 점검
rf check --strict               # 경고도 실패로 처리
```

확인 항목:
- `project.yaml` 스키마 (Pydantic v2)
- `documents/` 안 파일 포맷 인식 가능 여부 (PDF, HWP, HWPX, DOCX, DOC, PPT, PPTX, XLSX, XLS, HTML, TXT, MD 12종)
- Teacher 모델 Ollama 응답
- GPU/MPS/CPU 감지
- 디스크 여유 공간

## rf status

각 파이프라인 단계의 체크포인트 존재 여부와 결과 요약.

```bash
rf status
```

`output/` 안 어떤 단계까지 완료됐는지 한눈에 표시.

## rf clean

중간 산출물 정리.

```bash
rf clean                        # 캐시·중간 JSON만 (인덱스·모델은 보존)
rf clean --all                  # 전부 (다음 실행 시 처음부터)
rf clean --keep-index           # 인덱스만 보존
```

## rf tune (advanced)

풀 파이프라인 — 파싱부터 모델 내보내기까지 한 번에.

> **주의**: Student 모델 파인튜닝은 잘 안 됩니다. 소규모 데이터에서 과적합 100%. RAG만 쓰시는 게 안전합니다. 그래도 시도하시려면 아래 옵션 참고.

```bash
rf tune                         # 풀 파이프라인
rf tune --resume                # 마지막 체크포인트에서 재개
rf tune --skip-train            # 학습만 건너뛰기 (인덱스 빌드까지)
rf tune --no-chat               # 채팅 자동 시작 안 함
```

13단계: parse → generate QA → validate → score → augment → analyze → convert → train → export → eval → refine → corpus export → rag index. 각 단계 JSON 체크포인트로 resumable.

## rf train (advanced)

학습 단계만. QA 데이터 사전 생성 필요.

```bash
rf train                        # 기본
rf train --resume               # 학습 체크포인트에서 재개
rf train --data path/to/qa.json # 외부 QA 파일
```

`training.*` 섹션의 LoRA·배치·학습률 설정에 따라 동작.

## rf export (advanced)

LoRA 어댑터를 베이스 모델에 병합 + Ollama Modelfile 생성.

```bash
rf export                       # 기본 — Ollama용
rf export --target huggingface  # HF 포맷 (safetensors)
rf export --quantize q4_k_m     # GGUF 양자화 (Ollama만)
```

## rf eval (advanced)

학습 모델 BLEU/ROUGE 평가.

```bash
rf eval                         # eval_set 자동 분할
rf eval --baseline              # 베이스 모델과 비교
```

결과: `output/eval/eval_result.json`.

## rf tool

유틸리티 서브커맨드 모음. 자주 쓰진 않지만 디버깅·실험용.

```bash
rf tool compare-models a b      # 두 학습 모델 응답 비교
rf tool export-corpus           # 외부 평가용 corpus parquet
rf tool eval-retrieval          # RAG 검색 품질만 측정
rf tool review-qa               # 생성 QA 사람 검토
rf tool evolve                  # 자동 진화 학습 (실험적)
rf tool convert                 # 학습 모델 포맷 변환
rf tool rag-index               # 인덱스만 빌드 (서버 X)
rf tool rag-serve               # 서버만 (인덱스 빌드 X)
```

각 서브커맨드의 옵션은 `rf tool <name> --help`로 확인.

## 출력 파일 구조

`rf tune` 풀 파이프라인 산출물 (`rf rag`만 쓰면 `parsed/`, `rag/`, `logs/`만 생성):

```
output/
├─ parsed/         # 문서 파싱 (JSON)
├─ qa_raw.json     # Teacher가 생성한 raw QA          [tune only]
├─ qa_valid.json   # 검증 통과 QA                     [tune only]
├─ qa_scored.json  # 점수 부여 QA                     [tune only]
├─ qa_final.json   # 증강·필터링 완료 QA              [tune only]
├─ checkpoints/    # LoRA 학습 체크포인트              [tune only]
├─ adapter/        # 최종 LoRA 어댑터                  [tune only]
├─ merged/         # 베이스 + 어댑터 병합              [tune only]
├─ ollama/         # Ollama Modelfile + GGUF           [tune only]
├─ eval/           # BLEU/ROUGE 결과                   [tune only]
├─ rag/            # Qdrant 인덱스·corpus profile
└─ logs/           # 단계별 로그
```

## 관련

- [설정 레퍼런스](configuration.html) — `project.yaml` 필드 정리
- [Quick Start](index.html) — 5분 안에 첫 RAG 채팅
