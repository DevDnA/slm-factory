# 설정 레퍼런스

> `project.yaml` 필드 정리. RAG 중심 설명 — 실제로 거의 모든 운영 노브가 `rag.*` 안에 있습니다.

`rf init`이 복사하는 기본 템플릿(`templates/project.yaml`)이 바로 쓸 수 있는 합리적 기본값입니다. 이 문서는 "어떤 값을 만져야 할지" 알아가기 위한 참고서.

설정 변경 후 `rf check`로 검증을 권장합니다 (Pydantic v2 스키마 + 의미적 검증).

## 1. 자주 만지는 필드

| 필드 | 기본 | 역할 |
|---|---|---|
| `project.name` | `"my-project"` | 프로젝트 이름 (로그·Ollama 모델명 prefix 등) |
| `paths.documents` | `"./documents"` | 도메인 문서 디렉터리 (상대경로는 config 파일 기준) |
| `paths.output` | `"./output"` | 산출물 디렉터리 |
| `parsing.formats` | `[pdf, txt, html]` | 파싱할 확장자 (필요시 `docx, hwp, hwpx, ppt, pptx, xlsx, xls, doc, md` 추가) |
| `teacher.model` | `"qwen3.5:9b"` | Teacher LLM (Ollama 모델) |
| `rag.ollama_model` | `"qwen3.5:9b"` | RAG 답변 합성 모델 (빈 값이면 `teacher.model` 사용) |
| `rag.agent.smart_mode` | `true` | Agent RAG 한 줄 활성화 — 추천 기본값 |

대부분 사용자는 `paths.documents`만 바꾸고 바로 `rf rag`를 실행해도 동작합니다.

## 2. 문서 파싱·청킹

```yaml
parsing:
  formats: [pdf, txt, html]      # 활성 파서
  encoding: "utf-8"              # txt 인코딩
  pdf_extract_images: false      # OCR 사용 여부
  ocr_lang: "kor+eng"            # OCR 언어 (tesseract 설치 필요)

chunking:
  chunk_size: 800                # 청크 길이 (auto 캘리브레이션 가능: "auto")
  chunk_overlap: 100             # 청크 간 중복
  section_aware: true            # 장·절·항 구조 자동 감지
```

> **섹션 인식 청킹**: 한국어 정책 문서·법령에서 "제N조", "제N절" 같은 패턴을 자동 감지해 의미 단위로 분할합니다. 일반 문서엔 큰 영향 없음.

## 3. RAG (주력)

`rag.*` 섹션이 가장 중요합니다. 검색 품질·답변 정확도가 여기서 결정.

```yaml
rag:
  port: 8000
  log_level: "info"
  cors_origins: ["*"]
  request_timeout: 600.0
  max_tokens: -1                  # EOS까지 자연 종료 (-1) 또는 2048~4096 cap
  ollama_model: "qwen3.5:9b"      # 합성 fallback. 빈 값이면 teacher.model

  # ---- 검색 강화 ----
  hyde_enabled: true              # LLM 가상 답변으로 임베딩 강화 (recall↑, +~1.5s)
  multi_query_enabled: true       # 질의 N개 변형 + RRF 병합 (어휘 변동성에 강건)
  multi_query_count: 3
  hybrid_search: true             # 벡터 + BM25 키워드 결합
  reranker_enabled: true          # cross-encoder 재정렬
  reranker_model: "dragonkue/bge-reranker-v2-m3-ko"  # 한국어 파인튜닝
  min_score: 0.0                  # 최소 유사도 cutoff

  # ---- 도메인 자기 적응 ----
  corpus_profile:
    enabled: true                 # 인덱싱 시 LLM이 corpus의 도메인·약어 자동 추출
    auto_generate: true
    sample_size: 16               # 표본 청크 수 (8은 약어 누락 빈번)
```

### Agent RAG (`rag.agent`)

`smart_mode: true` 한 줄로 다단계 검색·합성 cascade가 켜집니다.

```yaml
rag:
  agent:
    enabled: true
    smart_mode: true              # planner+verifier+intent_classifier+clarifier+
                                  # personas+session_source_reuse+legacy_fallback 일괄 ON
    intent_verbalization_enabled: true  # 라우팅 결정을 thought 이벤트로 발화
    parallel_steps: false         # macOS Python 3.14에서 false 권장 (loky SIGSEGV 회피)
    ollama_keep_alive: "168h"     # 정수 -1은 OK이나 문자열 "-1"은 Ollama가 거부

    max_iterations: 5             # ReAct 루프 상한
    session_ttl: 3600             # 대화 세션 유지 (초)
    max_history_turns: 20
    stream_reasoning: true        # thought/action/observation SSE 스트리밍

    # 컴포넌트별 모델 (빈 슬롯은 rag.ollama_model로 fallback)
    models:
      synthesis_model: "qwen3.5:9b"    # 답변 합성 — 가장 큰 영향
      clarifier_model: "qwen3.5:9b"
      planner_model:   "qwen3.5:9b"    # JSON plan 생성
      verifier_model:  "qwen3.5:9b"    # 사전 충분성 게이트
      router_model:    "qwen3.5:9b"    # 의도 분류 + HyDE/Multi-Query enhancer
```

> **24GB 통합 메모리 권장 구성**: 모든 슬롯 `qwen3.5:9b` 통일 + LaunchAgent warmup. 16GB 이하 환경은 모두 `qwen3.5:4b`로 다운그레이드.

#### 의도 분류 8 카테고리

IntentClassifier가 corpus profile과 8 카테고리로 query를 분류:

| 카테고리 | 처리 |
|---|---|
| `chitchat` | 정규식 fast-path, LLM 호출 없이 1초 응답 |
| `general` | corpus 외 일반 지식 — 추측 없이 정중한 거절 + 도메인 안내 |
| `factual`/`comparative`/`analytical`/`procedural`/`exploratory` | 도메인 — Agent 경로로 다단계 검색·합성 |
| `ambiguous` | Clarifier가 역질문으로 명확화 |

#### Contextual Retrieval

```yaml
rag:
  contextual_retrieval:
    enabled: true                 # Anthropic 패턴 — 각 청크에 부모 문서 맥락 prefix 부여
    cache_enabled: true           # 동일 청크 재인덱싱 시 LLM 호출 생략
```

## 4. Teacher LLM

```yaml
teacher:
  model: "qwen3.5:9b"            # Ollama 모델 (qwen3.5:27b도 가능, 24GB+ VRAM 필요)
  api_base: "http://localhost:11434"
  temperature: 0.3
  max_tokens: 2048
  request_timeout: 120.0
```

> **모델 추천**: `qwen3.5:9b` (기본·권장), `qwen3.5:27b` (고품질, 메모리 여유 시), `exaone3.5:7.8b` (한국어 특화).

## 5. Fine-tuning (실험적, 잘 안 됨)

> 1B Student 파인튜닝은 소규모 데이터(<100 QA)에서 거의 100% 과적합합니다. RAG가 도메인 답변의 대부분을 해결하므로 평소엔 건드리지 마세요. 아래는 시도하실 분을 위한 참조.

| 섹션 | 역할 |
|---|---|
| `questions` | Teacher가 청크당 생성할 질문 수·다양성 |
| `validation` | 정규식·유사도 기반 QA 거르기 |
| `scoring` | Teacher가 1-5점으로 QA 채점 |
| `augment` | 질문 패러프레이즈 증강 |
| `analyzer` | QA 통계 분석 |
| `student` | Student 모델 (`Qwen/Qwen2.5-1.5B-Instruct` 권장) |
| `training.lora` | LoRA rank/alpha/dropout |
| `training` | batch/lr/epochs/quantization |
| `export` | LoRA 병합 + Ollama Modelfile |
| `eval` | BLEU/ROUGE |

소규모 학습 검증 파라미터(29 QA, MPS, Qwen2.5-1.5B):

```yaml
student:
  model: "Qwen/Qwen2.5-1.5B-Instruct"
training:
  lora: { r: 8, alpha: 8, dropout: 0.1 }
  batch_size: 1
  gradient_accumulation_steps: 4
  learning_rate: 3e-5
  num_epochs: 3
  quantization: { enabled: false }    # MPS 안정성
  weight_decay: 0.01
  label_smoothing_factor: 0.0
  neftune_noise_alpha: 5.0
  completion_only_loss: true          # assistant 토큰만 loss 계산
```

## 6. 운영 (incremental·review)

```yaml
incremental:
  enabled: true                     # 문서 hash 기반 변경분만 재처리

review:
  enabled: false                    # 사람 검토 단계 (rf tool review-qa)

refinement:
  enabled: false                    # Iterative refinement (실험적)

evolve:
  enabled: false                    # 자동 진화 학습 (실험적)
```

## 7. 알려진 호환성 이슈

| 이슈 | 회피 |
|---|---|
| macOS Python 3.14에서 `parallel_steps: true` 시 SIGSEGV/SIGABRT | `parallel_steps: false`, `TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1` 환경변수 |
| Ollama `keep_alive: "-1"` HTTP 400 | `ollama_keep_alive: "168h"` 사용 (정수 `-1`은 OK이나 YAML 직렬화상 문자열 `"-1"`은 거부됨) |
| 24GB 메모리에서 큰 모델 동시 상주 swap thrashing | 단일 9b 또는 9b+4b 분리 |
| Gemma-3 GGUF 변환 시 vocab mismatch | Student는 `Qwen/Qwen2.5-1.5B-Instruct` 사용 |

## 8. 설정 검증

```bash
rf check                          # 스키마 + 의미 검증
rf check --strict                 # 경고도 실패로 처리
```

확인 사항:
- Pydantic v2 스키마 (필수 필드, 타입, 범위)
- 파일 경로 존재 여부
- Ollama 모델 응답
- 디바이스(GPU/MPS/CPU) 가용성

## 관련

- [CLI 레퍼런스](cli-reference.html) — 명령어와 옵션
- [Quick Start](index.html) — 5분 안에 첫 RAG 채팅
