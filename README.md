<div align="center">

# rag-factory

**도메인 문서 넣으면, 30초 안에 로컬 RAG 채팅.**

ChatGPT에 사내 문서를 묻지 마세요. Ollama 기반으로 모든 처리가 로컬에서 실행되고, 문서에 없는 내용은 추측하지 않습니다.

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![Ollama](https://img.shields.io/badge/Ollama-black?logo=ollama&logoColor=white)](https://ollama.com)

[Quick Start](https://devdna.github.io/rag-factory/) · [CLI 레퍼런스](https://devdna.github.io/rag-factory/cli-reference.html) · [설정 레퍼런스](https://devdna.github.io/rag-factory/configuration.html)

</div>

<br>

## 빠른 시작

```bash
git clone https://github.com/DevDnA/rag-factory.git
cd rag-factory && ./setup.sh

./rf init my-project
cp /path/to/your/*.pdf my-project/documents/

cd my-project && ../rf rag
```

브라우저 → **http://localhost:8000** — 채팅 시작.

> `setup.sh`가 Python 환경·의존성·Ollama·기본 Teacher 모델(`qwen3.5:9b`)을 한 번에 준비합니다. PDF, HWP, HWPX, DOCX, DOC, PPTX, PPT, XLSX, XLS, HTML, TXT, MD 12종 자동 감지.

## 왜 RAG 우선인가

도메인 문서로 LLM을 "학습"시키는 건 직관적이지만 실제론 잘 안 됩니다 — 1B 모델은 소규모 데이터(<100 QA)에서 거의 100% 과적합되어 학습 데이터와 비슷한 질문엔 답하지만 조금만 표현이 달라도 깨진 출력을 냅니다.

**대안은 RAG**: 매 질의마다 관련 문서 청크를 검색해 LLM에 넘기고, LLM은 그 청크만 근거로 답변합니다. 문서가 바뀌면 인덱스만 다시 빌드하면 되고, "문서에 없으면 없다고 답한다"는 정책도 프롬프트로 강제할 수 있습니다.

rag-factory는 이 RAG 파이프라인의 **검색·합성 품질에 집중합니다**:

| 컴포넌트 | 기여 |
|---|---|
| **하이브리드 검색** | 벡터 + BM25 키워드 RRF 결합으로 어휘 변동성에 강건 |
| **Cross-Encoder 리랭킹** | `dragonkue/bge-reranker-v2-m3-ko` — 한국어 파인튜닝 |
| **HyDE + Multi-Query** | LLM 가상 답변·질의 변형으로 짧은 질의 recall ↑ |
| **Contextual Retrieval** | 청크에 부모 문서 맥락 prefix 부여 (Anthropic 패턴) |
| **Corpus-Aware 라우팅** | 인덱싱 시 LLM이 도메인 약어·핵심 키워드 자동 추출 |
| **8 카테고리 의도 분류** | chitchat·general은 RAG 우회, domain만 다단계 검색 |
| **Agent RAG** | `smart_mode: true` 한 줄로 Planner/Verifier 다단계 검색 활성화 |

## 8 카테고리 의도 분류

IntentClassifier가 corpus profile + 8 카테고리로 질의를 분류해 비용·정확도를 모두 최적화합니다:

- **chitchat** — 인사·잡담은 정규식 fast-path로 LLM 호출 없이 1초 응답
- **general** — corpus 외 일반 지식은 정중히 거절 + 도메인 안내 (학습 prior 누설·환각 방지)
- **factual / comparative / analytical / procedural / exploratory** — 도메인 질의를 Agent 경로로 다단계 검색·합성
- **ambiguous** — 진짜 모호한 질의는 Clarifier가 역질문

## opencode-ai 스타일 추론 UI

웹 채팅(`http://localhost:8000`)은 RAG 에이전트의 다단계 검색 과정을 tool-call 카드로 렌더링합니다:

- `.r-toolcall` — 도구 호출 1건 카드, 좌측 border 색이 상태(`⏵` running / `✓` done / `✗` failed)로 전환
- `.r-toolhead` — `tool args` 한 줄 (monospace, 상태 아이콘)
- `.r-toolout` — `→ <observation summary>` indented dimmed

OpenAI 호환 엔드포인트(`/v1/chat/completions`, `/v1/models`)도 제공 — OpenWebUI·LangChain·LlamaIndex 등과 바로 연동.

## Fine-tuning은 실험적

`rf tune` 풀 파이프라인(파싱 → QA 생성 → LoRA 학습 → Ollama 등록)도 동작하지만 **소규모 데이터에서 과적합이 심해 추천하지 않습니다**. 도메인 답변의 대부분은 RAG가 해결합니다.

그래도 시도하실 분은 [CLI 레퍼런스의 advanced 섹션](https://devdna.github.io/rag-factory/cli-reference.html#rf-tune-advanced)을 참조하세요. 검증된 소규모 학습 파라미터(29 QA, MPS, Qwen2.5-1.5B)는 [설정 레퍼런스](https://devdna.github.io/rag-factory/configuration.html)에 정리되어 있습니다.

## 문서

| | |
|---|---|
| [Quick Start](https://devdna.github.io/rag-factory/) | 5분 안에 첫 RAG 채팅 |
| [CLI 레퍼런스](https://devdna.github.io/rag-factory/cli-reference.html) | `rf` 명령어와 옵션 |
| [설정 레퍼런스](https://devdna.github.io/rag-factory/configuration.html) | `project.yaml` 필드 정리 |

## 시스템 요구사항

- **Python** 3.11+ · **Ollama** — [ollama.com](https://ollama.com)
- **GPU** — NVIDIA CUDA (8GB+) 또는 Apple Silicon (MPS) 권장. CPU 폴백 가능(느림).
- **메모리** — 24GB 통합 메모리 기준 `qwen3.5:9b` 단일 모델 안정. 16GB 이하는 모든 슬롯 `qwen3.5:4b`로 다운그레이드.

## 라이선스

추후 결정
