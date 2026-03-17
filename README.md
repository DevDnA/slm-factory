<div align="center">

# slm-factory

**도메인 문서만 넣으면 AI 채팅 서비스가 만들어집니다.**

문서 넣고, 명령어 하나, 30초 만에 RAG 웹 채팅.<br>
문서가 쌓이면 SLM 파인튜닝까지 자동으로.

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![Ollama](https://img.shields.io/badge/Ollama-black?logo=ollama&logoColor=white)](https://ollama.com)

[문서 홈](https://devdna.github.io/slm-factory/) · [사용 가이드](https://devdna.github.io/slm-factory/guide.html) · [CLI 레퍼런스](https://devdna.github.io/slm-factory/cli-reference.html) · [RAG 서비스 가이드](https://devdna.github.io/slm-factory/integration-guide.html)

</div>

<br>

## 왜 필요한가

범용 LLM은 우리 조직의 문서를 모릅니다. ChatGPT에 사내 규정을 물어보면 그럴듯한 거짓말을 합니다.

slm-factory는 **도메인 문서를 넣으면 그 문서만으로 답변하는 AI 채팅 서비스**를 만듭니다. 외부 API 호출이 없고, 사내 문서가 밖으로 나가지 않으며, 문서에 없는 내용은 "없다"고 답합니다.

## 빠른 시작

```bash
git clone https://github.com/DevDnA/slm-factory.git
cd slm-factory && ./setup.sh

slf init my-project
cp /path/to/documents/*.pdf my-project/documents/

slf rag
```

브라우저에서 **http://localhost:8000/chat** — 채팅이 시작됩니다.

> `setup.sh`가 Python 환경, 의존성, Ollama 모델을 한 번에 준비합니다. PDF, HWP, HWPX, DOCX, HTML, TXT, MD 7개 포맷을 자동 감지합니다.

## 세 가지 패턴

| 패턴 | 명령어 | LLM | 적합한 경우 |
|------|--------|-----|-------------|
| **RAG + 베이스 모델** | `slf rag` | Teacher(9B) — 파인튜닝 없이 즉시 | 문서 20건 미만, 빠른 검증 |
| **RAG + 파인튜닝 SLM** | `slf tune` | Student(1B) — 9배 빠르고 1/9 비용 | 문서 20건+, 프로덕션 |
| **파인튜닝 SLM 단독** | `slf tune --no-chat` | Student(1B) — RAG 없이 모델만 | 오프라인, 변하지 않는 지식 |

```bash
slf rag                # RAG + Teacher 즉시 시작 (30초)
slf tune               # 파인튜닝 + RAG + 채팅 (30분)
slf tune --no-chat     # 파인튜닝만 (Ollama에 모델 등록)
```

> `slf rag`는 파인튜닝된 Student 모델이 Ollama에 있으면 자동으로 사용합니다. 패턴 1에서 시작하고, 문서가 쌓이면 패턴 2로 자연스럽게 전환됩니다.

### RAG 서비스 — `slf rag`

```
문서 ─→ 섹션 인식 청킹 ─→ 벡터 임베딩 ─→ Qdrant 인덱스
                                              │
질문 ─→ 하이브리드 검색(벡터+BM25) ─→ 리랭킹 ─→ LLM 답변 ─→ 웹 채팅
```

### 파인튜닝 — `slf tune`

```
문서 ─→ Teacher(9B)가 QA 생성 ─→ 검증/채점/증강
  ─→ Student(1B) LoRA 학습 ─→ 평가 ─→ Ollama 등록 ─→ RAG + 웹 채팅
```

> 두 모드 모두 `--chat` 옵션으로 완료 후 웹 채팅을 자동 시작합니다. 상세 비교는 [사용 가이드](https://devdna.github.io/slm-factory/guide.html#7-지식-증류와-rag--언제-무엇을-쓸-것인가)를 참고하세요.

## 주요 기능

**섹션 인식 청킹** — 문서의 장/절/항 구조를 자동 감지하여 논리적 단위로 분할합니다. 작은 청크로 정밀 검색하고, 부모 청크로 충분한 맥락을 전달합니다.

**하이브리드 검색 + 리랭킹** — 벡터 검색과 BM25 키워드 검색을 RRF로 결합하고, Cross-Encoder로 최종 순위를 재조정합니다. Lost-in-the-middle 재정렬로 LLM이 관련 문서를 놓치지 않습니다.

**자동 캘리브레이션** — 문서 길이, 밀도, 구조를 분석하여 chunk_size, 에포크 수, 청크당 질문 수를 자동 결정합니다. 설정 없이 바로 실행해도 합리적인 결과를 냅니다.

**Teacher-Student 증류** — 9B Teacher 모델이 문서 기반 QA를 자동 생성하고, 검증·채점·증강을 거쳐 1B Student 모델을 학습시킵니다. 13단계 파이프라인이 명령어 하나로 실행됩니다.

## 문서

> **[devdna.github.io/slm-factory](https://devdna.github.io/slm-factory/)**

| | |
|---|---|
| [사용 가이드](https://devdna.github.io/slm-factory/guide.html) | 설치부터 모델 배포까지 단계별 안내 |
| [빠른 참조](https://devdna.github.io/slm-factory/quick-reference.html) | 명령어·설정 치트시트 |
| [CLI 레퍼런스](https://devdna.github.io/slm-factory/cli-reference.html) | 전체 명령어 옵션 |
| [설정 레퍼런스](https://devdna.github.io/slm-factory/configuration.html) | project.yaml 전체 설정 |
| [RAG 서비스 가이드](https://devdna.github.io/slm-factory/integration-guide.html) | RAG 구축·API·프로덕션 배포 |
| [아키텍처](https://devdna.github.io/slm-factory/architecture.html) | 기술 설계·데이터 흐름 |

## 시스템 요구사항

- **Python** 3.11+ · **Ollama** — [ollama.com](https://ollama.com)
- **GPU** — NVIDIA CUDA (8GB+) 또는 Apple Silicon (MPS) 권장. CPU 폴백 가능(느림).

## 라이선스

추후 결정
