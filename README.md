<div align="center">

# slm-factory

### 도메인 문서 → 명령어 하나 → AI 채팅 서비스

도메인 문서만 넣으면 **RAG 웹 채팅 서비스**를 즉시 구축합니다.<br>
문서가 쌓이면 **SLM 파인튜닝**으로 경량 모델까지 자동 생산합니다.

<br>

[사용 가이드](https://devdna.github.io/slm-factory/guide.html) · [CLI 레퍼런스](https://devdna.github.io/slm-factory/cli-reference.html) · [RAG 서비스 가이드](https://devdna.github.io/slm-factory/integration-guide.html)

</div>

<br>

## 두 가지 서비스 모드

```bash
# 문서 1건부터 — RAG 웹 채팅 즉시 시작 (30초)
slm-factory serve

# 문서 20건+ — 파인튜닝 + RAG 통합 서비스 (30분)
slm-factory run
```

| 모드 | 명령어 | 문서 수 | 소요 시간 | 서비스 모델 |
|------|--------|---------|-----------|------------|
| **RAG 전용** | `slm-factory serve` | 1건부터 | 30초 | Teacher (9B) |
| **파인튜닝 + RAG** | `slm-factory run` | 20건+ 권장 | 30분 | Student (파인튜닝) + RAG |

## 빠른 시작

### 1. 설치

```bash
git clone https://github.com/DevDnA/slm-factory.git
cd slm-factory
./setup.sh
```

> 설치 완료 후 가상환경을 활성화하세요:
> - **bash/zsh**: `source .venv/bin/activate`
> - **fish**: `source .venv/bin/activate.fish`
> - **Windows**: `.venv\Scripts\activate`

### 2. 프로젝트 생성 + 문서 추가

```bash
slm-factory init my-project
cp /path/to/documents/*.pdf my-project/documents/
```

> PDF, HWP, HWPX, DOCX, HTML, TXT, MD — 7개 포맷을 자동 감지합니다.

### 3. 서비스 시작

```bash
slm-factory serve
```

브라우저에서 **http://localhost:8000/chat** 접속

## 어떻게 동작하는가

### RAG 서비스 (slm-factory serve)

```
문서 → 섹션 인식 청킹 → 벡터 임베딩(bge-m3) → ChromaDB
                                                    ↓
사용자 질문 → 벡터 검색 → 관련 문서 참조 → LLM 답변 생성 → 웹 채팅
```

### 파인튜닝 파이프라인 (slm-factory run)

```
문서 → Teacher(9B)가 QA 자동 생성 → 검증/채점 → 증강
  → Student(1B) LoRA 학습 → Ollama 모델 등록 → 평가 → RAG 인덱싱 → 웹 채팅
```

## 핵심 기술

| 기능 | 설명 |
|------|------|
| **섹션 인식 청킹** | 문서의 논리적 구조(장/절/항)를 자동 감지하여 분할 |
| **부모-자식 청킹** | 작은 청크로 정밀 검색, 큰 부모 청크로 충분한 맥락 전달 |
| **컨텍스트 프리픽스** | 모든 청크에 섹션 계층 정보 자동 추가 |
| **Auto 캘리브레이션** | 문서 특성에 따라 chunk_size, 질문 수 자동 최적화 |
| **Magic bytes 감지** | 파일 확장자가 아닌 실제 내용으로 포맷 자동 판별 |
| **Teacher-Student 증류** | 대형 모델이 생성한 QA로 소형 모델 학습 |
| **웹 채팅 UI** | SSE 스트리밍, 참조 문서 표시, 한국어 IME 지원 |

## 무엇을 해결하는가

| 문제 | slm-factory |
|------|-------------|
| 범용 LLM은 도메인을 모른다 | 도메인 문서 기반 RAG — 문서에 있는 내용만 답변 |
| LLM API 비용이 계속 발생 | 로컬 실행, 외부 API 호출 없음 |
| 사내 문서가 외부로 유출 | 온프레미스 완전 격리, 데이터 유출 제로 |
| 할루시네이션 | RAG 검색 근거 기반 답변, 없는 정보는 "없다"고 답변 |
| RAG 품질이 낮다 | 섹션 인식 + 부모-자식 청킹 + 컨텍스트 프리픽스 |
| 파인튜닝이 어렵다 | 명령어 하나로 자동화 (문서 20건+ 필요) |

## 문서

> **[devdna.github.io/slm-factory](https://devdna.github.io/slm-factory/)**

| 문서 | 내용 |
|------|------|
| [사용 가이드](https://devdna.github.io/slm-factory/guide.html) | 설치, 서비스 모드, 트러블슈팅 |
| [RAG 서비스 가이드](https://devdna.github.io/slm-factory/integration-guide.html) | RAG 구축, API, 프로덕션 배포 |
| [빠른 참조](https://devdna.github.io/slm-factory/quick-reference.html) | 명령어 치트시트 |
| [CLI 레퍼런스](https://devdna.github.io/slm-factory/cli-reference.html) | 전체 명령어 옵션 |
| [설정 레퍼런스](https://devdna.github.io/slm-factory/configuration.html) | project.yaml 전체 설정 |
| [아키텍처](https://devdna.github.io/slm-factory/architecture.html) | 기술 설계, 데이터 흐름 |

## 시스템 요구사항

- **Python** 3.11+
- **Ollama** — [ollama.com](https://ollama.com)
- **GPU** — Apple Silicon (MPS) / NVIDIA CUDA (8GB+) / CPU 폴백

## 라이선스

추후 결정
