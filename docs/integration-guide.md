# 기술 확장 가이드

SLM + RAG로 도메인 AI 서비스를 구축하는 종합 가이드

---

## 1. 개요

### SLM 단독 사용의 한계

slm-factory로 만든 도메인 특화 SLM은 특정 분야에서 뛰어난 성능을 발휘하지만, 모든 상황을 완벽하게 대응하기는 어렵습니다.

- **할루시네이션(Hallucination)**: 학습 데이터에 포함되지 않은 질문에 대해 그럴듯하지만 부정확한 답변을 생성할 수 있습니다.
- **최신 정보 부재**: 파인튜닝 시점 이후에 추가되거나 변경된 정보를 반영할 수 없습니다.
- **근거 제시 불가**: 답변의 출처를 명시할 수 없습니다.
- **문서 간 관계 누락**: 각 문서를 개별 처리하므로, 서로 다른 문서에 걸친 개념의 연결을 파악하지 못합니다.

### RAG가 보완하는 방법

**RAG(Retrieval-Augmented Generation)**는 사용자 질문에 관련 문서를 검색하여 컨텍스트로 제공합니다. 할루시네이션을 억제하고, 최신 정보를 반영하며, 출처를 명시할 수 있습니다.

> **비유**: SLM은 **전문가의 뇌**, RAG는 **참고 도서관**입니다. 전문가가 도서관에서 근거를 찾아 답변하면 — 그것이 SLM + RAG 조합입니다.

---

## 2. 핵심 기술 소개

### AutoRAG란?

[AutoRAG](https://github.com/Marker-Inc-Korea/AutoRAG)는 RAG 파이프라인의 각 단계별 최적 전략 조합을 자동으로 탐색하는 오픈소스 프레임워크입니다.

RAG 파이프라인은 검색(Retrieval), 리랭킹(Reranking), 프롬프트(Prompt), 생성(Generation) 단계로 구성됩니다. AutoRAG는 이 모든 단계의 전략 조합을 자동으로 벤치마킹하여, 주어진 도메인 데이터에 최적화된 RAG 파이프라인 설정을 도출합니다.

| 항목 | slm-factory | AutoRAG |
|------|-------------|---------|
| **목적** | 도메인 특화 SLM 생성 | 최적 RAG 파이프라인 탐색 |
| **접근법** | Teacher-Student 지식 증류 | RAG 전략 벤치마크 |
| **입력** | 도메인 문서 | 도메인 문서 + QA 데이터셋 |
| **출력** | 파인튜닝된 SLM (Ollama) | 최적화된 RAG 파이프라인 + 즉시 배포 가능한 API 서버 |
| **문서 활용** | QA 학습 데이터 생성 | 검색 코퍼스 (벡터 DB) |

---

## 3. 활용 패턴

### 패턴 1: SLM 단독

| 항목 | 내용 |
|------|------|
| **장점** | 빠른 응답, 오프라인 동작, 프라이버시 보장, 낮은 운영 비용 |
| **단점** | 학습 시점 이후 정보 부재, 할루시네이션 가능, 출처 제시 불가 |
| **적합한 경우** | 변하지 않는 도메인 지식, 오프라인 환경, 단순 Q&A |
| **slm-factory 설정** | 기본 파이프라인 (추가 설정 불필요) |

### 패턴 2: SLM + RAG (권장)

| 항목 | 내용 |
|------|------|
| **장점** | 도메인 이해 + 최신 정보 + 출처 제시, 가장 널리 검증된 조합 |
| **단점** | RAG 서버 운영 필요, 검색 지연 추가 |
| **적합한 경우** | 대부분의 도메인 AI 시스템 |
| **slm-factory 설정** | [섹션 4 (RAG 서비스 가이드)](#4-rag-서비스-가이드) 참조 |

### 패턴 선택 기준

| 질문 | 예 | 아니오 |
|------|-----|--------|
| 문서가 자주 변경되거나 출처 제시가 필요한가? | **패턴 2** (SLM + RAG) | 패턴 1 (SLM 단독) |
| 오프라인·격리 환경에서만 운영하는가? | 패턴 1 (SLM 단독) | **패턴 2** (SLM + RAG) |

**권장**: 대부분의 경우 패턴 2로 시작하십시오.

---

## 4. RAG 서비스 가이드

slm-factory는 AutoRAG 데이터 포맷을 활용하여 도메인 특화 RAG 서비스를 구축합니다. export-autorag로 데이터를 준비하고 내장 RAG로 즉시 서비스하거나, AutoRAG 최적화를 거쳐 프로덕션으로 확장합니다.

```
export-autorag → corpus.parquet → rag-index → ChromaDB → rag-serve → REST API
                                                            ↑ (선택) AutoRAG 최적화
```

### 4.1 데이터 준비

```bash
# 1. slm-factory 파이프라인 실행 (파싱 + QA 생성)
uv run slm-factory run --config project.yaml

# 2. AutoRAG 평가용 데이터 내보내기
uv run slm-factory tool export-autorag --config project.yaml

# 결과물:
#   output/autorag/corpus.parquet  — 문서 청크 (검색 대상)
#   output/autorag/qa.parquet      — QA 평가 데이터
```

코퍼스 데이터 (`corpus.parquet`):

| 컬럼 | 타입 | 설명 |
|------|------|------|
| `doc_id` | `str` | 문서 청크 고유 ID |
| `contents` | `str` | 청크 텍스트 내용 |
| `metadata` | `dict` | 메타데이터 |

평가용 QA 데이터 (`qa.parquet`):

| 컬럼 | 타입 | 설명 |
|------|------|------|
| `qid` | `str` | 질문 고유 ID |
| `query` | `str` | 질문 텍스트 |
| `retrieval_gt` | `list[list[str]]` | 정답 근거 문서 ID 목록 |
| `generation_gt` | `list[str]` | 정답 텍스트 목록 |

### 4.2 내장 RAG 서빙 — 즉시 시작

slm-factory에 내장된 **ChromaDB 인덱싱 + FastAPI 서빙**으로, AutoRAG 없이도 즉시 RAG 서비스를 구동할 수 있습니다.

```bash
# 1. corpus.parquet 생성 (4.1의 export-autorag 실행 후)
uv run slm-factory tool export-autorag --config project.yaml

# 2. ChromaDB에 벡터 임베딩 적재
uv run slm-factory tool rag-index --config project.yaml

# 3. RAG API 서버 실행
uv run slm-factory tool rag-serve --config project.yaml
# → POST http://localhost:8000/v1/query       질의 엔드포인트
# → GET  http://localhost:8000/health          기본 헬스체크
# → GET  http://localhost:8000/health/ready    Ollama+ChromaDB 연결 확인
# → GET  http://localhost:8000/health/live     라이브니스 체크
```

> **팁**: `uv run slm-factory run --serve --config project.yaml` 명령으로 전체 파이프라인 실행 후 RAG 서버까지 한 번에 시작할 수 있습니다.

```bash
# API 호출 테스트
curl -X POST http://localhost:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "도메인 특화 질문", "top_k": 5}'

# SSE 스트리밍 (토큰 단위 실시간 전송)
curl -N -X POST http://localhost:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "도메인 특화 질문", "stream": true}'
```

**동작 방식:**
- `rag-index`: `corpus.parquet` 문서를 `BAAI/bge-m3`로 임베딩 → ChromaDB에 적재
- `rag-serve`: 질의 임베딩 → ChromaDB 유사도 검색 → Ollama SLM 생성 → JSON 응답
- `stream: true` 요청 시 SSE(Server-Sent Events)로 토큰을 실시간 전송 (TTFT < 0.5초)

**질의 → 응답 흐름:**

```
사용자 질의: "이 규정의 예외 조항은?"
       │
       ▼
  ① POST /v1/query ──────── FastAPI (rag-serve)
       │
       ▼
  ② 질의 임베딩 ─────────── bge-m3로 벡터 변환
       │
       ▼
  ③ ChromaDB 검색 ────────── cosine 유사도 top_k개 문서 청크 반환
       │
       ▼
  ④ 프롬프트 조합 ────────── 시스템 지시 + 검색 문서 + 질문 결합
       │                     "다음 문서를 참고하여 답변하십시오.
       │                      {검색된 문서 청크들}
       │                      질문: 이 규정의 예외 조항은?"
       ▼
  ⑤ Ollama SLM 생성 ──────── 도메인 특화 SLM이 컨텍스트 기반 답변 생성
       │
       ▼
  ⑥ JSON 응답
       {
         "answer": "제3조에 따르면 예외 조항은...",
         "sources": [
           {"content": "제3조 예외...", "doc_id": "chunk_03", "score": 0.92},
           {"content": "관련 규정...", "doc_id": "chunk_07", "score": 0.85}
         ],
         "query": "이 규정의 예외 조항은?"
       }
```

**적합한 경우:** PoC, 데모, 소규모 사내 서비스. AutoRAG 최적화 과정 없이 즉시 시작하고 싶을 때.

> **팁**: 검색 품질 최적화가 필요하면 4.3절(AutoRAG)으로 최적 검색·리랭킹·생성 조합을 탐색하십시오.

### 4.3 AutoRAG 최적화 (선택)

내장 RAG로 충분하지만, 검색 품질 최적화가 필요하면 AutoRAG로 최적 조합을 탐색할 수 있습니다.

```bash
uv pip install autorag

# 최적화 실행 (검색·리랭킹·생성 조합 자동 탐색)
autorag evaluate \
  --qa_data_path output/autorag/qa.parquet \
  --corpus_data_path output/autorag/corpus.parquet \
  --config autorag_config.yaml

# 결과 확인
autorag dashboard --trial_dir ./benchmark/0
```

**한국어 최적화 권장 컴포넌트:**

| 단계 | 권장 모듈 | 설명 |
|------|-----------|------|
| Retrieval | `bm25` + `vectordb` | BM25(형태소)와 벡터 검색 하이브리드 |
| Embedding | `BAAI/bge-m3` | 다국어 임베딩 (한국어 성능 우수) |
| Reranking | `Dongjin-kr/ko-reranker` | 한국어 특화 리랭커 |
| Tokenizer | `ko_kiwi` | 한국어 형태소 분석 (BM25 토크나이저) |

AutoRAG YAML에서 slm-factory 모델을 연결할 때는 `export.ollama.model_name` 값을 그대로 사용합니다.

```yaml
# autorag-config.yaml
node_lines:
  - node_line_name: retrieve_node_line
    nodes:
      - node_type: semantic_retrieval
        strategy:
          metrics: [retrieval_f1, retrieval_recall, retrieval_precision]
        top_k: 3
        modules:
          - module_type: vectordb
            vectordb: domain_chroma

  - node_line_name: post_retrieve_node_line
    nodes:
      - node_type: prompt_maker
        strategy:
          metrics: [meteor, rouge, bert_score]
        modules:
          - module_type: fstring
            prompt: "다음 문서를 참고하여 질문에 답변하세요.\n\n질문: {query}\n\n참고 문서:\n{retrieved_contents}\n\n답변: "

      - node_type: generator
        strategy:
          metrics: [bleu, rouge, bert_score]
        modules:
          - module_type: llama_index_llm
            llm: ollama
            model: my-domain-model    # slm-factory의 export.ollama.model_name과 동일
            batch: 1
            request_timeout: 100
```

> **핵심 연결 포인트**: slm-factory의 `export.ollama.model_name: "my-domain-model"` 값이 AutoRAG YAML의 `model: my-domain-model`과 일치해야 합니다. Ollama가 두 시스템의 다리 역할을 합니다.

AutoRAG 최적화가 완료되면, 별도 개발 없이 **즉시 RAG API 서비스를 배포**할 수 있습니다. AutoRAG 내장 Quart 서버가 최적화 결과를 그대로 서빙합니다.

```bash
# 로컬 실행 — 이것만으로 RAG 서비스가 동작합니다
autorag run_api \
  --trial_dir ./benchmark/0 \
  --host 0.0.0.0 --port 8000

# Docker 실행 — 배포 환경에서도 동일하게 동작합니다
docker run -p 8000:8000 \
  -v $(pwd)/benchmark:/app/benchmark \
  autoraghq/autorag:api-latest \
  run_api --trial_dir /app/benchmark/0
```

**이 시점에서 무엇이 동작하는가:**
- 사용자 질문을 받으면 벡터 검색 → 리랭킹 → SLM 생성까지 자동 실행
- 최적화 과정에서 선택된 최적 검색·생성 전략이 그대로 적용됨
- REST API를 통해 외부 시스템(웹앱, 챗봇, 내부 도구)에서 호출 가능

**적합한 경우:** 사내 도구, 소규모 팀 서비스, 부서 단위 AI 어시스턴트 (동시 사용자 ~10명)

### 4.4 FastAPI 프로덕션 서버 (선택)

AutoRAG 최적화 결과를 분석하여 **최적 조합을 직접 구현**하는 방식입니다.

```
┌─────────────────────────────────────────────────┐
│              FastAPI Application                │
│                                                 │
│  /v1/query ──► Retriever ──► Reranker ──► LLM   │
│                  │              │           │    │
│              ChromaDB     ko-reranker   Ollama   │
│              (bge-m3)                   (SLM)    │
└─────────────────────────────────────────────────┘
```

**구현 단계:**

1. **AutoRAG 최적화 결과 분석** — `summary.csv`에서 최적 retriever/reranker/generator 조합 확인
2. **벡터 DB 구축** — `corpus.parquet`의 청크를 `bge-m3`로 임베딩 → ChromaDB 적재
3. **FastAPI 서버 구현** — 검색 → 리랭킹 → SLM 생성 파이프라인
4. **SLM 연동** — slm-factory로 학습한 모델을 Ollama로 서빙, FastAPI에서 호출

```python
# 최소 구조 예시
from fastapi import FastAPI
import chromadb
import httpx  # Ollama 호출용

app = FastAPI()
chroma = chromadb.PersistentClient(path="./chroma_db")
collection = chroma.get_collection("domain_docs")

@app.post("/v1/query")
async def query(request: QueryRequest):
    # 1. 벡터 검색
    results = collection.query(
        query_texts=[request.query], n_results=10
    )
    # 2. 리랭킹 (ko-reranker)
    reranked = rerank(request.query, results)
    # 3. SLM 생성 (Ollama)
    context = "\n".join(reranked[:3])
    answer = await call_ollama(request.query, context)
    return {"answer": answer, "sources": reranked[:3]}
```

**적합한 경우:** 프로덕션 배포, 커스텀 로직 필요, 높은 동시성 요구

### 4.5 구축 단계 요약

```
Phase 1: slm-factory (SLM 학습 + Ollama 배포)
    ↓
Phase 2: export-autorag → corpus.parquet 생성
    ↓
Phase 3: RAG 서비스 운영
    ├─ A: slm-factory 내장 RAG (즉시 시작)
    ├─ B: AutoRAG 내장 서버 (최적화 탐색)
    └─ C: FastAPI 커스텀 서버 (대규모 확장)
```

| 단계 | 기간 | 목표 | 판단 기준 |
|------|------|------|-----------|
| Phase 1 | 1-2주 | SLM 학습 완료 | BLEU ≥ 0.3, ROUGE-L ≥ 0.4 |
| Phase 2 | 1주 | RAG 파이프라인 최적화 | Retrieval MRR ≥ 0.7 |
| Phase 3 | 1-2주 | RAG 서비스 배포 | API 응답 품질 + 지연 시간 |

> **Phase 3 시점에서 실체적인 RAG 서비스가 동작합니다.** 경로 A라면 `uv run slm-factory tool rag-serve` 한 줄로, 경로 B라면 `autorag run_api` 한 줄로 REST API를 제공하는 도메인 AI 서비스가 완성됩니다.

### 4.6 프로덕션 보완 체크리스트

slm-factory 내장 RAG 또는 AutoRAG 서버로 RAG 서비스를 즉시 운영할 수 있지만, **엔터프라이즈 프로덕션 환경**에서는 다음 항목을 보완해야 합니다.

#### 즉시 보완 (서비스 공개 전)

| 항목 | 설명 | 보완 방법 |
|------|------|-----------|
| **HTTPS/TLS** | API 통신 암호화 | Nginx/Caddy 리버스 프록시로 TLS 종단. `certbot`으로 인증서 자동 발급 |
| **인증/인가** | 허가된 사용자만 접근 | API 키 기반: Nginx `auth_request` 또는 FastAPI 미들웨어. 기업 환경: SSO/LDAP 연동 |
| **입력 검증** | 프롬프트 인젝션 방어 | 질의 길이 제한 (예: 2,000자), 금칙어 필터링, 시스템 프롬프트 고정 |
| **헬스체크** | 서비스 상태 모니터링 | 내장 `/health`(기본), `/health/ready`(Ollama+ChromaDB 연결), `/health/live`(라이브니스 체크, `{"status": "alive"}` 응답) 엔드포인트 제공. 프로덕션에서는 `/health/ready`를 로드밸런서 헬스체크에, `/health/live`를 Kubernetes 라이브니스 프로브에 사용 |

#### 운영 안정화 (서비스 공개 후)

| 항목 | 설명 | 보완 방법 |
|------|------|-----------|
| **요청 제한** | 과부하 방지 | Nginx `limit_req_zone` 또는 FastAPI `slowapi`. 사용자별 분당 요청 수 제한 |
| **로깅** | 질의/응답 추적 | 구조화 로깅 (JSON). 질의, 검색된 문서 ID, 응답 시간, 에러를 기록 |
| **모니터링** | 성능·품질 추적 | Prometheus + Grafana: 응답 지연(p50/p95/p99), 에러율, Ollama 추론 시간 |
| **자동 재시작** | 장애 복구 | systemd `Restart=always` 또는 Docker `restart: unless-stopped` |
| **백업** | 데이터 보호 | 벡터 DB(ChromaDB) 정기 백업. AutoRAG trial 디렉토리 버전 관리 |

#### 프로덕션 확장 (사용자 증가 시)

| 항목 | 설명 | 보완 방법 |
|------|------|-----------|
| **로드 밸런서** | 수평 확장 | 서버 인스턴스 다중 구동 + Nginx upstream. 또는 FastAPI 전환 (경로 C) |
| **GPU 추론 최적화** | 높은 동시성 | Ollama 대신 vLLM 또는 TGI(Text Generation Inference)로 전환. 배치 추론 지원 |
| **응답 캐싱** | 반복 질의 최적화 | Redis 캐시. 동일 질의 재처리 방지 (TTL 기반) |
| **벡터 DB 확장** | 대규모 코퍼스 | ChromaDB → Milvus 또는 Weaviate. 분산 인덱싱, 수백만 문서 지원 |
| **모델 버전 관리** | SLM 업데이트 | `tool evolve`로 SLM 갱신 시, Ollama 모델명에 버전 태그 부여 (`v1`, `v2`). Blue-Green 배포로 무중단 전환 |

#### 전환 판단 기준: 경로 A → B → C

slm-factory 내장 RAG(경로 A) 또는 AutoRAG 서버(경로 B)에서 FastAPI 커스텀 서버(경로 C)로 전환해야 하는 시점:

| 신호 | 의미 |
|------|------|
| 동시 사용자 10명 이상에서 응답 지연 증가 | 내장 서버의 동시성 한계 |
| 인증·권한 로직이 복잡해짐 | 내장 서버에 미들웨어 추가 어려움 |
| 검색 결과 후처리 로직 필요 | 커스텀 비즈니스 로직 삽입 필요 |
| 멀티 모델 라우팅 필요 | 질문 유형별 다른 SLM 호출 |

경로 A에서 시작하여, 검색 품질 최적화가 필요하면 경로 B(AutoRAG)로, 위 신호가 나타나면 경로 C(FastAPI)로 전환하십시오. AutoRAG 최적화 결과(`summary.csv`)의 최적 조합 정보를 그대로 FastAPI 구현에 반영하면 됩니다.

### 4.7 운영 팁

- **문서 분리 전략**: 핵심 도메인 지식은 SLM 학습에, 세부 참조 자료는 RAG 검색 코퍼스에 활용합니다.
- **모델 크기**: RAG와 함께 사용하면 1B~3B 소형 모델로도 충분합니다. SLM이 모든 지식을 기억할 필요 없이 "검색된 문서 이해"에 집중하면 됩니다.
- **시스템 프롬프트 통일**: slm-factory 학습 시 사용한 시스템 프롬프트와 AutoRAG 생성 시 프롬프트를 일관되게 유지하십시오.
- **업데이트**: slm-factory의 `tool evolve`로 SLM을 진화시키고, RAG 코퍼스에 새 문서를 추가하여 양쪽을 동시에 갱신합니다.

---

## 5. 활용 시나리오

- **사내 규정 Q&A**: 규정 용어를 이해하는 SLM + 최신 규정 문서를 검색하는 RAG. 정확한 조항과 출처를 제시합니다.
- **장애 대응 AI**: 장애 패턴을 인식하는 SLM + 유사 장애 사례와 복구 절차서를 검색하는 RAG.
- **의료/법률 자문**: 전문 용어를 정확히 사용하는 SLM + 관련 조문/판례 원문을 검색하는 RAG.

---

## 6. 주의사항

- **단순한 문서 세트**: FAQ나 단일 매뉴얼에서는 RAG 추가 효과가 미미합니다. 패턴 1(SLM 단독)로 충분합니다.
- **검색 품질 의존**: RAG 응답 품질은 검색 정확도에 직접 의존합니다. 코퍼스 청킹 전략과 임베딩 모델 선택이 중요합니다.
- **추가 비용**: RAG 검색과 벡터 DB 운영에 처리 시간과 인프라 비용이 추가됩니다.
- **운영 복잡도**: SLM + RAG 조합은 두 시스템을 함께 운영해야 합니다. 팀 역량과 인프라를 고려하십시오.

---

## 7. 관련 문서

| 문서 | 내용 |
|------|------|
| [사용 가이드](guide.html) | slm-factory 설치부터 모델 배포까지 단계별 안내 |
| [설정 레퍼런스](configuration.html) | `rag`, `export.ollama` 설정 상세 |
| [CLI 레퍼런스](cli-reference.html) | `tool rag-index`, `tool rag-serve`, `tool wizard` 명령어 상세 |
| [아키텍처 가이드](architecture.html) | 모듈 내부 구조와 설계 원칙 |
| [개발 가이드](development.html) | 모듈 확장 방법 |
| [AutoRAG GitHub](https://github.com/Marker-Inc-Korea/AutoRAG) | AutoRAG 공식 저장소 |


