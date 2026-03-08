# 기술 확장 가이드

SLM을 넘어서: RAG와 온톨로지로 도메인 AI를 확장하는 종합 가이드

---

## 1. 개요

### SLM 단독 사용의 한계

slm-factory로 만든 도메인 특화 SLM은 특정 분야에서 뛰어난 성능을 발휘하지만, 모든 상황을 완벽하게 대응하기는 어렵습니다.

- **할루시네이션(Hallucination)**: 학습 데이터에 포함되지 않은 질문에 대해 그럴듯하지만 부정확한 답변을 생성할 수 있습니다.
- **최신 정보 부재**: 파인튜닝 시점 이후에 추가되거나 변경된 정보를 반영할 수 없습니다.
- **근거 제시 불가**: 답변의 출처를 명시할 수 없습니다.
- **문서 간 관계 누락**: 각 문서를 개별 처리하므로, 서로 다른 문서에 걸친 개념의 연결을 파악하지 못합니다.

### 확장 기술이 보완하는 방법

**RAG(Retrieval-Augmented Generation)**는 사용자 질문에 관련 문서를 검색하여 컨텍스트로 제공합니다. 할루시네이션을 억제하고, 최신 정보를 반영하며, 출처를 명시할 수 있습니다.

**온톨로지(Ontology)**는 문서에서 엔티티와 관계를 추출하여 지식 그래프를 구성합니다. QA 생성 시 문서 간 관계를 인식하여 더 깊이 있는 학습 데이터를 만듭니다.

> **비유**: SLM은 **전문가의 뇌**이고, RAG는 **참고 도서관**이고, 온톨로지는 **마인드맵**입니다. 전문가가 마인드맵으로 지식 체계를 정리한 뒤, 도서관에서 근거를 찾아 답변하면 — 그것이 SLM + RAG + 온톨로지 조합입니다.

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
| **출력** | 파인튜닝된 SLM (Ollama) | 최적화된 RAG 파이프라인 설정 |
| **문서 활용** | QA 학습 데이터 생성 | 검색 코퍼스 (벡터 DB) |

### 온톨로지란?

온톨로지(지식 그래프)는 정보를 **엔티티(Entity)**와 **관계(Relation)**로 구조화한 것입니다.

- **엔티티**: 문서에서 추출한 핵심 개체 (사람, 조직, 기술, 개념 등)
- **관계**: 엔티티 간의 연결 (예: "LoRA"는 "파인튜닝"의 한 방법이다)
- **트리플**: `(주어, 술어, 목적어)` 형식의 기본 단위

slm-factory는 **LLM 기반 추출 방식**을 사용합니다. 이미 설정된 Teacher LLM이 문서에서 엔티티와 관계를 추출하고, 신뢰도 점수로 필터링하여 `ontology.json`에 저장합니다.

| 항목 | slm-factory 온톨로지 | 전문 KG 도구 (Neo4j 등) |
|------|----------------------|------------------------|
| **목적** | QA 생성 품질 향상 | 범용 지식 관리/쿼리 |
| **추출 방식** | Teacher LLM 기반 | NER + 규칙 + ML 파이프라인 |
| **쿼리** | 불가 (SPARQL 미지원) | 풍부한 그래프 쿼리 |
| **설치 비용** | 추가 설치 없음 | 별도 DB 서버 필요 |
| **사용 난이도** | YAML 설정 한 줄 | 스키마 설계 + 쿼리 학습 |

---

## 3. 확장 패턴별 분석

slm-factory 사용자 관점에서 실제로 의미 있는 4가지 확장 패턴을 분석합니다.

### 패턴 1: SLM 단독

| 항목 | 내용 |
|------|------|
| **장점** | 빠른 응답, 오프라인 동작, 프라이버시 보장, 낮은 운영 비용 |
| **단점** | 학습 시점 이후 정보 부재, 할루시네이션 가능, 출처 제시 불가 |
| **적합한 경우** | 변하지 않는 도메인 지식, 오프라인 환경, 단순 Q&A |
| **slm-factory 설정** | 기본 파이프라인 (추가 설정 불필요) |

### 패턴 2: SLM + RAG

| 항목 | 내용 |
|------|------|
| **장점** | 도메인 이해 + 최신 정보 + 출처 제시, 가장 널리 검증된 조합 |
| **단점** | 두 시스템 운영 복잡도, 검색 지연 추가 |
| **적합한 경우** | 대부분의 도메인 AI 시스템 (가장 범용적 조합) |
| **slm-factory 설정** | slm-factory + AutoRAG 연동 (섹션 4 참조) |

### 패턴 3: SLM + 온톨로지

| 항목 | 내용 |
|------|------|
| **장점** | 관계 인식 학습 데이터, 더 깊이 있는 QA, 체계적 지식 내재화 |
| **단점** | 온톨로지 품질이 학습 데이터에 직접 영향, 추출 비용 |
| **적합한 경우** | 개념 간 관계가 중요한 도메인 (법률, 의료, 기술 표준) |
| **slm-factory 설정** | `ontology.enrich_qa: true` (섹션 5 참조) |

### 패턴 4: SLM + RAG + 온톨로지

| 항목 | 내용 |
|------|------|
| **장점** | 최고 수준의 도메인 AI — 지식 내재화 + 실시간 검색 + 구조적 이해 |
| **단점** | 최고 수준의 복잡도, 3개 시스템 운영/유지보수 |
| **적합한 경우** | 대규모 엔터프라이즈 지식 관리, 복잡한 도메인 (금융 규제, 의료 프로토콜) |
| **slm-factory 설정** | slm-factory (온톨로지 활성화) + AutoRAG 연동 |

### slm-factory 범위 밖의 패턴

다음 패턴은 slm-factory의 핵심 범위 밖이지만, 참고로 정리합니다.

| 패턴 | 설명 | 비고 |
|------|------|------|
| **범용 LLM + RAG** | 파인튜닝 없이 범용 모델에 RAG만 적용 | 도메인 용어 이해가 약함, 빠른 프로토타입에 적합 |
| **온톨로지 단독** | 지식 그래프만 추출, SLM 학습에 미반영 | `tool ontology`로 독립 실행 가능, 데이터 감사 목적 |
| **RAG + 온톨로지 (GraphRAG)** | 그래프 기반 검색 강화 | 전문 그래프 DB 필요, slm-factory에서 수동 통합 |

### 패턴 선택 가이드

어떤 패턴을 선택해야 할까요? 다음 질문에 답하며 결정하십시오.

1. **문서가 자주 변경되는가?** → 예: RAG 추가 (패턴 2 또는 4)
2. **출처 제시가 필수인가?** → 예: RAG 추가 (패턴 2 또는 4)
3. **도메인 전문 용어가 중요한가?** → 예: SLM 필수 (패턴 1~4 모두 해당)
4. **문서 간 관계가 복잡한가?** → 예: 온톨로지 추가 (패턴 3 또는 4)
5. **운영 복잡도를 감당할 수 있는가?** → 아니오: 단순 패턴 선택 (패턴 1 또는 2)

**권장 시작점:**
- **대부분의 경우**: 패턴 2 (SLM + RAG)로 시작
- **관계가 복잡한 도메인**: 패턴 3 (SLM + 온톨로지)로 시작, 필요시 RAG 추가
- **빠른 검증**: 범용 LLM + RAG로 프로토타입 후, SLM 추가

---

## 4. AutoRAG 연동 가이드

### 통합 아키텍처

slm-factory와 AutoRAG의 통합은 하나의 도메인 문서 세트에서 시작하여 두 갈래 경로로 처리됩니다.

- **왼쪽 경로 (slm-factory)**: 도메인 문서에서 QA 학습 데이터를 생성하고, Student 모델을 파인튜닝하여 Ollama에 배포합니다.
- **오른쪽 경로 (AutoRAG)**: 동일한 도메인 문서를 벡터 DB에 색인하고, 최적의 RAG 파이프라인을 자동으로 탐색합니다.
- **수렴 지점 (Ollama)**: `export.ollama.model_name` 설정으로 배포된 모델명이 AutoRAG YAML의 `model:` 값과 동일하면, AutoRAG는 자동으로 slm-factory가 만든 SLM을 사용합니다.

### 단계별 연동

**Step 1: slm-factory로 도메인 SLM 생성**

```bash
slm-factory init my-domain-project
cp /path/to/domain-docs/*.pdf my-domain-project/documents/
slm-factory tool wizard --config my-domain-project/project.yaml
```

```yaml
# project.yaml 설정
export:
  ollama:
    model_name: "my-domain-model"
    system_prompt: "당신은 도메인 전문 AI 어시스턴트입니다."
```

```bash
# 모델 배포
cd my-domain-project/output/merged_model
ollama create my-domain-model -f Modelfile
ollama run my-domain-model "이 도메인에 대해 설명해주세요"
```

**Step 2: AutoRAG 설치 및 데이터 준비**

```bash
pip install autorag
mkdir -p autorag-project/data
```

AutoRAG에 필요한 두 개의 parquet 파일을 준비합니다.

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

**Step 3: AutoRAG YAML에서 slm-factory 모델 연결**

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

**Step 4: AutoRAG 실행**

```bash
autorag evaluate \
  --config autorag-config.yaml \
  --qa_data_path autorag-project/data/qa.parquet \
  --corpus_data_path autorag-project/data/corpus.parquet \
  --project_dir autorag-project/results

# 결과 확인
autorag dashboard --trial_dir autorag-project/results/0
```

**Step 5: 최적 RAG 시스템 배포**

```bash
autorag run_api \
  --config_path autorag-project/best_pipeline.yaml \
  --host 0.0.0.0 --port 8000

# API 호출 테스트
curl -X POST http://localhost:8000/v1/run \
  -H "Content-Type: application/json" \
  -d '{"query": "이 규정의 예외 조항은 무엇입니까?"}'
```

### 운영 팁

- **문서 분리 전략**: 핵심 도메인 지식은 SLM 학습에, 세부 참조 자료는 RAG 검색 코퍼스에 활용합니다.
- **모델 크기**: RAG와 함께 사용하면 1B~3B 소형 모델로도 충분합니다. SLM이 모든 지식을 기억할 필요 없이 "검색된 문서 이해"에 집중하면 됩니다.
- **시스템 프롬프트 통일**: slm-factory 학습 시 사용한 시스템 프롬프트와 AutoRAG 생성 시 프롬프트를 일관되게 유지하십시오.
- **업데이트**: slm-factory의 `tool evolve`로 SLM을 진화시키고, AutoRAG 코퍼스에 새 문서를 추가하여 양쪽을 동시에 갱신합니다.

---

## 5. 온톨로지 활용 가이드

### 파이프라인 흐름

온톨로지는 파이프라인에서 **Parse 이후, Generate 이전**에 실행됩니다.

```
문서 → Parse → [Ontology 추출] → Generate (+ 온톨로지 맥락) → Validate → ... → SLM
```

### 단계별 사용법

**Step 1: 온톨로지 활성화**

```yaml
# project.yaml
ontology:
  enabled: true
  entity_types: [Person, Organization, Concept, Technology, Document, Date, Location]
  min_confidence: 0.5
  max_concurrency: 4
  enrich_qa: false         # 처음에는 false로 두고 결과 먼저 확인
  output_file: "ontology.json"
```

**Step 2: 온톨로지 추출 실행**

```bash
slm-factory tool ontology --config my-project/project.yaml
```

**Step 3: 결과 검증**

```bash
cat my-project/output/ontology.json | python -m json.tool | head -50
```

확인 포인트:
- 엔티티가 도메인에 적절한가? (무관한 일반 단어가 많으면 `min_confidence` 올리기)
- 관계가 의미 있는가? (너무 자명한 관계만 있으면 QA에 기여도가 낮음)
- 누락된 핵심 개념이 있는가? (있으면 `entity_types`에 타입 추가 고려)

**Step 4: QA 생성에 반영**

```yaml
ontology:
  enabled: true
  enrich_qa: true    # QA 생성 시 온톨로지 맥락 사용
```

```bash
slm-factory tool wizard --config my-project/project.yaml
```

wizard에서는 **Step 3a (온톨로지 추출)**로 표시되며, Parse 후 자동으로 실행됩니다.

**Step 5: 품질 비교 (선택)**

```bash
# enrich_qa: false로 QA 생성 → qa_without_ontology.json
# enrich_qa: true로 QA 생성 → qa_with_ontology.json
# 두 파일을 비교하여 질문 다양성, 관계 질문 비율 등 확인
```

### slm-factory 온톨로지가 하는 것 / 하지 않는 것

| 하는 것 | 하지 않는 것 |
|---------|-------------|
| LLM 기반 엔티티/관계 추출 | 외부 KG 연동 (Wikidata, DBpedia) |
| 신뢰도 기반 필터링 | SPARQL 쿼리 |
| 엔티티 정규화 (대소문자 무시 병합) | 그래프 시각화 |
| JSON 형식 저장 | 관계 추론 (삼단논법 등) |
| QA 생성 시 맥락 주입 | 그래프 알고리즘 (경로 탐색, 중심성) |
| 증분 업데이트 (변경/삭제 문서 반영) | 퍼지 매칭 (유사 이름 병합) |
| Wizard 통합 (Step 3a) | 스키마 강제 (관계 타입 제약) |

---

## 6. 활용 시나리오

### SLM + RAG 시나리오

- **사내 규정 Q&A**: 규정 용어를 이해하는 SLM + 최신 규정 문서를 검색하는 RAG. 정확한 조항과 출처를 제시합니다.
- **장애 대응 AI**: 장애 패턴을 인식하는 SLM + 유사 장애 사례와 복구 절차서를 검색하는 RAG.
- **의료/법률 자문**: 전문 용어를 정확히 사용하는 SLM + 관련 조문/판례 원문을 검색하는 RAG.

### SLM + 온톨로지 시나리오

- **기술 문서 체계화**: 기술 간 의존성을 파악하여 "A 기술이 B에 미치는 영향" 같은 연결 질문 생성.
- **조직 규정 연결**: "이 규정은 저 규정의 예외 조항이다" 같은 규정 간 관계 반영.
- **학술/연구 자료**: 연구자, 이론, 실험 결과 간의 관계를 추출하여 학술적 질문 생성.

---

## 7. 주의사항

> **"기술을 추가하면 무조건 좋아지는 것은 아닙니다."**

- **단순한 문서 세트**: FAQ나 단일 매뉴얼에서는 온톨로지/RAG 추가의 효과가 미미합니다. 비용만 늘어납니다.
- **Teacher LLM 품질 의존**: 온톨로지 추출 품질은 Teacher LLM 능력에 직접 의존합니다.
- **노이즈 위험**: 품질이 낮은 온톨로지를 QA에 반영하면 오히려 품질이 떨어집니다. 반드시 검증 후 `enrich_qa`를 활성화하십시오.
- **운영 복잡도**: 패턴 4(풀스택)는 3개 시스템을 모두 운영해야 합니다. 팀 역량과 인프라를 고려하십시오.
- **추가 비용**: 온톨로지 추출과 RAG 검색 모두 처리 시간과 비용이 추가됩니다.

---

## 8. 관련 문서

| 문서 | 내용 |
|------|------|
| [사용 가이드](guide.html) | slm-factory 설치부터 모델 배포까지 단계별 안내 |
| [설정 레퍼런스](configuration.html) | `ontology` 설정 블록, `export.ollama` 설정 상세 |
| [CLI 레퍼런스](cli-reference.html) | `tool ontology`, `tool wizard`, `tool evolve` 명령어 상세 |
| [아키텍처 가이드](architecture.html) | 온톨로지 모듈의 내부 구조와 설계 원칙 |
| [개발 가이드](development.html) | 모듈 확장 방법 |
| [AutoRAG GitHub](https://github.com/Marker-Inc-Korea/AutoRAG) | AutoRAG 공식 저장소 |
