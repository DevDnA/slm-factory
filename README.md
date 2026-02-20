# slm-factory

도메인 문서를 학습하여 특화된 소형 언어모델(SLM)을 자동 생성하는 Teacher-Student 지식 증류 프레임워크

---

## 소개

**slm-factory**는 도메인 문서에서 소형 언어모델(SLM)을 자동으로 생성합니다.

대형 언어모델(Teacher)이 도메인 문서를 읽고 질문-답변 쌍을 생성하면, 소형 모델(Student)이 이를 학습하여 특정 분야에 특화된 경량 모델을 만듭니다. 이 전체 과정을 **한 번의 명령으로 자동화**합니다.

### 왜 소형 언어모델(SLM)인가?

| 장점 | 설명 |
|------|------|
| **비용 효율성** | 대형 모델 대비 추론 비용 10배 이상 저렴 |
| **속도** | 빠른 응답 생성, 실시간 서비스에 적합 |
| **프라이버시** | 로컬 실행으로 민감한 데이터 외부 전송 불필요 |
| **도메인 특화** | 특정 분야에서 범용 모델보다 높은 정확도 |

---

## 활용 예시: 자동 진화형 장애 대응 AI

장애 매뉴얼과 사고 보고서를 넣으면 **자동으로 진화하는 장애 대응 AI**가 만들어집니다. 기존 모니터링·장애 예측 시스템과 결합하여, 장애 감지 즉시 대응 가이드를 추천하는 경량 AI를 구축할 수 있습니다. 새로운 장애 사례가 추가될 때마다 모델이 자동으로 학습하여 지속적으로 진화합니다.

```
장애 문서 (매뉴얼/사고 보고서/복구 절차서)
  ▼
┌────────────────────────────────────────────────────────┐
  slm-factory 파이프라인
  문서 파싱 → QA 생성 → 검증 → LoRA 학습 → Ollama 배포
└──────────────────────────┬─────────────────────────────┘
                           ▼
              ┌──────────────────────────┐
                장애 대응 특화 SLM 완성
                Ollama API로 즉시 서빙
              └──────────────────────────┘

  새 장애 사례 추가 → tool update → 재학습 → 재배포 (진화 사이클 반복)
```

| 시점 | 투입 문서 | 모델이 학습한 지식 |
|------|----------|------------------|
| **초기 배포** | 장애 매뉴얼 10건 | DB, 네트워크, 서버 장애 대응 |
| **1개월 후** | + 신규 사고 보고서 5건 | + K8s 장애, 메모리 릭 대응 추가 |
| **3개월 후** | + 복구 절차서 8건 | + 클라우드 장애, 보안 사고 대응 추가 |

**첫 모델 생성**

```bash
slm-factory init fault-response
cp incident-reports/*.pdf fault-response/documents/
slm-factory tool wizard --config fault-response/project.yaml
```

**모델 진화 — 새 장애 사례 추가 시**

```bash
cp new-incident.pdf fault-response/documents/
slm-factory tool evolve --config fault-response/project.yaml    # 단일 명령: 증분→학습→품질게이트→버전배포
```

**외부 시스템 연동**

```bash
# 모니터링/장애 예측 시스템에서 Ollama API로 대응 가이드 즉시 요청
curl http://localhost:11434/api/generate -d '{
  "model": "fault-response-model",
  "prompt": "DB 커넥션 풀 사용률 95% 초과, 응답 지연 급증. 대응 방안은?",
  "stream": false
}'
```

> 장애 대응 외에도 **사내 규정 Q&A**, **제품 기술 지원**, **의료 가이드라인**, **법률 자문** 등 도메인 문서가 축적되는 모든 분야에 동일하게 적용할 수 있습니다.

---

## 파이프라인 개요

```
문서 (PDF/HWPX/HTML/TXT/DOCX)
  ▼
┌────────────┐ ───▶ ┌────────────┐ ───▶ ┌────────────┐
     Parse             Generate            Validate        필수 단계
   문서 파싱            QA 생성             QA 검증
└────────────┘      └────────────┘      └──────┬─────┘
                                               ▼
                           ┌───────────────────┬───────────────────┐
                           ▼                   ▼                   ▼
                    ┌────────────┐      ┌────────────┐      ┌────────────┐
                         Score              Augment             Analyze        선택 단계
                       품질 평가          데이터증강           통계 분석
                    └────────────┘      └──────┬─────┘      └────────────┘
                                               ▼
                    ┌────────────┐ ───▶ ┌────────────┐ ───▶ ┌────────────┐
                        Convert              Train              Export        필수 단계
                      데이터변환           LoRA 학습           모델 배포
                    └────────────┘      └────────────┘      └──────┬─────┘
                                                                   ▼
                                                         도메인 특화 SLM 완성
```

| 단계 | 입력 | 출력 |
|------|------|------|
| Parse | PDF/HWPX/HTML/TXT/DOCX | `ParsedDocument` |
| Generate | ParsedDocument + Teacher LLM | QA 쌍 (Alpaca JSON) |
| Validate | QA 쌍 | 필터링된 QA 쌍 |
| Score (선택) | QA 쌍 | 품질 점수 평가된 QA 쌍 |
| Augment (선택) | QA 쌍 | 패러프레이즈 증강된 QA 쌍 |
| Analyze (선택) | QA 쌍 | 데이터 분석 보고서 |
| Convert | Alpaca JSON | 채팅 템플릿 JSONL |
| Train | JSONL | LoRA 어댑터 |
| Export | LoRA 어댑터 | 병합 모델 + Ollama Modelfile |

---

## 주요 기능

- **다중 형식 파싱** — PDF, HWPX(한글), HTML, TXT/MD, DOCX(Word)
- **유연한 Teacher LLM** — Ollama(로컬) 또는 OpenAI 호환 API
- **다중 QA 검증** — 규칙 기반 + 임베딩 기반 자동 필터링
- **품질 점수 평가** — Teacher LLM이 1~5점으로 평가, threshold 필터링
- **데이터 증강** — 질문 패러프레이즈로 학습 데이터 확장
- **자동 데이터 분석** — 카테고리 분포, 길이 통계, 불균형 경고
- **자동 채팅 템플릿** — HuggingFace 모든 대화형 모델 지원
- **LoRA 파인튜닝** — 효율적 학습 + 조기 종료로 과적합 방지
- **원클릭 Ollama 배포** — Modelfile 자동 생성, 즉시 서비스
- **GGUF 양자화** — llama.cpp 호환 형식 변환
- **증분 학습** — 문서 추가 시 기존 QA 유지, 새 문서만 처리
- **자동 진화** — 단일 명령으로 증분 업데이트 → 재학습 → 품질 게이트 → 버전된 모델 배포
- **멀티턴 대화 생성** — QA 쌍을 다중 턴 대화로 확장
- **QA 수동 리뷰 (TUI)** — 승인/거부/편집 인터페이스
- **파이프라인 대시보드 (TUI)** — 실시간 진행 모니터링
- **자동 모델 평가** — BLEU/ROUGE 메트릭 + Before/After 비교
- **대화형 wizard** — 문서 선택부터 배포까지 단계별 안내

---

## 기술 스택

| 카테고리 | 주요 패키지 | 역할 |
|---------|------------|------|
| **Core** | typer, pydantic, pyyaml, rich, httpx | CLI, 설정, 출력 |
| **Parsing** | pymupdf, beautifulsoup4, lxml, python-docx | 문서 파싱 |
| **ML/Training** | torch, transformers, peft, trl, accelerate | LoRA 학습 |
| **Evaluation** | evaluate, rouge-score, nltk | 모델 평가 |
| **TUI** | textual | 대시보드, 리뷰 |
| **Optional** | pykospacing, sentence-transformers, pdfplumber, bitsandbytes | 한국어 교정, 임베딩, CUDA 양자화 |

---

## 시스템 요구사항

- **Python** 3.11 이상
- **GPU** (자동 감지):
  - **NVIDIA GPU** — CUDA 지원, VRAM 8GB 이상 권장
  - **Apple Silicon** (M1/M2/M3/M4/M5) — MPS 백엔드, Unified Memory 활용
  - **CPU** — GPU 미감지 시 자동 폴백 (학습 속도 느림)
- **Ollama**: Teacher LLM으로 사용 시 설치 필요 ([ollama.com](https://ollama.com))
- **디스크**: 약 5GB 이상 (모델 다운로드 + 체크포인트)

> **Apple Silicon 참고**: macOS에서는 BitsAndBytes 양자화를 사용할 수 없지만, Unified Memory 구조 덕분에 시스템 RAM 전체를 GPU가 공유하므로 양자화 없이도 비교적 큰 모델을 로드할 수 있습니다. 학습 정밀도는 자동으로 float16으로 설정됩니다.

---

## 빠른 시작

### 1. 설치

```bash
git clone https://github.com/DevDnA/slm-factory.git
cd slm-factory
python3 -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -e ".[all]"
slm-factory --install-completion
```

> ⏳ `pip install -e ".[all]"`은 PyTorch, CUDA 런타임 등 대용량 패키지를 포함하므로 **초회 설치 시 10~20분 이상** 소요될 수 있습니다. 네트워크 환경에 따라 다를 수 있으니 여유를 갖고 기다려 주십시오.

> 가상환경(venv) 활성화는 필수입니다. 시스템 Python에 직접 설치하면 `externally-managed-environment` 에러가 발생합니다 (PEP 668).

### 2. 사전 준비

```bash
ollama serve              # 별도 터미널에서 Ollama 실행
ollama pull qwen3:8b      # Teacher 모델 다운로드
```

### 3. 프로젝트 생성 및 실행

```bash
# 프로젝트 초기화
slm-factory init my-project

# 학습할 문서 추가
cp /path/to/documents/*.pdf my-project/documents/

# wizard로 전체 파이프라인 실행 (권장)
slm-factory tool wizard --config my-project/project.yaml
```

wizard가 문서 선택 → 파싱 → QA 생성 → 검증 → 학습 → 배포까지 단계별로 안내합니다. 상세한 진행 방법은 [사용 가이드](docs/guide.md)를 참조하십시오.

### 4. 모델 테스트

```bash
cd my-project/output/merged_model
ollama create my-project-model -f Modelfile
ollama run my-project-model
```

---

## CLI 명령어 요약

```
시작하기          init <name>              새 프로젝트 초기화
                 check                    설정 및 환경 점검

파이프라인        run [--until step]       파이프라인 실행
                 train [--data file]      LoRA 학습
                 export [--adapter dir]   모델 내보내기

평가             eval run --model name    BLEU/ROUGE 평가
                 eval compare --base-model --ft   Base vs Fine-tuned 비교

도구             tool wizard              대화형 파이프라인 (권장)
                 tool evolve              자동 진화 (증분→학습→품질게이트→배포)
                 tool review              QA 수동 리뷰 TUI
                 tool dashboard           대시보드 TUI
                 tool convert             QA → JSONL 변환
                 tool dialogue            멀티턴 대화 생성
                 tool gguf                GGUF 양자화 변환
                 tool update              증분 업데이트

정보             status                   진행 상태 확인
                 clean [--all]            중간 파일 정리
                 version                  버전 출력
```

전체 옵션과 사용법은 [CLI 레퍼런스](docs/cli-reference.md)를 참조하십시오.

---

## 문서 안내

| 문서 | 내용 | 대상 |
|------|------|------|
| **[빠른 참조](docs/quick-reference.md)** | 명령어 요약, 워크플로우, 빠른 해결 | 모든 사용자 |
| **[사용 가이드](docs/guide.md)** | 설치, 튜토리얼, 활용 예시, 트러블슈팅 | 새 사용자 + 활성 사용자 |
| **[CLI 레퍼런스](docs/cli-reference.md)** | 모든 명령어의 전체 옵션과 사용법 | 활성 사용자 |
| **[설정 레퍼런스](docs/configuration.md)** | project.yaml 전체 설정 옵션 | 활성 사용자 |
| **[아키텍처 가이드](docs/architecture.md)** | 설계 철학, 패턴, 데이터 흐름 | 기여자/개발자 |
| **[개발자 가이드](docs/development.md)** | 모듈 API, 프로젝트 구조, 확장 방법 | 기여자/개발자 |

---

## 라이선스

이 프로젝트의 라이선스는 추후 결정됩니다.

---

**slm-factory**로 도메인 특화 언어모델을 쉽고 빠르게 구축하십시오!
