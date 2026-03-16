# 아키텍처 가이드

> slm-factory의 내부 설계와 동작 원리를 이해하세요.

---

## 1. 설계 철학

SLM Factory는 다음 5가지 핵심 원칙을 기반으로 설계되었습니다.

### 1.1 YAML 중심 설정

모든 파이프라인 동작은 `project.yaml` 파일 하나로 제어됩니다. 코드 수정 없이 문서 파싱 옵션, Teacher 모델, 질문 카테고리, 검증 규칙, Student 모델, 학습 하이퍼파라미터, 내보내기 형식을 변경할 수 있습니다.

### 1.2 단계별 독립 실행

전체 파이프라인은 독립적인 단계(parse, chunking, generate, validate, score, augment, analyze, convert, train, export, eval, autorag_export, rag_index)로 구성됩니다. 각 단계는 CLI 명령어로 개별 실행 가능하며, 이전 단계의 출력 파일을 입력으로 사용합니다. 특정 단계만 재실행하거나 중간 결과를 검토할 수 있습니다.

### 1.3 지연 임포트 (Lazy Import)

`torch`, `transformers`, `peft`, `sentence-transformers` 등 무거운 라이브러리는 실제 사용 시점에만 로드됩니다. CLI 초기 응답 속도를 빠르게 유지하고, 특정 단계만 실행할 때 불필요한 의존성을 로드하지 않습니다. `parse` 명령어는 딥러닝 라이브러리를 전혀 로드하지 않습니다.

### 1.4 Pydantic v2 타입 안전

모든 설정은 Pydantic v2 모델로 정의되어 YAML 로드 시 자동으로 타입 검증이 수행됩니다. 기본값, 필수 필드, 값 범위 제약이 명시적으로 정의되어 있어 오타나 잘못된 값이 입력되면 즉시 에러 메시지를 출력합니다.

### 1.5 실패 격리

개별 파일 파싱 실패, 특정 문서의 QA 생성 실패, 단일 QA 쌍의 검증 실패 등은 전체 파이프라인을 중단시키지 않습니다. 실패한 항목은 로그에 기록되고 건너뛰며, 나머지 항목은 정상적으로 처리됩니다.

---

## 2. 전체 아키텍처

### 2.1 컴포넌트 구조 다이어그램

```
                     ┌─────────────────────────────┐
                       CLI (cli.py)
                       Entry Point (Typer)
                     └──────────────┬──────────────┘
                                    ▼
                     ┌─────────────────────────────┐
                         Pipeline (pipeline.py)
                          Orchestrator (12 steps)
                     └──────────────┬──────────────┘
                                    ▼

┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
  1. Parse             2. Generate          3. Validate
  parsers/             teacher/             validator/
└────────┬─────────┘ └────────┬─────────┘ └────────┬─────────┘
         ▼                    ▼                    ▼
  parsed_documents.json qa_alpaca.json       (filtered pairs)

┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
  4. Score             5. Augment           6. Analyze
  scorer.py            augmenter.py         analyzer.py
└────────┬─────────┘ └────────┬─────────┘ └────────┬─────────┘
         ▼                    ▼                    ▼
  qa_scored.json       qa_augmented.json   data_analysis.json

┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
  7. Convert           8. Train             9. Export
  converter.py         trainer/             exporter/
└────────┬─────────┘ └────────┬─────────┘ └────────┬─────────┘
         ▼                    ▼                    ▼
  training_data.jsonl  adapter/            merged_model/ Modelfile

┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
  10. Eval             11. Autorag Export   12. RAG Index
  evaluator.py         exporter/            rag/
└────────┬─────────┘ └────────┬─────────┘ └────────┬─────────┘
         ▼                    ▼                    ▼
  eval_results.json    autorag/*.parquet    chroma_db/

                     ┌─────────────────────────────┐
                       Config (config.py)
                       Injected into all components
                     └─────────────────────────────┘
```

### 2.2 모듈 의존성 다이어그램

```
cli.py
  └─→ pipeline.py
        ├─→ parsers/
        │     ├─→ base.py (BaseParser, ParserRegistry)
        │     ├─→ pdf.py (PDFParser)
        │     ├─→ hwpx.py (HWPXParser)
        │     ├─→ html.py (HTMLParser)
        │     ├─→ text.py (TextParser)
        │     └─→ docx.py (DOCXParser)
        │
        ├─→ teacher/
        │     ├─→ base.py (BaseTeacher)
        │     ├─→ ollama.py (OllamaTeacher)
        │     ├─→ openai_compat.py (OpenAICompatTeacher)
        │     └─→ qa_generator.py (QAGenerator)
        │
        ├─→ validator/
        │     ├─→ rules.py (RuleValidator)
        │     └─→ similarity.py (GroundednessChecker)
        │
        ├─→ scorer.py (QualityScorer)
        ├─→ augmenter.py (DataAugmenter)
        ├─→ analyzer.py (DataAnalyzer)
        ├─→ converter.py (ChatFormatter)
        │
        ├─→ trainer/
        │     └─→ lora_trainer.py (DataLoader, LoRATrainer)
        │
        ├─→ exporter/
        │     ├─→ hf_export.py (HFExporter)
        │     └─→ ollama_export.py (OllamaExporter)
        │
        ├─→ evaluator.py (ModelEvaluator)
        ├─→ comparator.py (ModelComparator)
        ├─→ incremental.py (IncrementalTracker)
        └─→ tui/ (QAReviewerApp, PipelineDashboard)

모든 모듈
  ├─→ config.py (SLMConfig + 20개 하위 모델)
  ├─→ models.py (ParsedDocument, QAPair, EvalResult, CompareResult)
  └─→ utils.py (setup_logging, get_logger)
```

---

## 3. 핵심 설계 패턴

### 3.1 Registry 패턴 (parsers/)

파서 시스템은 Registry 패턴으로 확장 가능한 파일 형식 지원을 제공합니다. `ParserRegistry`가 `BaseParser` 인스턴스 목록을 관리하며, `get_parser(path)`는 등록된 파서를 순회하여 `can_parse(path)`를 만족하는 첫 번째 파서를 반환합니다.

```python
@registry.register
class CSVParser(BaseParser):
    extensions = [".csv"]

    def parse(self, path: Path) -> ParsedDocument:
        ...
```

`parsers/__init__.py`에서 전역 `registry` 인스턴스를 생성하고 5개 파서(PDF, HWPX, HTML, Text, DOCX)를 자동 등록합니다. 새 파서를 `@registry.register`로 등록하면 즉시 파이프라인에서 처리됩니다.

`parsers/base.py`의 `detect_encoding()` 함수는 HTML 파서와 텍스트 파서가 공유하는 인코딩 감지 유틸리티입니다. charset-normalizer를 사용하여 EUC-KR, CP949 등 한국어 인코딩을 정확하게 감지합니다. `HTMLParser`는 BeautifulSoup4의 lxml 백엔드를 사용하며, h1–h6 헤딩 태그를 마크다운 헤딩으로 변환하여 문서 구조를 보존합니다.

→ 커스텀 파서 구현 전체 예제는 [개발자 가이드](development.md) 참조

### 3.2 Factory 패턴 (teacher/)

Teacher 백엔드 생성은 Factory 패턴으로 설정 기반의 적절한 구현체를 반환합니다.

```python
def create_teacher(config: TeacherConfig) -> BaseTeacher:
    if config.backend == "ollama":
        from .ollama import OllamaTeacher
        return OllamaTeacher(config)
    elif config.backend == "openai":
        from .openai_compat import OpenAICompatTeacher
        return OpenAICompatTeacher(config)
    raise ValueError(f"Unknown backend: {config.backend}")
```

`BaseTeacher`는 `generate(prompt, **kwargs) → str`와 `health_check() → bool`을 추상 메서드로 정의합니다.

→ 새 백엔드 추가 방법은 [개발자 가이드](development.md) 참조

### 3.3 Strategy 패턴 (validator/)

검증 시스템은 Strategy 패턴으로 여러 검증 전략을 조합합니다.

- `RuleValidator`: empty → length → pattern → dedup 순서로 4개 규칙을 체인 적용합니다.
- `GroundednessChecker`: sentence-transformers 임베딩 기반 코사인 유사도 검증을 수행합니다.
- 두 검증기 모두 `(accepted, rejected)` 튜플을 반환하며, `step_validate()`에서 순차 적용됩니다.

### 3.4 Adapter 패턴 (converter.py)

`ChatFormatter`는 HuggingFace `tokenizer.apply_chat_template()`을 래핑하여 통일된 `QAPair` 형식을 모델별 채팅 형식으로 변환합니다. Gemma처럼 시스템 역할을 지원하지 않는 모델은 시스템 메시지를 제거하고 자동으로 재시도합니다. `max_seq_length` 초과 항목은 자동으로 제외됩니다.

### 3.5 비동기 동시성 패턴 (scorer, augmenter, regeneration)

Scorer, Augmenter, 그리고 저품질 QA 재생성 루프는 `asyncio.Semaphore`로 동시 요청 수를 제한하면서 병렬 처리합니다.

```python
semaphore = asyncio.Semaphore(config.max_concurrency)

async def _bounded_task(item):
    async with semaphore:
        return await process(item)

results = await asyncio.gather(*tasks, return_exceptions=True)
```

`return_exceptions=True`로 개별 실패를 허용하며, Rich Progress 바로 실시간 진행 상황을 표시합니다.

### 3.6 통계 분석 패턴 (analyzer)

`DataAnalyzer`는 LLM 의존성 없이 순수 통계 분석을 수행합니다. `Counter`로 분포를 계산하고, `statistics` 모듈로 기초 통계를 산출합니다. 데이터 불균형, 이상치 등을 자동으로 감지하여 경고를 생성하고 Rich 콘솔로 시각적 보고서를 출력합니다.

→ 각 패턴의 실제 API는 [개발자 가이드](development.md) 참조

---

## 4. 데이터 흐름

### 4.1 파이프라인 단계별 입출력

| 단계 | 입력 타입 | 출력 타입 | 저장 파일 |
|------|----------|----------|----------|
| parse | 파일 시스템 (PDF/HWPX/HTML/TXT/MD/DOCX/HWP) | `list[ParsedDocument]` | `parsed_documents.json` |
| generate | `list[ParsedDocument]` | `list[QAPair]` | `qa_alpaca.json` |
| validate | `list[QAPair]` | `list[QAPair]` (필터링) | 없음 (메모리 전달) |
| score | `list[QAPair]` | `list[QAPair]` (필터링) | `qa_scored.json` |
| augment | `list[QAPair]` | `list[QAPair]` (원본+증강) | `qa_augmented.json` |
| analyze | `list[QAPair]` | 보고서 (JSON) | `data_analysis.json` |
| convert | `list[QAPair]` | JSONL 파일 경로 | `training_data.jsonl` |
| train | JSONL 파일 경로 | 어댑터 디렉토리 경로 | `checkpoints/adapter/` |
| export | 어댑터 디렉토리 경로 | 병합 모델 디렉토리 경로 | `merged_model/` + `Modelfile` |
| eval | `list[QAPair]` + 모델 이름 | `list[EvalResult]` | `eval_results.json` |
| compare | `list[QAPair]` | `list[CompareResult]` | `compare_results.json` |
| autorag_export | `list[ParsedDocument]` + `list[QAPair]` | parquet files | `autorag/corpus.parquet`, `autorag/qa.parquet` |
| rag_index | `corpus.parquet` | ChromaDB collection | `chroma_db/` |

### 4.2 중간 파일 체인 다이어그램

```
documents/
  (PDF, HWPX, HTML, TXT, DOCX)
        │
        ▼ step_parse()
  parsed_documents.json
        │
        ├──▶ chunking (chunking.enabled=true)
        │      긴 문서를 청크로 분할 → 각 청크마다 QA 생성
        │
        ▼ step_generate()
qa_alpaca.json
        │
        ▼ step_validate()
    [메모리]
        │
        ├─▶ step_score()   ──▶  qa_scored.json
        │
        ├─▶ step_augment() ──▶  qa_augmented.json
        │
        ├─▶ step_analyze() ──▶  data_analysis.json
        │
        ▼ step_convert()
training_data.jsonl
        │
        ▼ step_train()
checkpoints/adapter/
  ├── adapter_config.json
  ├── adapter_model.safetensors
  └── tokenizer_config.json
        │
        ▼ step_export()
merged_model/
  ├── model.safetensors
  ├── config.json
  ├── tokenizer.json
  └── Modelfile
        │
        ▼ step_eval()           (eval.enabled)
  eval_results.json
        │
        ▼ step_autorag_export() (autorag_export.enabled)
  autorag/corpus.parquet + qa.parquet
        │
        ▼ step_rag_index()
  chroma_db/
```

### 4.3 재개 메커니즘 (--resume)

`--resume` 옵션을 사용하면 CLI가 중간 파일의 존재 여부를 확인하여 완료된 단계를 건너뜁니다. 체크포인트 감지 우선순위는 다음과 같습니다.

```
qa_augmented.json 존재   →  analyze 단계부터 재개
qa_scored.json 존재      →  augment 단계부터 재개
qa_alpaca.json 존재      →  validate 단계부터 재개
parsed_documents.json    →  generate 단계부터 재개
없음                     →  처음부터 실행
```

`slm-factory status` 명령으로 각 중간 파일의 존재 여부와 항목 수를 확인할 수 있습니다.

→ 각 단계 사용법은 [사용 가이드](guide.md) 참조  
→ 각 단계 API는 [개발자 가이드](development.md) 참조

---

## 5. 설정 시스템 아키텍처

### 5.1 설정 모델 계층 구조

```
SLMConfig (root)
├── project: ProjectConfig
│   ├── name: str = "my-project"
│   ├── version: str = "1.0.0"
│   └── language: str = "en"
│
├── paths: PathsConfig
│   ├── documents: Path = "./documents"
│   └── output: Path = "./output"
│
├── parsing: ParsingConfig
│   ├── formats: list[str] = ["pdf", "txt", "html"]
│   ├── pdf: PdfOptions
│   │   └── extract_tables: bool = True
│   └── hwpx: HwpxOptions
│       └── apply_spacing: bool = True
│
├── teacher: TeacherConfig
│   ├── backend: Literal["ollama", "openai"] = "ollama"
│   ├── model: str = "qwen3.5:9b"
│   ├── api_base: str = "http://localhost:11434"
│   ├── api_key: str | None = None
│   ├── temperature: float = 0.3
│   ├── timeout: int = 300
│   ├── max_context_chars: int = 12000
│   └── max_concurrency: int = 2
│
├── questions: QuestionsConfig
│   ├── categories: dict[str, list[str]] = {}
│   ├── file: Path | None = None
│   ├── system_prompt: str = "You are a helpful..."
│   └── output_format: str = "alpaca"
│
├── validation: ValidationConfig
│   ├── enabled: bool = True
│   ├── min_answer_length: int = 20
│   ├── max_answer_length: int = 2000
│   ├── remove_empty: bool = True
│   ├── deduplicate: bool = True
│   ├── reject_patterns: list[str] = [...]
│   └── groundedness: GroundednessConfig
│       ├── enabled: bool = True
│       ├── model: str = "all-MiniLM-L6-v2"
│       └── threshold: float = 0.3
│
├── chunking: ChunkingConfig
│   ├── enabled: bool = True
│   ├── chunk_size: int = 10000
│   └── overlap_chars: int = 500
│
├── scoring: ScoringConfig
│   ├── enabled: bool = True
│   ├── threshold: float = 3.0
│   └── max_concurrency: int = 2
│
├── augment: AugmentConfig
│   ├── enabled: bool = True
│   ├── num_variants: int = 2
│   ├── max_concurrency: int = 2
│   └── min_similarity: float = 0.3
│
├── analyzer: AnalyzerConfig
│   ├── enabled: bool = True
│   └── output_file: str = "data_analysis.json"
│
├── student: StudentConfig
│   ├── model: str = "google/gemma-3-1b-it"
│   └── max_seq_length: int = 4096
│
├── training: TrainingConfig
│   ├── batch_size: int = 1
│   ├── gradient_accumulation_steps: int = 8
│   ├── learning_rate = 2e-4
│   ├── lr_scheduler: str = "cosine"
│   ├── warmup_ratio: float = 0.1
│   ├── num_epochs: int = 5
│   ├── optimizer: str = "adamw_torch_fused"
│   ├── bf16: bool = True
│   ├── train_split: float = 0.9
│   ├── save_strategy: str = "epoch"
│   ├── lora: LoraConfig
│   │   ├── r: int = 8
│   │   ├── alpha: int = 16
│   │   ├── dropout: float = 0.1
│   │   ├── target_modules: str | list[str] = "auto"
│   │   └── use_rslora: bool = False
│   ├── early_stopping: EarlyStoppingConfig
│   │   ├── enabled: bool = True
│   │   ├── patience: int = 3
│   │   └── threshold: float = 0.01
│   ├── quantization: QuantizationConfig
│   │   ├── enabled: bool = True
│   │   └── bits: int = 4
│   └── neftune_noise_alpha: float | None = None
│
├── export: ExportConfig
│   ├── merge_lora: bool = True
│   ├── output_format: str = "safetensors"
│   └── ollama: OllamaExportConfig
│       ├── enabled: bool = True
│       ├── model_name: str = "my-project-model"
│       ├── system_prompt: str = "You are a helpful..."
│       └── parameters: dict = {temperature: 0.7, top_p: 0.9, num_ctx: 4096}
│
├── eval: EvalConfig
│   ├── enabled: bool = True
│   ├── test_split: float = 0.1
│   ├── metrics: list[str] = ["bleu", "rouge"]
│   ├── max_samples: int = 50
│   ├── max_tokens: int = 512
│   ├── output_file: str = "eval_results.json"
│   ├── quality_gate: bool = True
│   └── quality_thresholds: dict = {"bleu": 0.1, "rougeL": 0.2}
│
├── incremental: IncrementalConfig
│   ├── enabled: bool = True
│   ├── hash_file: str = "document_hashes.json"
│   ├── merge_strategy: Literal["append", "replace"] = "append"
│   └── resume_adapter: str = ""
│
├── review: ReviewConfig
│   ├── enabled: bool = True
│   ├── auto_open: bool = True
│   └── output_file: str = "qa_reviewed.json"
│
├── compare: CompareConfig
│   ├── enabled: bool = False
│   ├── base_model: str = ""
│   ├── finetuned_model: str = ""
│   ├── metrics: list[str] = ["bleu", "rouge"]
│   ├── max_samples: int = 20
│   ├── max_tokens: int = 512
│   └── output_file: str = "compare_results.json"
│
├── evolve: EvolveConfig
│   ├── quality_gate: bool = True
│   ├── gate_metric: str = "rougeL"
│   ├── gate_min_improvement: float = 0.0
│   ├── version_format: str = "date"
│   ├── history_file: str = "evolve_history.json"
│   └── keep_previous_versions: int = 3
│
├── dashboard: DashboardConfig
│   ├── enabled: bool = True
│   ├── refresh_interval: float = 2.0
│   └── theme: str = "dark"
│
├── autorag_export: AutoRAGExportConfig
│   ├── enabled: bool = True
│   ├── output_dir: str = "autorag"
│   ├── chunk_size: int = 512
│   └── overlap_chars: int = 64
│
└── rag: RagConfig
    ├── embedding_model: str = "BAAI/bge-m3"
    ├── vector_db_path: str = "chroma_db"
    ├── collection_name: str = "corpus"
    ├── top_k: int = 5
    ├── server_host: str = "0.0.0.0"
    ├── server_port: int = 8000
    ├── ollama_model: str = ""
    ├── workers: int = 1
    ├── log_level: str = "info"
    ├── cors_origins: list[str] = ["*"]
    ├── batch_size: int = 64
    ├── request_timeout: float = 120.0
    └── max_tokens: int = 512
```

→ 각 필드의 상세 설명은 [설정 레퍼런스](configuration.md) 참조

### 5.2 설정 로드 프로세스

CLI에서 `--config` 옵션으로 경로를 지정하거나, 지정하지 않으면 `_find_config()`가 현재 디렉토리부터 상위 디렉토리를 순회하며 `project.yaml`을 자동 탐색합니다.

**`load_config()` 흐름:**

```
CLI --config project.yaml
        │
        ▼ _find_config() (자동 탐색 시)
        │  현재 디렉토리 → 상위 디렉토리 순회
        │
        ▼ load_config(path)
        │  1. Path(path).resolve()          — 절대 경로 변환
        │  2. yaml.safe_load()              — YAML 파싱
        │  3. SLMConfig.model_validate(raw) — Pydantic 검증
        │     └─ _strip_none_sections()     — None 키 제거 (기본값 적용)
        │  4. 상대 경로 → 절대 경로 변환
        │     (paths.documents, paths.output)
        │
        ▼ SLMConfig 객체 반환
```

**`_strip_none_sections` 검증기:**

YAML에서 섹션을 생략하거나 `null`로 설정하면, 이 `model_validator`가 `None` 키를 제거하여 Pydantic 기본값이 자동으로 적용되도록 합니다.

```python
@model_validator(mode="before")
@classmethod
def _strip_none_sections(cls, values: dict) -> dict:
    if isinstance(values, dict):
        return {k: v for k, v in values.items() if v is not None}
    return values
```

예를 들어 `project.yaml`에 `parsing` 섹션이 없으면 `ParsingConfig()`의 기본값이 그대로 적용됩니다.

**에러 처리:**

| 상황 | 발생 예외 |
|------|----------|
| 파일 없음 | `FileNotFoundError` |
| 유효하지 않은 YAML | `yaml.YAMLError` |
| 스키마 불일치 | `pydantic.ValidationError` |

### 5.3 기본값 생성 로직

`create_default_config()`는 `slm-factory init` 명령 실행 시 호출되며, 다음 순서로 기본 YAML 템플릿을 반환합니다.

```
1. 패키지 루트의 templates/project.yaml 읽기 시도
        │ 실패 시
        ▼
2. importlib.resources로 설치된 wheel에서 읽기 시도
        │ 실패 시
        ▼
3. yaml.dump(SLMConfig().model_dump(), ...) — 최소 기본값을 YAML 형식으로 반환
```

개발 환경(편집 가능 설치)과 배포된 wheel 모두에서 동일하게 동작합니다.

---

## 6. 에러 처리 전략

### 6.1 계층별 에러 처리

| 계층 | 에러 타입 | 처리 방식 | 사용자 영향 |
|------|----------|----------|------------|
| CLI | `FileNotFoundError`, `ValidationError`, 기타 | Rich 포맷 에러 출력 + `typer.Exit(1)` | 명확한 에러 메시지, 비정상 종료 |
| Pipeline | 모든 예외 | `logger.exception()` 후 예외 전파 | 단계 실패 시 파이프라인 중단 |
| Parser | 개별 파일 파싱 예외 | `logger.exception()` 후 해당 파일 건너뜀 | 나머지 파일 계속 처리 |
| Teacher | `httpx.TimeoutException`, `ConnectError`, `HTTPStatusError` | 분류된 `RuntimeError`로 변환, 지수 백오프 재시도 (3회) | 명확한 연결/타임아웃 안내 |
| Validator | 규칙 위반 | `ValidationResult.reasons` 리스트에 기록 | 거부 사유 추적 가능 |
| Trainer | `RuntimeError` (CUDA OOM 등) | OOM 감지 시 권장 사항 출력 후 재발생 | 배치 크기/양자화 조정 안내 |

### 6.2 복구 전략

| 상황 | 전략 | 구현 위치 |
|------|------|----------|
| 개별 파일 파싱 실패 | 로그 후 건너뜀, 나머지 계속 처리 | `ParserRegistry.parse_directory()` |
| QA 생성 JSON 파싱 실패 | 원본 응답 로그 출력 후 건너뜀 | `QAGenerator.parse_response()` |
| 시스템 역할 미지원 모델 | 시스템 메시지 제거 후 재시도 | `ChatFormatter.format_one()` |
| Teacher 타임아웃 | 설정 값 증가 권장 메시지 출력 후 실패 | `BaseTeacher.generate()` |
| Ollama 응답 실패 | 지수 백오프로 최대 3회 재시도 후 최종 예외 발생 | `ollama_generate()` |
| CUDA OOM | 배치 크기/양자화 권장 후 예외 재발생 | `LoRATrainer.train()` |
| QA 생성 결과 0건 | `RuntimeError` 발생하여 파이프라인 중단 | `Pipeline.step_generate()` |
| 변환 결과 빈 배치 | `RuntimeError` 발생하여 파이프라인 중단 | `ChatFormatter.format_batch()` |
| JSON 파일 손상 | 빈 기본값으로 복구 후 로그 출력 | `EvolveHistory`, `IncrementalTracker` |
| 진화 품질 게이트 실패 | 생성된 Ollama 모델 자동 삭제 후 이전 모델 유지 | `cli.py` evolve 명령 |
| 설정 파일 없음 | 기본 설정 생성 제안 (`slm-factory init`) | `cli.py` |
| 파이프라인 중단 | `--resume`으로 중간 파일에서 재개 | `cli.py` + 중간 파일 체인 |

---

## 7. Wizard 아키텍처

Wizard 모드는 처음 사용자에게 권장하는 기본 실행 방식입니다. CLI의 개별 명령어를 직접 호출하는 대신, 단일 `tool wizard` 명령으로 전체 파이프라인을 단계별 확인하며 실행할 수 있습니다.

**핵심 원칙:**
- 사용자가 파이프라인 구조를 몰라도 진행 가능합니다.
- 어느 단계에서든 건너뛸 수 있고, 나중에 개별 CLI 명령으로 재개할 수 있습니다.
- 선택적 단계의 기본값은 `project.yaml` 설정을 따릅니다.
- 각 단계 완료 시 건수/경로 등 결과를 즉시 표시합니다.

### 7.1 실행 흐름 다이어그램

```
wizard
  ├─ 준비. 설정 파일 ──────────── _find_config() → load_config()
  │    └─ 자동 탐색 실패 시 Prompt.ask()
  │
  ├─ 준비. 문서 선택 ──────────── 디렉토리 스캔 → Rich Table → Confirm/번호 입력
  │    └─ 전체 선택 또는 개별 번호 입력
  │
  ├─ Step 1. 문서 파싱 ──────────── pipeline.step_parse(files=selected)
  │    └─ 자동 진행 (확인 없음)
  │
  ├─ Step 2. QA 생성 ────────────── Confirm → pipeline.step_generate(docs)
  │    └─ 거부 시 → 파싱 결과 경로 안내 후 종료
  │
  ├─ Step 3. 검증 ───────────────── pipeline.step_validate(pairs, docs)
  │    └─ 자동 진행 (확인 없음)
  │
  ├─ Step 4. 품질 평가 (선택적) ──── Confirm(default=scoring.enabled) → step_score()
  │    └─ config.scoring.enabled 기본값 반영
  │
  ├─ Step 5. 데이터 증강 (선택적) ── Confirm(default=augment.enabled) → step_augment()
  │    └─ config.augment.enabled 기본값 반영
  │
  ├─ Step 6. 통계 분석 (자동) ───── pipeline.step_analyze(pairs)
  │    └─ 자동 진행 (확인 없음)
  │
  ├─ Step 7. 데이터 변환 (자동) ─── step_convert()
  │    └─ 자동 진행 (확인 없음)
  │
  ├─ Step 8. LoRA 학습 ──────────── Confirm → step_train()
  │    └─ 거부 시 → 학습 데이터 경로 + train 명령어 안내 후 종료
  │
  ├─ Step 9. 내보내기 ──────────── Confirm → step_export()
  │    └─ 거부 시 → 어댑터 경로 + export 명령어 안내 후 종료
  │
  ├─ Step 10. 모델 평가 (선택적) ──── Confirm(default=eval.enabled) → step_eval()
  │    └─ config.eval.enabled 기본값 반영
  │
  ├─ Step 11. 코퍼스 내보내기 (선택적) ── Confirm(default=autorag_export.enabled) → step_autorag_export()
  │    └─ config.autorag_export.enabled 기본값 반영
  │
  └─ Step 12. RAG 인덱싱 (선택적) ──────── Confirm(default=True) → step_rag_index()
       └─ rag 설정 반영
```

### 7.2 필수/선택 단계 분류

| 분류 | 단계 | 확인 방식 | 기본값 |
|------|------|----------|--------|
| **필수 (자동)** | 설정 로드, 문서 파싱, QA 검증, 통계 분석, 데이터 변환 | 자동 진행 | — |
| **필수 (확인)** | QA 생성, LoRA 학습, 모델 내보내기 | `Confirm.ask(default=True)` | Y |
| **선택 (설정 의존)** | 품질 평가, 데이터 증강, 모델 평가, 코퍼스 내보내기, RAG 인덱싱 | `Confirm.ask(default=config값)` | config 의존 |

### 7.3 건너뛰기 시 복구 안내 메커니즘

각 단계를 건너뛸 때 wizard는 나중에 해당 단계를 개별 실행할 수 있는 정확한 CLI 명령어를 안내합니다.

```
QA 생성 건너뜀:
  → "나중에 실행: slm-factory run --until generate --config {config_path}"

학습 건너뜀:
  → "나중에 실행: slm-factory train --config {config_path} --data {training_data_path}"

내보내기 건너뜀:
  → "나중에 실행: slm-factory export --config {config_path} --adapter {adapter_path}"
```

중간 결과 파일은 보존되므로 `--resume` 옵션 또는 개별 명령어로 이어서 진행할 수 있습니다.

### 7.4 Rich UI 컴포넌트 활용

| 컴포넌트 | 용도 |
|----------|------|
| `Panel` | 시작 배너, 완료 요약 |
| `Table` | 문서 목록 (번호/파일명/크기) |
| `Confirm.ask()` | Y/n 선택 (기본값 지원) |
| `Prompt.ask()` | 텍스트 입력 (설정 파일 경로, 문서 번호) |
| 색상 마커 | `[green]✓` 성공, `[red]✗` 실패, `[yellow]⏭` 건너뜀 |

---

## 관련 문서

| 문서 | 내용 |
|------|------|
| [README](../README.md) | 프로젝트 소개, 설치, 빠른 시작 |
| [사용 가이드](guide.md) | 단계별 사용법, 예제, 트러블슈팅 |
| [CLI 레퍼런스](cli-reference.md) | 모든 명령어와 옵션 상세 설명 |
| [설정 레퍼런스](configuration.md) | `project.yaml` 모든 필드 상세 설명 |
| [개발자 가이드](development.md) | 커스텀 파서/백엔드 추가, 모듈 API |
