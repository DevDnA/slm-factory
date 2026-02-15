# 모듈별 상세 문서

## 1. 개요

### 1.1 디렉토리 구조

```
src/slm_factory/
├── __init__.py              (4줄)     패키지 초기화 + 버전
├── config.py                (298줄)   설정 시스템
├── cli.py                   (331줄)   CLI 인터페이스
├── pipeline.py              (331줄)   파이프라인 오케스트레이터
├── converter.py             (~265줄)  채팅 포맷터 (converter/ 통합)
├── models.py                (~37줄)   공유 데이터 모델 (QAPair, ParsedDocument)
├── utils.py                 (~30줄)   로깅 유틸리티 (utils/ 통합)
├── scorer.py                (125줄)   QA 품질 점수 평가
├── augmenter.py             (119줄)   QA 데이터 증강
├── analyzer.py              (173줄)   학습 데이터 분석
├── parsers/
│   ├── __init__.py          (24줄)    파서 레지스트리
│   ├── base.py              (163줄)   기본 클래스
│   ├── pdf.py               (165줄)   PDF 파서
│   ├── hwpx.py              (181줄)   HWPX 파서
│   ├── html.py              (180줄)   HTML 파서
│   └── text.py              (101줄)   텍스트 파서
├── teacher/
│   ├── __init__.py          (50줄)    팩토리 함수
│   ├── base.py              (56줄)    기본 클래스
│   ├── ollama.py            (169줄)   Ollama 백엔드
│   ├── openai_compat.py     (186줄)   OpenAI 호환 백엔드
│   └── qa_generator.py      (386줄)   QA 생성기
├── validator/
│   ├── __init__.py          (13줄)    re-export
│   ├── rules.py             (114줄)   규칙 검증기
│   └── similarity.py        (142줄)   임베딩 검증기
├── trainer/
│   ├── __init__.py          (7줄)
│   └── lora_trainer.py      (~315줄)  LoRA 트레이너 (DataLoader 흡수)
├── exporter/
│   ├── __init__.py          (6줄)
│   ├── hf_export.py         (155줄)   HuggingFace 내보내기
│   └── ollama_export.py     (177줄)   Ollama 내보내기
```

### 1.2 모듈 의존성 요약

| 모듈 | 의존하는 모듈 | 사용하는 외부 패키지 |
|------|--------------|-------------------|
| config | - | pydantic, yaml |
| cli | config, pipeline, utils | typer, rich |
| pipeline | parsers, teacher, validator, scorer, augmenter, analyzer, converter, trainer, exporter, utils | - |
| converter | config, models, utils | transformers |
| models | - | dataclasses |
| utils | - | logging, rich |
| parsers | models, utils | fitz, bs4, lxml, zipfile, pykospacing (선택) |
| teacher | config, models, utils | httpx |
| validator | config, models, utils | sentence_transformers (선택) |
| scorer | config, models, teacher, utils | - |
| augmenter | config, models, teacher, utils | - |
| analyzer | models, utils | - |
| trainer | config, models, utils | torch, transformers, datasets, peft, trl |
| exporter | config, utils | torch, transformers, peft, subprocess |

### 1.3 데이터 흐름

```
문서 파일 (PDF/HWPX/HTML/TXT)
    ↓ parsers/
ParsedDocument 객체 (models.py에서 정의)
    ↓ teacher/qa_generator.py
QAPair 객체 (models.py에서 정의, Alpaca 형식)
    ↓ validator/rules.py + similarity.py
검증된 QAPair
    ↓ scorer.py (선택적)
품질 점수 평가 및 필터링
    ↓ augmenter.py (선택적)
데이터 증강 (질문 패러프레이즈)
    ↓ analyzer.py (선택적)
학습 데이터 분석 및 보고서
    ↓ converter.py
채팅 템플릿 JSONL
    ↓ trainer/lora_trainer.py
LoRA 어댑터
    ↓ exporter/hf_export.py + ollama_export.py
병합된 모델 + Ollama Modelfile
```

---

## 2. config.py — 설정 시스템 (298줄)

### 2.1 역할

YAML 설정 파일을 Pydantic v2 모델로 로드하고 검증합니다. 전체 파이프라인의 동작을 제어하는 중앙 설정 시스템입니다. 모든 모듈은 이 설정 객체를 통해 파라미터를 전달받습니다.

### 2.2 주요 클래스

#### SLMConfig (루트 설정 객체)

전체 프로젝트 설정을 담는 최상위 Pydantic 모델입니다. 9개의 하위 설정 모델을 포함합니다.

```python
class SLMConfig(BaseModel):
    project: ProjectConfig
    paths: PathsConfig
    parsing: ParsingConfig
    teacher: TeacherConfig
    questions: QuestionsConfig
    validation: ValidationConfig
    student: StudentConfig
    training: TrainingConfig
    export: ExportConfig
```

**주요 메서드:**
- `@model_validator(mode="before")` `_strip_none_sections`: YAML에서 `null` 값으로 설정된 섹션을 자동으로 제거하여 기본값이 적용되도록 합니다.

#### ProjectConfig

프로젝트 메타데이터를 정의합니다.

```python
class ProjectConfig(BaseModel):
    name: str = "my-project"
    version: str = "1.0.0"
    language: str = "en"
```

#### PathsConfig

입출력 경로를 관리합니다.

```python
class PathsConfig(BaseModel):
    documents: str = "documents"
    output: str = "output"
    
    def ensure_dirs(self) -> None:
        """출력 디렉토리를 자동으로 생성합니다."""
        Path(self.output).mkdir(parents=True, exist_ok=True)
```

**사용 예시:**
```python
config.paths.ensure_dirs()  # output/ 디렉토리 생성
```

#### ParsingConfig

문서 파싱 옵션을 설정합니다.

```python
class PdfOptions(BaseModel):
    extract_tables: bool = True

class HwpxOptions(BaseModel):
    apply_spacing: bool = True

class ParsingConfig(BaseModel):
    formats: list[str] = ["pdf"]
    pdf: PdfOptions = PdfOptions()
    hwpx: HwpxOptions = HwpxOptions()
```

#### TeacherConfig

Teacher LLM 백엔드 설정을 정의합니다.

```python
class TeacherConfig(BaseModel):
    backend: Literal["ollama", "openai"] = "ollama"
    model: str = "qwen3:8b"
    api_base: str = "http://localhost:11434"
    api_key: str | None = None
    temperature: float = 0.3
    timeout: int = 180
    max_context_chars: int = 12000
    max_concurrency: int = 4
```

**지원 백엔드:**
- `"ollama"`: 로컬 Ollama 서버
- `"openai"`: OpenAI 호환 API (vLLM, LiteLLM, OpenRouter 등)

#### QuestionsConfig

QA 생성에 사용할 질문 목록을 관리합니다.

```python
class QuestionsConfig(BaseModel):
    categories: dict[str, list[str]] = {}
    file: Path | None = None
    system_prompt: str = "You are a helpful assistant that answers..."
    output_format: str = "alpaca"
    
    def get_all_questions(self) -> list[str]:
        """모든 카테고리 질문을 단일 리스트로 평탄화합니다."""
        if self.file is not None:
            path = Path(self.file)
            if path.is_file():
                return [line.strip() for line in path.read_text(...).splitlines() if line.strip()]
        return [q for questions in self.categories.values() for q in questions]
```

**사용 패턴:**
- `categories`에 카테고리별 질문 딕셔너리 직접 작성 (소규모 프로젝트)
- `file`로 외부 텍스트 파일 참조 (줄당 하나의 질문, 대규모 프로젝트)

#### ValidationConfig

QA 쌍 검증 규칙을 설정합니다.

```python
class GroundednessConfig(BaseModel):
    enabled: bool = False
    model: str = "all-MiniLM-L6-v2"
    threshold: float = 0.3

class ValidationConfig(BaseModel):
    enabled: bool = True
    min_answer_length: int = 20
    max_answer_length: int = 2000
    remove_empty: bool = True
    deduplicate: bool = True
    reject_patterns: list[str] = [
        "(?i)i don't know",
        "(?i)not (available|provided|mentioned|found)",
        "(?i)the document does not contain",
    ]
    groundedness: GroundednessConfig = GroundednessConfig()
```

**검증 단계:**
1. 빈값 제거 (`remove_empty`)
2. 길이 검사 (`min_answer_length`, `max_answer_length`)
3. 패턴 매칭 (`reject_patterns`)
4. 중복 제거 (`deduplicate`)
5. 근거성 검사 (`groundedness.enabled`)

#### StudentConfig

학습할 Student 모델을 지정합니다.

```python
class StudentConfig(BaseModel):
    model: str = "google/gemma-3-1b-it"
    max_seq_length: int = 4096
```

#### TrainingConfig

LoRA 학습 하이퍼파라미터를 설정합니다.

```python
class LoraConfig(BaseModel):
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: str | list[str] = "auto"
    use_rslora: bool = False

class EarlyStoppingConfig(BaseModel):
    enabled: bool = True
    patience: int = 3
    threshold: float = 0.01

class QuantizationConfig(BaseModel):
    enabled: bool = False
    bits: int = 4

class TrainingConfig(BaseModel):
    lora: LoraConfig = LoraConfig()
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-5
    lr_scheduler: str = "cosine"
    warmup_ratio: float = 0.1
    num_epochs: int = 20
    early_stopping: EarlyStoppingConfig = EarlyStoppingConfig()
    optimizer: str = "adamw_torch_fused"
    bf16: bool = True
    train_split: float = 0.9
    save_strategy: str = "epoch"
    quantization: QuantizationConfig = QuantizationConfig()
```

**주요 설정:**
- `target_modules: "auto"`: PEFT가 자동으로 LoRA 적용 레이어 선택
- `target_modules: ["q_proj", "v_proj"]`: 수동으로 레이어 지정
- `use_rslora: True`: Rank-Stabilized LoRA 사용
- `quantization.enabled: True`: 4-bit NF4 양자화 활성화

#### ExportConfig

모델 내보내기 옵션을 설정합니다.

```python
class OllamaExportConfig(BaseModel):
    enabled: bool = True
    model_name: str = "my-project-model"
    system_prompt: str = "You are a helpful domain-specific assistant."
    parameters: dict[str, Any] = {"temperature": 0.7, "top_p": 0.9, "num_ctx": 4096}

class ExportConfig(BaseModel):
    merge_lora: bool = True
    output_format: str = "safetensors"
    ollama: OllamaExportConfig = OllamaExportConfig()
```

**내보내기 모드:**
- `merge_lora: True`: 어댑터를 기본 모델에 병합
- `merge_lora: False`: 어댑터만 저장 (PEFT 형식)

### 2.3 주요 함수

#### load_config(path: str | Path) → SLMConfig

YAML 파일을 읽어 `SLMConfig` 객체로 변환합니다.

```python
def load_config(path: str | Path) -> SLMConfig:
    """YAML 설정 파일을 로드하고 검증합니다."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return SLMConfig(**data)
```

**에러 처리:**
- `FileNotFoundError`: 파일이 없을 때
- `ValidationError`: Pydantic 검증 실패 시 (필수 필드 누락, 타입 불일치 등)

#### create_default_config() → str

기본 설정 템플릿을 문자열로 반환합니다.

```python
def create_default_config() -> str:
    """번들된 기본 설정 템플릿을 반환합니다."""
    # 1. 소스 디렉토리에서 시도
    # 2. 설치된 패키지에서 시도
    # 3. 하드코딩된 fallback 반환
```

**사용 위치:**
- `cli.py`의 `init` 명령어에서 `project.yaml` 생성 시 사용

### 2.4 설계 포인트

#### 계층적 구조

9개의 독립적인 설정 섹션으로 분리하여 관심사를 명확히 구분합니다. 각 섹션은 독립적으로 수정 가능하며, `None` 값으로 설정하면 기본값이 적용됩니다.

#### 타입 안전성

Pydantic v2의 강력한 타입 검증을 활용하여 런타임 에러를 사전에 방지합니다. 모든 필드는 명시적 타입 힌트를 가지며, 잘못된 값이 입력되면 즉시 `ValidationError`가 발생합니다.

#### 유연한 기본값

모든 필드에 합리적인 기본값을 제공하여 최소한의 설정만으로도 파이프라인을 실행할 수 있습니다. 사용자는 필요한 부분만 오버라이드하면 됩니다.

#### 메서드 통합

단순 데이터 컨테이너를 넘어 유틸리티 메서드를 제공합니다:
- `PathsConfig.ensure_dirs()`: 디렉토리 생성
- `QuestionsConfig.get_all_questions()`: 질문 로드 로직 캡슐화

---

## 3. cli.py — CLI 인터페이스 (331줄)

### 3.1 역할

사용자 진입점입니다. Typer 프레임워크를 기반으로 10개의 CLI 명령어를 제공하며, Rich 라이브러리를 사용하여 시각적으로 풍부한 출력을 생성합니다.

### 3.2 주요 구성

```python
import typer
from rich.console import Console

app = typer.Typer(
    name="slm-factory",
    help="Small Language Model Factory - 문서에서 SLM까지 자동화"
)
console = Console()
```

**전역 객체:**
- `app`: Typer 애플리케이션 루트
- `console`: Rich 콘솔 출력 인스턴스

### 3.3 헬퍼 함수

#### _load_pipeline(config_path: str) → Pipeline

설정 파일을 로드하고 `Pipeline` 인스턴스를 생성합니다.

```python
def _load_pipeline(config_path: str) -> Pipeline:
    """설정을 로드하고 파이프라인을 초기화합니다."""
    config = load_config(config_path)
    setup_logging()
    return Pipeline(config)
```

**사용 위치:**
- `run`, `parse`, `generate`, `validate`, `train` 명령어에서 공통으로 사용

### 3.4 명령어 상세

#### 1. init — 프로젝트 초기화

```python
@app.command()
def init(
    name: str = typer.Argument(..., help="프로젝트 이름"),
    path: str = typer.Option(".", help="프로젝트 생성 경로")
):
    """새 SLM Factory 프로젝트를 초기화합니다."""
```

**동작:**
1. `{path}/{name}/` 디렉토리 생성
2. 하위 디렉토리 생성: `documents/`, `output/`
3. `project.yaml` 템플릿 작성 (프로젝트 이름 자동 치환)
4. 성공 메시지 출력

**출력 예시:**
```
Project 'my-project' created at ./my-project

프로젝트 구조:
  ./my-project/
  ./my-project/documents/
  ./my-project/output/
  ./my-project/project.yaml

다음 단계:
  1. ./my-project/documents/에 문서 추가
  2. ./my-project/project.yaml를 편집하여 설정 커스터마이징
  3. 실행: slm-factory run --config ./my-project/project.yaml
```

#### 2. run — 전체 파이프라인 실행

```python
@app.command()
def run(
    config: str = typer.Argument(..., help="설정 파일 경로 (project.yaml)")
):
    """전체 6단계 파이프라인을 실행합니다."""
```

**동작:**
1. 설정 로드 및 파이프라인 초기화
2. `pipeline.run()` 호출 (6단계 순차 실행)
3. 최종 모델 경로 출력

**에러 처리:**
```python
try:
    pipeline = _load_pipeline(config)
    final_path = pipeline.run()
    console.print(f"[bold green]✓ 완료![/] 모델: {final_path}")
except FileNotFoundError as e:
    console.print(f"[bold red]Error:[/] {e}")
    raise typer.Exit(1)
except Exception as e:
    console.print(f"[bold red]Pipeline failed:[/] {e}")
    raise typer.Exit(1)
```

#### 3. parse — 문서 파싱만 실행

```python
@app.command()
def parse(
    config: str = typer.Argument(..., help="설정 파일 경로")
):
    """문서를 파싱하여 ParsedDocument 객체로 변환합니다."""
```

**동작:**
1. `pipeline.step_parse()` 호출
2. `output/parsed_documents.json` 저장
3. 파싱된 문서 수 출력

**사용 사례:**
- 파싱 결과만 확인하고 싶을 때
- QA 생성 전에 문서 품질 검증

#### 4. generate — QA 생성까지 실행

```python
@app.command()
def generate(
    config: str = typer.Argument(..., help="설정 파일 경로")
):
    """문서를 파싱하고 QA 쌍을 생성합니다."""
```

**동작:**
1. `pipeline.step_parse()` → 문서 파싱
2. `pipeline.step_generate(docs)` → QA 생성
3. `output/qa_alpaca.json` 저장
4. 생성된 QA 쌍 수 출력

#### 5. validate — 검증까지 실행

```python
@app.command()
def validate(
    config: str = typer.Argument(..., help="설정 파일 경로")
):
    """문서 파싱, QA 생성, 검증을 수행합니다."""
```

**동작:**
1. `pipeline.step_parse()` → 문서 파싱
2. `pipeline.step_generate(docs)` → QA 생성
3. `pipeline.step_validate(pairs, docs)` → 검증
4. 검증 통과/거부 통계 출력

#### 6. train — 학습 실행

```python
@app.command()
def train(
    config: str = typer.Argument(..., help="설정 파일 경로"),
    data: str | None = typer.Option(None, help="기존 학습 데이터 JSONL 경로")
):
    """LoRA 파인튜닝을 수행합니다."""
```

**동작:**
- `--data` 옵션이 있으면: 기존 JSONL 파일 직접 사용
- `--data` 옵션이 없으면: 파싱 → 생성 → 검증 → 변환 후 학습

**사용 예시:**
```bash
# 전체 파이프라인 후 학습
slm-factory train project.yaml

# 기존 데이터로 학습
slm-factory train project.yaml --data output/training_data.jsonl
```

#### 7. version — 버전 출력

```python
@app.command()
def version():
    """slm-factory 버전을 출력합니다."""
    from slm_factory import __version__
    console.print(f"slm-factory version {__version__}")
```

### 3.5 진입점 설정

`pyproject.toml`에서 CLI 진입점을 정의합니다:

```toml
[project.scripts]
slm-factory = "slm_factory.cli:app"
```

설치 후 `slm-factory` 명령어로 직접 실행 가능합니다.

### 3.6 에러 처리 패턴

모든 명령어는 일관된 에러 처리 패턴을 따릅니다:

1. **FileNotFoundError**: 설정 파일 또는 데이터 파일이 없을 때
   ```python
   console.print(f"[bold red]Error:[/] {e}")
   raise typer.Exit(1)
   ```

2. **일반 예외**: 파이프라인 실행 중 발생한 모든 에러
   ```python
   console.print(f"[bold red]Pipeline failed:[/] {e}")
   raise typer.Exit(1)
   ```

3. **종료 코드**: 에러 발생 시 항상 `1` 반환 (스크립트 통합 용이)

---

## 4. pipeline.py — 파이프라인 오케스트레이터 (331줄)

### 4.1 역할

9단계 파이프라인의 순차 실행을 관리합니다. 각 단계를 독립 메서드로 제공하여 CLI에서 개별 호출이 가능하며, 중간 결과를 JSON/JSONL 파일로 저장하여 디버깅과 재개를 지원합니다. 검증 후 선택적으로 품질 점수 평가(step_score), 데이터 증강(step_augment), 분석(step_analyze) 단계를 수행할 수 있습니다.

### 4.2 Pipeline 클래스

```python
class Pipeline:
    def __init__(self, config: SLMConfig):
        self.config = config
        self.output_dir = Path(config.paths.output)
        self.logger = get_logger("pipeline")
```

**초기화:**
- 설정 객체 저장
- 출력 디렉토리 경로 설정
- 로거 생성

### 4.3 9단계 파이프라인

#### Step 1: step_parse() → list[ParsedDocument]

문서 파일을 파싱하여 구조화된 객체로 변환합니다.

```python
def step_parse(self) -> list[ParsedDocument]:
    """문서를 파싱합니다."""
    self.logger.info("Step 1/9: 문서 파싱 시작")
    
    docs_dir = Path(self.config.paths.documents)
    formats = self.config.parsing.formats
    
    docs = registry.parse_directory(docs_dir, formats)
    
    # 중간 결과 저장
    output_path = self.output_dir / "parsed_documents.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump([asdict(doc) for doc in docs], f, ensure_ascii=False, indent=2)
    
    self.logger.info(f"파싱 완료: {len(docs)}개 문서 → {output_path}")
    return docs
```

**출력 파일:**
- `output/parsed_documents.json`: 파싱된 문서 목록 (디버깅용)

#### Step 2: step_generate(docs) → list[QAPair]

Teacher LLM을 사용하여 QA 쌍을 생성합니다.

```python
def step_generate(self, docs: list[ParsedDocument]) -> list[QAPair]:
    """QA 쌍을 생성합니다."""
    self.logger.info("Step 2/9: QA 생성 시작")
    
    generator = QAGenerator(self.config)
    questions = self.config.questions.get_all_questions()
    
    pairs = generator.generate_all(docs, questions)
    
    # Alpaca 형식으로 저장
    output_path = self.output_dir / "qa_alpaca.json"
    generator.save_alpaca(pairs, output_path)
    
    self.logger.info(f"QA 생성 완료: {len(pairs)}개 쌍 → {output_path}")
    return pairs
```

**출력 파일:**
- `output/qa_alpaca.json`: Alpaca 형식 QA 데이터

#### Step 3: step_validate(pairs, docs) → list[QAPair]

규칙 기반 검증과 선택적 근거성 검사를 수행합니다.

```python
def step_validate(
    self, 
    pairs: list[QAPair], 
    docs: list[ParsedDocument]
) -> list[QAPair]:
    """QA 쌍을 검증합니다."""
    self.logger.info("Step 3/9: 검증 시작")
    
    if not self.config.validation.enabled:
        self.logger.info("검증 비활성화됨 - 건너뜀")
        return pairs
    
    # 규칙 검증
    rule_validator = RuleValidator(self.config.validation)
    accepted, rejected = rule_validator.validate_batch(pairs)
    
    # 근거성 검증 (선택적)
    if self.config.validation.groundedness.enabled:
        checker = GroundednessChecker(self.config.validation.groundedness)
        source_texts = {doc.doc_id: doc.content for doc in docs}
        grounded, ungrounded = checker.check_batch(accepted, source_texts)
        
        self.logger.info(f"근거성 검증: {len(grounded)} 통과, {len(ungrounded)} 거부")
        return grounded
    
    return accepted
```

**검증 단계:**
1. 규칙 검증 (빈값, 길이, 패턴, 중복)
2. 근거성 검증 (선택적, 임베딩 유사도)

#### Step 4: step_convert(pairs) → Path

Alpaca 형식을 채팅 템플릿 JSONL로 변환합니다.

```python
def step_convert(self, pairs: list[QAPair]) -> Path:
    """채팅 템플릿으로 변환합니다."""
    self.logger.info("Step 7/9: 형식 변환 시작")
    
    formatter = ChatFormatter(self.config)
    output_path = self.output_dir / "training_data.jsonl"
    
    formatter.save_training_data(pairs, output_path)
    
    self.logger.info(f"변환 완료 → {output_path}")
    return output_path
```

**출력 파일:**
- `output/training_data.jsonl`: 학습용 채팅 템플릿 데이터

#### Step 5: step_train(training_data_path) → Path

LoRA 파인튜닝을 수행합니다.

```python
def step_train(self, training_data_path: Path) -> Path:
    """LoRA 학습을 수행합니다."""
    self.logger.info("Step 8/9: 학습 시작")
    
    # 데이터 로드
    loader = DataLoader(train_split=self.config.training.train_split)
    dataset_dict = loader.load_and_split(training_data_path)
    
    # 학습
    trainer = LoRATrainer(self.config)
    adapter_path = trainer.train(dataset_dict)
    
    self.logger.info(f"학습 완료 → {adapter_path}")
    return adapter_path
```

**출력 디렉토리:**
- `output/checkpoints/adapter/`: LoRA 어댑터 가중치

#### Step 6: step_export(adapter_path) → Path

모델을 병합하고 Ollama Modelfile을 생성합니다.

```python
def step_export(self, adapter_path: Path) -> Path:
    """모델을 내보냅니다."""
    self.logger.info("Step 9/9: 내보내기 시작")
    
    export_dir = self.output_dir / "final_model"
    
    # HuggingFace 형식 내보내기
    hf_exporter = HFExporter(self.config)
    model_path = hf_exporter.export(adapter_path, export_dir)
    
    # Ollama Modelfile 생성 (선택적)
    if self.config.export.ollama.enabled:
        ollama_exporter = OllamaExporter(self.config)
        ollama_exporter.export(model_path, export_dir)
    
    self.logger.info(f"내보내기 완료 → {model_path}")
    return model_path
```

**출력 디렉토리:**
- `output/final_model/`: 병합된 모델 또는 어댑터
- `output/final_model/Modelfile`: Ollama 모델 정의 파일

### 4.4 전체 파이프라인 실행

#### run() → Path

6단계를 순차적으로 실행하고 최종 모델 경로를 반환합니다.

```python
def run(self) -> Path:
    """전체 파이프라인을 실행합니다."""
    start_time = time.time()
    
    try:
        self.config.paths.ensure_dirs()
        
        # Step 1-6 순차 실행
        docs = self.step_parse()
        pairs = self.step_generate(docs)
        validated = self.step_validate(pairs, docs)
        training_data = self.step_convert(validated)
        adapter = self.step_train(training_data)
        final_model = self.step_export(adapter)
        
        elapsed = time.time() - start_time
        self.logger.info(f"파이프라인 완료 (소요 시간: {elapsed:.1f}초)")
        
        return final_model
        
    except Exception as e:
        self.logger.error(f"파이프라인 실패: {e}", exc_info=True)
        raise
```

**특징:**
- 각 단계의 출력이 다음 단계의 입력으로 전달
- 예외 발생 시 스택 트레이스 로깅 후 재발생
- 전체 실행 시간 측정

### 4.5 중간 저장 파일

각 단계에서 중간 결과를 저장하여 다음과 같은 이점을 제공합니다:

| 파일 | 단계 | 용도 |
|------|------|------|
| `parsed_documents.json` | 1 | 파싱 결과 검증, 재생성 없이 QA 생성 재시도 |
| `qa_alpaca.json` | 2 | 생성된 QA 품질 확인, 검증 규칙 조정 |
| `training_data.jsonl` | 4 | 학습 데이터 직접 확인, 외부 도구로 분석 |

**재개 시나리오:**
```python
# 기존 학습 데이터로 학습만 재실행
pipeline = Pipeline(config)
adapter = pipeline.step_train(Path("output/training_data.jsonl"))
final_model = pipeline.step_export(adapter)
```

---

## 5. parsers/ — 문서 파서 모듈

### 5.1 base.py (183줄)

#### 5.1.1 역할

모든 파서의 기본 클래스, 데이터 구조, 레지스트리를 정의합니다. 플러그인 아키텍처를 통해 새로운 파서를 쉽게 추가할 수 있습니다.

#### 5.1.2 ParsedDocument (dataclass)

파싱된 문서의 표준 데이터 구조입니다.

```python
@dataclass
class ParsedDocument:
    doc_id: str              # 파일명 기반 고유 식별자
    title: str               # 문서 제목
    content: str             # 마크다운 형식 본문
    tables: list[str] = field(default_factory=list)  # 마크다운 표 목록
    metadata: dict = field(default_factory=dict)     # 추가 메타데이터
```

**필드 설명:**
- `doc_id`: 파일명에서 확장자를 제거한 값 (예: `"report_20240115"`)
- `title`: 문서 제목 (메타데이터 또는 파일명에서 추출)
- `content`: 전체 텍스트 (마크다운 형식, 표는 별도 저장)
- `tables`: 추출된 표 목록 (각 표는 마크다운 문자열)
- `metadata`: `{"author": "...", "date": "2024-01-15", "page_count": 10}` 등

#### 5.1.3 BaseParser (ABC)

모든 파서가 상속해야 하는 추상 기본 클래스입니다.

```python
class BaseParser(ABC):
    extensions: ClassVar[list[str]]  # 처리 가능한 확장자
    
    @abstractmethod
    def parse(self, path: Path) -> ParsedDocument:
        """파일을 파싱하여 ParsedDocument를 반환합니다."""
        pass
    
    def can_parse(self, path: Path) -> bool:
        """이 파서가 해당 파일을 처리할 수 있는지 확인합니다."""
        return path.suffix.lower() in self.extensions
```

**구현 규칙:**
- `extensions` 클래스 변수에 처리 가능한 확장자 목록 정의
- `parse()` 메서드에서 실제 파싱 로직 구현
- `can_parse()`는 기본 구현 사용 (확장자 매칭)

#### 5.1.4 ParserRegistry

등록된 파서를 관리하고 파일에 맞는 파서를 자동으로 선택합니다.

```python
class ParserRegistry:
    def __init__(self):
        self._parsers: list[BaseParser] = []
    
    def register(self, parser_cls: type[BaseParser]) -> type[BaseParser]:
        """파서 클래스를 등록합니다 (데코레이터로 사용)."""
        self._parsers.append(parser_cls())
        return parser_cls
    
    def get_parser(self, path: Path) -> BaseParser | None:
        """파일 확장자에 맞는 파서를 반환합니다."""
        for parser in self._parsers:
            if parser.can_parse(path):
                return parser
        return None
```

**사용 예시:**
```python
registry = ParserRegistry()

@registry.register
class PDFParser(BaseParser):
    extensions = [".pdf"]
    ...
```

#### 5.1.5 parse_directory() — 디렉토리 일괄 파싱

```python
def parse_directory(
    self, 
    dir_path: Path, 
    formats: list[str]
) -> list[ParsedDocument]:
    """디렉토리의 모든 문서를 파싱합니다."""
    
    # 1. 파일 수집
    files = []
    for fmt in formats:
        files.extend(dir_path.glob(f"**/*{fmt}"))
    
    # 2. Rich Progress 바 표시
    docs = []
    with Progress() as progress:
        task = progress.add_task("파싱 중...", total=len(files))
        
        for file_path in files:
            try:
                parser = self.get_parser(file_path)
                if parser:
                    doc = parser.parse(file_path)
                    docs.append(doc)
            except Exception as e:
                logger.error(f"파싱 실패: {file_path} - {e}")
            finally:
                progress.advance(task)
    
    return docs
```

**특징:**
- 재귀적 디렉토리 스캔 (`**/*` glob 패턴)
- 개별 파일 파싱 실패 시 격리 (전체 프로세스 중단 없음)
- Rich Progress 바로 진행 상황 시각화

#### 5.1.6 extract_date_from_filename() — 날짜 추출

파일명에서 날짜 패턴을 추출합니다.

```python
def extract_date_from_filename(filename: str) -> str | None:
    """파일명에서 YYMMDD 패턴을 찾아 ISO 형식으로 반환합니다."""
    pattern = r"(\d{2})(\d{2})(\d{2})"
    match = re.search(pattern, filename)
    if match:
        yy, mm, dd = match.groups()
        return f"20{yy}-{mm}-{dd}"
    return None
```

**예시:**
- `"report_240115.pdf"` → `"2024-01-15"`
- `"memo.pdf"` → `None`

### 5.2 pdf.py (165줄)

#### 5.2.1 역할

PyMuPDF(fitz) 라이브러리를 사용하여 PDF 파일에서 텍스트와 표를 추출합니다.

#### 5.2.2 PDFParser 클래스

```python
@registry.register
class PDFParser(BaseParser):
    extensions = [".pdf"]
    
    def parse(self, path: Path) -> ParsedDocument:
        """PDF 파일을 파싱합니다."""
        doc = fitz.open(path)
        
        # 텍스트 추출
        content_parts = []
        tables = []
        
        for page in doc:
            # 페이지 텍스트
            text = page.get_text("text")
            cleaned = self._clean_page_numbers(text)
            content_parts.append(cleaned)
            
            # 표 추출
            if self.config.parsing.pdf.extract_tables:
                for table in page.find_tables():
                    md_table = self._table_to_markdown(table)
                    tables.append(md_table)
        
        # 메타데이터 추출
        metadata = {
            "author": doc.metadata.get("author", ""),
            "creation_date": doc.metadata.get("creationDate", ""),
            "page_count": len(doc),
            "date": extract_date_from_filename(path.name)
        }
        
        return ParsedDocument(
            doc_id=path.stem,
            title=doc.metadata.get("title", path.stem),
            content="\n\n".join(content_parts),
            tables=tables,
            metadata=metadata
        )
```

#### 5.2.3 _clean_page_numbers() — 페이지 번호 제거

```python
def _clean_page_numbers(self, text: str) -> str:
    """페이지 번호 패턴을 제거합니다."""
    # 패턴 1: "- 1 -" 형식
    text = re.sub(r"-\s*\d+\s*-", "", text)
    
    # 패턴 2: 줄 끝의 순수 숫자
    text = re.sub(r"^\d+$", "", text, flags=re.MULTILINE)
    
    # 패턴 3: "Page N" 형식
    text = re.sub(r"Page\s+\d+", "", text, flags=re.IGNORECASE)
    
    return text
```

#### 5.2.4 _table_to_markdown() — 표 변환

```python
def _table_to_markdown(self, table) -> str:
    """fitz.Table 객체를 마크다운 표로 변환합니다."""
    rows = table.extract()
    if not rows:
        return ""
    
    # 헤더
    header = "| " + " | ".join(rows[0]) + " |"
    separator = "|" + "|".join(["---"] * len(rows[0])) + "|"
    
    # 데이터 행
    body = []
    for row in rows[1:]:
        body.append("| " + " | ".join(row) + " |")
    
    return "\n".join([header, separator] + body)
```

### 5.3 hwpx.py (181줄)

#### 5.3.1 역할

한국어 HWPX 문서(ZIP 아카이브)에서 텍스트와 표를 추출합니다. 선택적으로 `pykospacing`을 사용하여 띄어쓰기를 보정합니다.

#### 5.3.2 HWPXParser 클래스

```python
@registry.register
class HWPXParser(BaseParser):
    extensions = [".hwpx"]
    
    def parse(self, path: Path) -> ParsedDocument:
        """HWPX 파일을 파싱합니다."""
        with zipfile.ZipFile(path) as zf:
            # section0.xml 읽기
            xml_content = zf.read("Contents/section0.xml")
            soup = BeautifulSoup(xml_content, "xml")
            
            # 텍스트 추출
            paragraphs = []
            for p_tag in soup.find_all("hp:p"):
                text_parts = [t.get_text() for t in p_tag.find_all("hp:t")]
                para = "".join(text_parts)
                
                # 띄어쓰기 보정 (선택적)
                if HAS_PYKOSPACING:
                    para = spacing(para)
                
                paragraphs.append(para)
            
            # 표 추출
            tables = []
            for tbl in soup.find_all("hp:tbl"):
                md_table = self._table_to_markdown(tbl)
                tables.append(md_table)
            
            return ParsedDocument(
                doc_id=path.stem,
                title=path.stem,
                content="\n\n".join(paragraphs),
                tables=tables,
                metadata={"date": extract_date_from_filename(path.name)}
            )
```

#### 5.3.3 pykospacing 통합

```python
try:
    from pykospacing import spacing
    HAS_PYKOSPACING = True
except ImportError:
    HAS_PYKOSPACING = False
```

**설치:**
```bash
pip install slm-factory[korean]
```

**동작:**
- `apply_spacing: true` + pykospacing 설치됨 → 띄어쓰기 보정 적용
- `apply_spacing: false` 또는 미설치 → 원본 텍스트 사용

### 5.4 html.py (180줄)

#### 5.4.1 역할

BeautifulSoup을 사용하여 HTML 파일에서 텍스트와 표를 추출합니다.

#### 5.4.2 HTMLParser 클래스

```python
@registry.register
class HTMLParser(BaseParser):
    extensions = [".html", ".htm"]
    
    def parse(self, path: Path) -> ParsedDocument:
        """HTML 파일을 파싱합니다."""
        encoding = self._detect_encoding(path)
        with open(path, encoding=encoding) as f:
            soup = BeautifulSoup(f, "html.parser")
        
        # 불필요한 태그 제거
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        
        # 제목 추출 (우선순위: title > h1 > 파일명)
        title = path.stem
        if soup.title:
            title = soup.title.get_text().strip()
        elif soup.h1:
            title = soup.h1.get_text().strip()
        
        # 텍스트 추출
        content = soup.get_text(separator="\n", strip=True)
        
        # 표 추출
        tables = []
        for table in soup.find_all("table"):
            md_table = self._table_to_markdown(table)
            tables.append(md_table)
        
        return ParsedDocument(
            doc_id=path.stem,
            title=title,
            content=content,
            tables=tables,
            metadata={}
        )
```

#### 5.4.3 _detect_encoding() — 인코딩 감지

```python
def _detect_encoding(self, path: Path) -> str:
    """파일 인코딩을 감지합니다."""
    with open(path, "rb") as f:
        raw = f.read(10000)
        result = chardet.detect(raw)
        return result["encoding"] or "utf-8"
```

### 5.5 text.py (101줄)

#### 5.5.1 역할

일반 텍스트와 마크다운 파일을 읽습니다.

#### 5.5.2 TextParser 클래스

```python
@registry.register
class TextParser(BaseParser):
    extensions = [".txt", ".md"]
    
    def parse(self, path: Path) -> ParsedDocument:
        """텍스트 파일을 파싱합니다."""
        encoding = self._detect_encoding(path)
        with open(path, encoding=encoding) as f:
            content = f.read()
        
        # 마크다운 제목 추출
        title = path.stem
        if path.suffix == ".md":
            match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
            if match:
                title = match.group(1)
        
        return ParsedDocument(
            doc_id=path.stem,
            title=title,
            content=content,
            tables=[],  # 텍스트 파일은 표 추출 없음
            metadata={"date": extract_date_from_filename(path.name)}
        )
```

### 5.6 __init__.py (24줄)

#### 5.6.1 역할

전역 레지스트리를 생성하고 모든 파서를 등록합니다.

```python
from .base import ParserRegistry, ParsedDocument
from .pdf import PDFParser
from .hwpx import HWPXParser
from .html import HTMLParser
from .text import TextParser

# 전역 레지스트리 생성
registry = ParserRegistry()

# 파서 등록 (순서대로)
registry.register(PDFParser)
registry.register(HWPXParser)
registry.register(HTMLParser)
registry.register(TextParser)

__all__ = ["registry", "ParsedDocument"]
```

**등록 순서:**
1. PDFParser
2. HWPXParser
3. HTMLParser
4. TextParser

---

## 6. teacher/ — Teacher LLM 모듈

### 6.1 base.py (36줄)

#### 6.1.1 역할

모든 Teacher 백엔드의 추상 인터페이스를 정의합니다.

#### 6.1.2 BaseTeacher (ABC)

```python
class BaseTeacher(ABC):
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """프롬프트를 전송하고 응답 텍스트를 반환합니다."""
        pass
    
    def health_check(self) -> bool:
        """백엔드 연결 상태를 확인합니다."""
        return True  # 기본 구현
```

**구현 규칙:**
- `generate()`: 프롬프트 전송 + 응답 텍스트 반환 (필수)
- `health_check()`: 백엔드 연결 확인 (선택적, 기본 True)

### 6.2 ollama.py (121줄)

#### 6.2.1 역할

로컬 Ollama 서버의 REST API를 호출합니다.

#### 6.2.2 OllamaTeacher 클래스

```python
class OllamaTeacher(BaseTeacher):
    _STOP_TOKENS = [
        "</think>", "<think>", 
        "Reasoning:", "Let me think", "Step by step:"
    ]
    
    def __init__(self, config: TeacherConfig):
        self.model = config.model
        self.api_base = config.api_base.rstrip("/")
        self.temperature = config.temperature
        self.timeout = config.timeout
        self.client = httpx.Client(timeout=timeout)
```

#### 6.2.3 generate() — 텍스트 생성

```python
def generate(self, prompt: str, **kwargs) -> str:
    """Ollama API로 텍스트를 생성합니다."""
    url = f"{self.api_base}/api/generate"
    
    payload = {
        "model": self.model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": self.temperature,
            "stop": self._STOP_TOKENS
        }
    }
    
    # JSON 모드 지원
    if "format" in kwargs:
        payload["format"] = kwargs["format"]
    
    try:
        response = self.client.post(url, json=payload)
        response.raise_for_status()
        return response.json()["response"]
    except (TimeoutException, ConnectError, HTTPStatusError) as e:
        raise RuntimeError(f"Ollama API 호출 실패: {e}")
```

**특징:**
- `stream: false`: 전체 응답을 한 번에 반환
- `stop`: chain-of-thought 토큰 차단 (불필요한 추론 과정 제거)
- `format: "json"`: JSON 모드 활성화 (QA 생성 시 사용)

#### 6.2.4 health_check() — 연결 확인

```python
def health_check(self) -> bool:
    """Ollama 서버 연결을 확인합니다."""
    try:
        response = self.client.get(f"{self.api_base}/api/tags")
        return response.status_code == 200
    except Exception:
        return False
```

### 6.3 openai_compat.py (132줄)

#### 6.3.1 역할

OpenAI 호환 API(vLLM, LiteLLM, OpenRouter, OpenAI)를 호출합니다.

#### 6.3.2 OpenAICompatTeacher 클래스

```python
class OpenAICompatTeacher(BaseTeacher):
    def __init__(self, config: TeacherConfig):
        self.model = config.model
        self.api_base = config.api_base.rstrip("/")
        self.api_key = config.api_key
        self.temperature = config.temperature
        self.timeout = config.timeout
        self.client = httpx.Client(timeout=timeout)
```

#### 6.3.3 generate() — 텍스트 생성

```python
def generate(self, prompt: str, **kwargs) -> str:
    """OpenAI 호환 API로 텍스트를 생성합니다."""
    url = f"{self.api_base}/v1/chat/completions"
    
    payload = {
        "model": self.model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": self.temperature
    }
    
    response = self.client.post(url, json=payload, headers=self._headers())
    response.raise_for_status()
    
    return response.json()["choices"][0]["message"]["content"]
```

#### 6.3.4 _headers() — 인증 헤더

```python
def _headers(self) -> dict[str, str]:
    """API 요청 헤더를 생성합니다."""
    headers = {"Content-Type": "application/json"}
    if self.api_key:
        headers["Authorization"] = f"Bearer {self.api_key}"
    return headers
```

#### 6.3.5 health_check() — 연결 확인

```python
def health_check(self) -> bool:
    """API 서버 연결을 확인합니다."""
    try:
        response = self.client.get(
            f"{self.api_base}/v1/models",
            headers=self._headers()
        )
        return response.status_code == 200
    except Exception:
        return False
```

### 6.4 __init__.py (50줄)

#### 6.4.1 역할

Teacher 백엔드 팩토리 함수를 제공합니다.

#### 6.4.2 create_teacher() — 팩토리 함수

```python
def create_teacher(config: TeacherConfig) -> BaseTeacher:
    """설정에 따라 적절한 Teacher 인스턴스를 생성합니다."""
    backend = config.backend.lower()
    
    if backend == "ollama":
        return OllamaTeacher(config)
    elif backend == "openai":
        return OpenAICompatTeacher(config)
    else:
        raise ValueError(
            f"지원하지 않는 백엔드: {backend}. "
            f"'ollama' 또는 'openai'를 사용하세요."
        )
```

**사용 예시:**
```python
from slm_factory.teacher import create_teacher

teacher = create_teacher(config.teacher)
response = teacher.generate("질문을 생성하세요.")
```

### 6.5 qa_generator.py (298줄)

#### 6.5.1 역할

Teacher LLM을 사용하여 문서에서 QA 쌍을 생성하는 핵심 오케스트레이터입니다.

#### 6.5.2 QAGenerator 클래스

```python
class QAGenerator:
    def __init__(self, config: SLMConfig):
        self.teacher = create_teacher(config.teacher)
        self.questions_config = config.questions
        self.max_context = config.teacher.max_context_chars
        self.logger = get_logger("qa_generator")
```

#### 6.5.3 build_prompt() — 프롬프트 구성

```python
def build_prompt(
    self,
    doc_title: str,
    content: str,
    question: str,
    tables: list[str],
    system_prompt: str
) -> str:
    """5개 섹션으로 구성된 프롬프트를 생성합니다."""
    
    # 컨텍스트 길이 제한
    if len(content) > self.max_context:
        content = content[:self.max_context] + "\n...(생략)"
    
    sections = [
        f"# System Instructions\n{system_prompt}",
        f"# Document\nTitle: {doc_title}\n\n{content}",
    ]
    
    # 표 추가 (있는 경우)
    if tables:
        tables_text = "\n\n".join(tables)
        sections.append(f"# Tables\n{tables_text}")
    
    sections.extend([
        f"# Question\n{question}",
        "# Output\nJSON 형식으로 응답하세요:\n"
        '{"instruction": "질문", "output": "답변"}'
    ])
    
    return "\n\n".join(sections)
```

**프롬프트 구조:**
1. **System Instructions**: 역할 정의 및 지시사항
2. **Document**: 문서 제목 + 본문 (최대 12,000자)
3. **Tables**: 추출된 표 (있는 경우)
4. **Question**: 생성할 질문 템플릿
5. **Output**: JSON 형식 지시

#### 6.5.4 parse_response() — 응답 파싱

```python
def parse_response(self, text: str) -> dict | None:
    """LLM 응답을 파싱하여 QA 쌍을 추출합니다."""
    
    # 코드 펜스 제거
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*$", "", text)
    
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    
    # 4가지 응답 형식 처리
    if isinstance(data, dict):
        # 형식 1: 직접 객체
        if "instruction" in data or "question" in data:
            return self._normalize_keys(data)
        
        # 형식 3: 래핑된 배열
        if "data" in data:
            data = data["data"]
        elif "items" in data:
            data = data["items"]
    
    # 형식 2: 배열
    if isinstance(data, list) and len(data) > 0:
        return self._normalize_keys(data[0])
    
    return None

def _normalize_keys(self, obj: dict) -> dict:
    """키를 표준 형식으로 정규화합니다."""
    return {
        "instruction": obj.get("instruction") or obj.get("question", ""),
        "output": obj.get("output") or obj.get("answer", "")
    }
```

**지원 형식:**
```json
// 형식 1: 직접 객체
{"instruction": "...", "output": "..."}

// 형식 2: 배열
[{"instruction": "...", "output": "..."}]

// 형식 3: 래핑된 배열
{"data": [{"instruction": "...", "output": "..."}]}

// 키 변형: question→instruction, answer→output
{"question": "...", "answer": "..."}
```

#### 6.5.5 generate_for_document() — 문서별 QA 생성

```python
def generate_for_document(
    self,
    doc: ParsedDocument,
    questions: list[str],
    category: str
) -> list[QAPair]:
    """하나의 문서에서 여러 QA 쌍을 생성합니다."""
    
    pairs = []
    
    for question in questions:
        try:
            # 프롬프트 구성
            prompt = self.build_prompt(
                doc.title,
                doc.content,
                question,
                doc.tables,
                self.questions_config.system_prompt
            )
            
            # LLM 호출
            kwargs = {}
            if isinstance(self.teacher, OllamaTeacher):
                kwargs["format"] = "json"
            
            response = self.teacher.generate(prompt, **kwargs)
            
            # 응답 파싱
            parsed = self.parse_response(response)
            if parsed:
                pairs.append(QAPair(
                    question=parsed["instruction"],
                    answer=parsed["output"],
                    instruction=parsed["instruction"],
                    source_doc=doc.doc_id,
                    category=category
                ))
            
        except Exception as e:
            self.logger.error(f"QA 생성 실패 ({doc.doc_id}, {question}): {e}")
    
    return pairs
```

**특징:**
- 질문별 1:1 처리 (최대 품질 보장)
- Ollama 백엔드: `format="json"` 자동 추가
- 개별 실패 로깅 후 건너뜀 (전체 프로세스 중단 없음)

#### 6.5.6 generate_all() — 다중 문서 일괄 처리

```python
def generate_all(
    self,
    docs: list[ParsedDocument],
    questions: dict[str, list[str]]
) -> list[QAPair]:
    """모든 문서에서 QA 쌍을 생성합니다."""
    
    all_pairs = []
    
    for doc in docs:
        for category, question_list in questions.items():
            pairs = self.generate_for_document(doc, question_list, category)
            all_pairs.extend(pairs)
            
            self.logger.info(
                f"{doc.doc_id} ({category}): {len(pairs)}개 QA 생성"
            )
    
    return all_pairs
```

#### 6.5.7 save_alpaca() — Alpaca 형식 저장

```python
def save_alpaca(self, pairs: list[QAPair], output_path: Path) -> Path:
    """QA 쌍을 Alpaca JSON 형식으로 저장합니다."""
    
    alpaca_data = []
    for pair in pairs:
        alpaca_data.append({
            "instruction": pair.instruction,
            "input": "",
            "output": pair.answer
        })
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(alpaca_data, f, ensure_ascii=False, indent=2)
    
    return output_path
```

**Alpaca 형식:**
```json
[
  {
    "instruction": "질문 내용",
    "input": "",
    "output": "답변 내용"
  }
]
```

---

## 7. validator/ — QA 검증 모듈

### 7.1 rules.py (123줄)

#### 7.1.1 역할

규칙 기반 QA 쌍 필터링을 수행합니다.

#### 7.1.2 QAPair (dataclass)

```python
@dataclass
class QAPair:
    question: str
    answer: str
    instruction: str = ""
    source_doc: str = ""
    category: str = ""
```

#### 7.1.3 ValidationResult (dataclass)

```python
@dataclass
class ValidationResult:
    passed: bool
    reasons: list[str] = field(default_factory=list)
```

#### 7.1.4 RuleValidator 클래스

```python
class RuleValidator:
    def __init__(self, config: ValidationConfig):
        self.config = config
        self._seen_pairs: set[str] = set()
        self._compiled_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in config.reject_patterns
        ]
```

#### 7.1.5 validate_one() — 단일 QA 검증

```python
def validate_one(self, pair: QAPair) -> ValidationResult:
    """4단계 순차 검증을 수행합니다."""
    reasons = []
    
    # 1. 빈값 검사
    if not pair.question.strip() or not pair.answer.strip():
        reasons.append("빈 질문 또는 답변")
    
    # 2. 길이 검사
    answer_len = len(pair.answer)
    if answer_len < self.config.min_answer_length:
        reasons.append(f"답변이 너무 짧음 ({answer_len}자)")
    if answer_len > self.config.max_answer_length:
        reasons.append(f"답변이 너무 김 ({answer_len}자)")
    
    # 3. 패턴 매칭
    for pattern in self._compiled_patterns:
        if pattern.search(pair.answer):
            reasons.append(f"거부 패턴 매칭: {pattern.pattern}")
    
    # 4. 중복 검사
    if self.config.deduplicate:
        key = (pair.question + pair.answer).lower()
        if key in self._seen_pairs:
            reasons.append("중복된 QA 쌍")
        else:
            self._seen_pairs.add(key)
    
    return ValidationResult(
        passed=len(reasons) == 0,
        reasons=reasons
    )
```

**검증 순서:**
1. 빈값 검사
2. 길이 검사 (20~2000자)
3. 거부 패턴 매칭 (정규식)
4. 중복 검사 (question+answer 소문자 조합)

#### 7.1.6 validate_batch() — 일괄 검증

```python
def validate_batch(
    self, 
    pairs: list[QAPair]
) -> tuple[list[QAPair], list[tuple[QAPair, list[str]]]]:
    """QA 쌍 목록을 검증하고 통과/거부로 분류합니다."""
    
    accepted = []
    rejected = []
    
    for pair in pairs:
        result = self.validate_one(pair)
        if result.passed:
            accepted.append(pair)
        else:
            rejected.append((pair, result.reasons))
    
    # 거부 사유 통계
    reason_counts = {}
    for _, reasons in rejected:
        for reason in reasons:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    
    self.logger.info(f"검증 완료: {len(accepted)} 통과, {len(rejected)} 거부")
    for reason, count in reason_counts.items():
        self.logger.info(f"  - {reason}: {count}개")
    
    return accepted, rejected
```

**반환값:**
- `accepted`: 검증 통과한 QA 쌍 목록
- `rejected`: `(QAPair, 거부사유 목록)` 튜플 목록

#### 7.1.7 reset_dedup() — 중복 캐시 초기화

```python
def reset_dedup(self) -> None:
    """중복 검사 캐시를 초기화합니다."""
    self._seen_pairs.clear()
```

### 7.2 similarity.py (143줄)

#### 7.2.1 역할

임베딩 코사인 유사도를 사용하여 답변이 원본 문서에 근거하는지 검증합니다.

#### 7.2.2 GroundednessChecker 클래스

```python
class GroundednessChecker:
    def __init__(self, config: GroundednessConfig):
        self._check_sentence_transformers()
        
        self.model_name = config.model
        self.threshold = config.threshold
        self.model = SentenceTransformer(self.model_name)
        self.logger = get_logger("groundedness")
```

#### 7.2.3 _check_sentence_transformers() — 의존성 확인

```python
def _check_sentence_transformers(self) -> None:
    """sentence-transformers 설치 여부를 확인합니다."""
    try:
        import sentence_transformers
    except ImportError:
        raise ImportError(
            "sentence-transformers가 설치되지 않았습니다.\n"
            "pip install slm-factory[validation] 명령으로 설치하세요."
        )
```

#### 7.2.4 _chunk_text() — 텍스트 청킹

```python
def _chunk_text(
    self, 
    text: str, 
    chunk_size: int = 512, 
    overlap: int = 64
) -> list[str]:
    """텍스트를 오버랩 청크로 분할합니다."""
    
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    
    return chunks
```

**파라미터:**
- `chunk_size: 512`: 청크당 단어 수
- `overlap: 64`: 인접 청크 간 중복 단어 수

#### 7.2.5 score() — 유사도 계산

```python
def score(self, answer: str, source_text: str) -> float:
    """답변과 원본 문서의 코사인 유사도를 계산합니다."""
    
    # 원본 문서 청킹
    chunks = self._chunk_text(source_text)
    
    # 임베딩 생성
    answer_emb = self.model.encode([answer])[0]
    chunk_embs = self.model.encode(chunks)
    
    # 코사인 유사도 계산
    similarities = cosine_similarity([answer_emb], chunk_embs)[0]
    
    # 최대 유사도 반환
    return float(max(similarities))
```

**동작:**
1. 원본 문서를 512단어 청크로 분할
2. 답변과 각 청크의 임베딩 생성
3. 코사인 유사도 계산
4. 최대 유사도 반환 (가장 관련성 높은 청크 기준)

#### 7.2.6 check() — 근거성 검사

```python
def check(self, pair: QAPair, source_text: str) -> tuple[bool, float]:
    """답변이 원본 문서에 근거하는지 확인합니다."""
    
    similarity = self.score(pair.answer, source_text)
    passed = similarity >= self.threshold
    
    return passed, similarity
```

**반환값:**
- `passed`: threshold 이상이면 True
- `similarity`: 실제 유사도 점수 (0.0~1.0)

#### 7.2.7 check_batch() — 일괄 검증

```python
def check_batch(
    self,
    pairs: list[QAPair],
    source_texts: dict[str, str]
) -> tuple[list[QAPair], list[QAPair]]:
    """QA 쌍 목록의 근거성을 일괄 검증합니다."""
    
    grounded = []
    ungrounded = []
    
    for pair in pairs:
        # source_doc이 없으면 자동 통과
        if pair.source_doc not in source_texts:
            grounded.append(pair)
            continue
        
        source_text = source_texts[pair.source_doc]
        passed, score = self.check(pair, source_text)
        
        if passed:
            grounded.append(pair)
        else:
            ungrounded.append(pair)
            self.logger.debug(
                f"근거성 부족: {pair.question[:50]}... "
                f"(유사도: {score:.3f})"
            )
    
    self.logger.info(
        f"근거성 검증: {len(grounded)} 통과, {len(ungrounded)} 거부"
    )
    
    return grounded, ungrounded
```

**특징:**
- `source_doc`이 `source_texts`에 없으면 자동 통과 (안전 장치)
- 거부된 QA는 디버그 로그에 유사도 점수 기록

---

## 8. scorer.py — QA 품질 점수 평가 모듈 (125줄)

### 8.1 역할

Teacher LLM을 사용하여 QA 쌍의 품질을 1~5점으로 평가하고 threshold 기반으로 필터링합니다. 생성된 QA 쌍 중 품질이 낮은 데이터를 자동으로 제거하여 학습 데이터의 전반적인 품질을 향상시킵니다.

### 8.2 QualityScorer 클래스

```python
class QualityScorer:
    """교사 LLM을 사용하여 QA 쌍의 품질을 1~5점으로 평가합니다."""
    
    def __init__(self, teacher: BaseTeacher, config: ScoringConfig, teacher_config: TeacherConfig):
        self.teacher = teacher
        self.config = config
        self.teacher_config = teacher_config
```

**초기화 파라미터:**
- `teacher`: Teacher LLM 백엔드 인스턴스
- `config`: 점수 평가 설정 (threshold, max_concurrency)
- `teacher_config`: Teacher LLM 설정 (backend, model 등)

### 8.3 _build_scoring_prompt(pair) — 점수 평가 프롬프트 구성

```python
def _build_scoring_prompt(self, pair: QAPair) -> str:
    """점수 평가를 위한 프롬프트를 구성합니다."""
```

**프롬프트 구조:**
- 1~5점 평가 기준 명시 (1점: 완전히 잘못됨, 5점: 정확하고 완전함)
- 평가 대상 QA 쌍 제시
- 참고 예시 제공 (5점, 2점, 1점 사례)
- JSON 형식 응답 요구: `{"score": <1-5>, "reason": "<평가 근거>"}`

### 8.4 _parse_score(text) → tuple[int, str] | None

```python
def _parse_score(self, text: str) -> tuple[int, str] | None:
    """LLM 응답에서 점수와 이유를 추출합니다."""
```

**파싱 전략:**
1. JSON 파싱 시도 → `score`와 `reason` 필드 추출
2. JSON 파싱 실패 시 정규식으로 1~5 숫자 추출
3. 완전 실패 시 `None` 반환 (경고 로그 출력)

### 8.5 async score_one(pair) → tuple[QAPair, int, str]

```python
async def score_one(self, pair: QAPair) -> tuple[QAPair, int, str]:
    """단일 QA 쌍을 점수 평가합니다."""
```

**동작:**
1. 점수 평가 프롬프트 구성
2. Ollama 백엔드인 경우 `format="json"` 옵션 추가
3. `teacher.agenerate()` 비동기 호출
4. 응답 파싱 → 실패 시 기본값 3점 적용
5. `(QAPair, 점수, 이유)` 튜플 반환

### 8.6 async score_all(pairs) → tuple[list[QAPair], list[tuple[QAPair, int, str]]]

```python
async def score_all(
    self,
    pairs: list[QAPair],
) -> tuple[list[QAPair], list[tuple[QAPair, int, str]]]:
    """전체 QA 쌍을 점수 평가하고 threshold 기준으로 필터링합니다."""
```

**동작:**
1. `asyncio.Semaphore`로 동시성 제한 (`max_concurrency`)
2. 모든 QA 쌍에 대해 `score_one()` 병렬 실행
3. `asyncio.gather()`로 결과 수집 (예외는 로그 출력)
4. `threshold` 이상 → `accepted`, 미만 → `filtered`
5. 통과/제거 통계 로그 출력

**반환값:**
- `accepted`: threshold 이상의 QA 쌍 리스트
- `filtered`: threshold 미만의 QA 쌍 + 점수 + 이유 리스트

**사용 예시:**
```python
scorer = QualityScorer(teacher, config.scoring, config.teacher)
accepted, filtered = await scorer.score_all(qa_pairs)

# 로그 출력 예시:
# 품질 점수 평가 완료: 85/100 통과 (threshold=3.0), 15 제거
```

---

## 9. augmenter.py — QA 데이터 증강 모듈 (119줄)

### 9.1 역할

Teacher LLM을 사용하여 질문을 패러프레이즈(paraphrase)하여 학습 데이터를 증강합니다. 원본 질문의 의미를 유지하면서 다양한 표현으로 변형하여 모델의 일반화 성능을 향상시킵니다.

### 9.2 DataAugmenter 클래스

```python
class DataAugmenter:
    """교사 LLM을 사용하여 질문을 패러프레이즈하여 데이터를 증강합니다."""
    
    def __init__(self, teacher: BaseTeacher, config: AugmentConfig, teacher_config: TeacherConfig):
        self.teacher = teacher
        self.config = config
        self.teacher_config = teacher_config
```

**초기화 파라미터:**
- `teacher`: Teacher LLM 백엔드 인스턴스
- `config`: 증강 설정 (num_variants, max_concurrency)
- `teacher_config`: Teacher LLM 설정

### 9.3 _build_paraphrase_prompt(question, num_variants) — 패러프레이즈 프롬프트 구성

```python
def _build_paraphrase_prompt(self, question: str, num_variants: int) -> str:
    """패러프레이즈 프롬프트를 구성합니다."""
```

**프롬프트 구조:**
- 원본 질문 제시
- `num_variants`개의 변형 질문 요청
- 규칙 명시:
  - 원래 질문과 동일한 의미 유지
  - 서로 다른 문장 구조나 어휘 사용
  - 자연스러운 한국어/영어 (원본 언어 따름)
  - 질문 형식 유지
- JSON 형식 응답 요구: `{"questions": ["변형1", "변형2", ...]}`

### 9.4 _parse_paraphrases(text) → list[str]

```python
def _parse_paraphrases(self, text: str) -> list[str]:
    """LLM 응답에서 패러프레이즈된 질문을 추출합니다."""
```

**파싱 전략:**
1. JSON 파싱 시도 → `questions` 배열 추출
2. 배열 직접 반환도 지원 (dict 없이 list만 있는 경우)
3. 파싱 실패 시 빈 리스트 반환 (경고 로그 출력)

### 9.5 async paraphrase_one(pair) → list[QAPair]

```python
async def paraphrase_one(self, pair: QAPair) -> list[QAPair]:
    """단일 QA 쌍의 질문을 패러프레이즈하여 증강된 QA 쌍을 생성합니다."""
```

**동작:**
1. 패러프레이즈 프롬프트 구성
2. Ollama 백엔드인 경우 `format="json"` 옵션 추가
3. `teacher.agenerate()` 비동기 호출
4. 응답 파싱 → 변형 질문 리스트 추출
5. 각 변형 질문으로 새 QAPair 생성:
   - `question`: 변형된 질문
   - `answer`: 원본과 동일
   - `source_doc`, `category`: 원본과 동일
   - `is_augmented=True` 플래그 설정
6. 증강된 QAPair 리스트 반환

### 9.6 async augment_all(pairs) → list[QAPair]

```python
async def augment_all(self, pairs: list[QAPair]) -> list[QAPair]:
    """전체 QA 쌍을 증강합니다. 원본 + 증강 쌍을 반환합니다."""
```

**동작:**
1. `is_augmented=False`인 원본 QA만 증강 대상으로 선택
2. `asyncio.Semaphore`로 동시성 제한
3. 모든 원본 QA에 대해 `paraphrase_one()` 병렬 실행
4. `asyncio.gather()`로 결과 수집 (예외는 로그 출력)
5. 원본 + 증강 QA 합쳐서 반환
6. 증강 통계 로그 출력

**반환값:**
- 원본 QA 쌍 + 증강된 QA 쌍 전체 리스트

**사용 예시:**
```python
augmenter = DataAugmenter(teacher, config.augment, config.teacher)
augmented_pairs = await augmenter.augment_all(qa_pairs)

# 로그 출력 예시:
# 데이터 증강 완료: 원본 100 + 증강 200 = 총 300개
```

---

## 10. analyzer.py — 학습 데이터 분석 모듈 (173줄)

### 10.1 역할

LLM 의존성 없이 순수 통계 분석으로 QA 데이터의 품질을 수치로 보고합니다. 데이터 분포, 길이 통계, 불균형 등을 자동으로 분석하여 학습 전에 데이터 품질을 검증할 수 있습니다.

### 10.2 AnalysisReport (dataclass)

```python
@dataclass
class AnalysisReport:
    """QA 데이터 분석 결과를 담는 보고서입니다."""
    total_pairs: int = 0
    original_pairs: int = 0
    augmented_pairs: int = 0
    category_distribution: dict[str, int] = field(default_factory=dict)
    source_doc_distribution: dict[str, int] = field(default_factory=dict)
    answer_length_stats: dict[str, float] = field(default_factory=dict)
    question_length_stats: dict[str, float] = field(default_factory=dict)
    quality_score_stats: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
```

**필드 설명:**
- `total_pairs`: 전체 QA 쌍 수
- `original_pairs`: 원본 QA 쌍 수 (`is_augmented=False`)
- `augmented_pairs`: 증강된 QA 쌍 수 (`is_augmented=True`)
- `category_distribution`: 카테고리별 QA 수 (예: `{"이해": 50, "분석": 30}`)
- `source_doc_distribution`: 문서별 QA 수
- `answer_length_stats`: 답변 길이 통계 (min, max, mean, median, stdev)
- `question_length_stats`: 질문 길이 통계
- `quality_score_stats`: 품질 점수 통계 (현재 미사용)
- `warnings`: 자동 생성된 경고 메시지 리스트

### 10.3 DataAnalyzer 클래스

```python
class DataAnalyzer:
    """QA 쌍의 통계를 분석하여 데이터 품질을 수치로 보고합니다.
    
    LLM 의존성이 없으며 순수 계산만 수행합니다.
    """
```

### 10.4 analyze(pairs) → AnalysisReport

```python
def analyze(self, pairs: list[QAPair]) -> AnalysisReport:
    """QA 쌍 리스트를 분석하여 AnalysisReport를 생성합니다."""
```

**동작:**
1. 전체/원본/증강 QA 수 카운트
2. 카테고리별 분포 계산 (`Counter` 사용)
3. 문서별 분포 계산
4. 답변 길이 통계 계산 (`_compute_stats()` 호출)
5. 질문 길이 통계 계산
6. 자동 경고 생성 (`_generate_warnings()` 호출)
7. `AnalysisReport` 객체 반환

### 10.5 _compute_stats(values) → dict[str, float]

```python
def _compute_stats(self, values: list[int | float]) -> dict[str, float]:
    """수치 리스트에서 기초 통계를 계산합니다."""
```

**계산 항목:**
- `min`: 최솟값
- `max`: 최댓값
- `mean`: 평균 (소수점 1자리 반올림)
- `median`: 중앙값 (소수점 1자리 반올림)
- `stdev`: 표준편차 (소수점 1자리 반올림, 데이터 1개면 0.0)

### 10.6 _generate_warnings(report, pairs)

```python
def _generate_warnings(self, report: AnalysisReport, pairs: list[QAPair]) -> None:
    """데이터 불균형이나 이상치를 경고합니다."""
```

**경고 조건:**
1. **문서별 QA 수 불균형**: `max > 5 * min`인 경우
   - "문서별 QA 쌍 수 불균형이 심합니다 (최소: X, 최대: Y). 특정 문서에 데이터가 편중되어 학습 편향이 발생할 수 있습니다."
2. **카테고리 1개만 존재**: `len(category_distribution) == 1`
   - "카테고리가 1개뿐입니다. 다양한 카테고리를 추가하면 모델 성능이 향상됩니다."
3. **답변 길이 편차 과다**: `stdev > mean * 1.5`
   - "답변 길이의 편차가 매우 큽니다. 일부 답변이 비정상적으로 길거나 짧을 수 있습니다."
4. **데이터 부족**: `total_pairs < 50`
   - "학습 데이터가 X개로 적습니다. 최소 100개 이상을 권장합니다."

### 10.7 print_summary(report) — Rich 콘솔 출력

```python
def print_summary(self, report: AnalysisReport) -> None:
    """Rich 콘솔에 분석 요약을 출력합니다."""
```

**출력 구조:**
1. **Panel**: "학습 데이터 분석 보고서" 제목
2. **기본 통계 Table**: 전체/원본/증강 QA 수
3. **카테고리 분포 Table**: 카테고리별 개수 + 비율
4. **길이 통계 Table**: 답변/질문 길이의 min, max, mean, median, stdev
5. **경고 메시지**: 노란색으로 `⚠` 아이콘과 함께 출력

### 10.8 save_report(report, path) — JSON 파일 저장

```python
def save_report(self, report: AnalysisReport, path: Path) -> None:
    """분석 보고서를 JSON 파일로 저장합니다."""
```

**동작:**
1. 부모 디렉토리 생성 (`parents=True, exist_ok=True`)
2. `dataclasses.asdict()`로 dict 변환
3. JSON 파일로 저장 (`ensure_ascii=False, indent=2`)
4. 저장 완료 로그 출력

**사용 예시:**
```python
analyzer = DataAnalyzer()
report = analyzer.analyze(qa_pairs)
analyzer.print_summary(report)
analyzer.save_report(report, Path("output/analysis_report.json"))
```

---

## 11. models.py — 공유 데이터 모델 (~37줄)

### 11.1 역할

QAPair와 ParsedDocument 등 파이프라인 전체에서 사용하는 공유 데이터 모델을 정의합니다. 이전에는 각각 `validator/rules.py`와 `parsers/base.py`에 분산되어 있었으나, 리팩토링으로 중앙화되었습니다.

### 11.2 ParsedDocument (dataclass)

파싱된 문서의 표준 데이터 구조입니다. 이전에는 `parsers/base.py`에 정의되었습니다.

```python
@dataclass
class ParsedDocument:
    doc_id: str              # 파일명 기반 고유 식별자
    title: str               # 문서 제목
    content: str             # 마크다운 형식 본문
    tables: list[str] = field(default_factory=list)  # 마크다운 표 목록
    metadata: dict = field(default_factory=dict)     # 추가 메타데이터
```

**필드 설명:**
- `doc_id`: 파일명에서 확장자를 제거한 값 (예: `"report_20240115"`)
- `title`: 문서 제목 (메타데이터 또는 파일명에서 추출)
- `content`: 전체 텍스트 (마크다운 형식, 표는 별도 저장)
- `tables`: 추출된 표 목록 (각 표는 마크다운 문자열)
- `metadata`: `{"author": "...", "date": "2024-01-15", "page_count": 10}` 등

### 11.3 QAPair (dataclass)

QA 쌍의 표준 데이터 구조입니다. 이전에는 `validator/rules.py`에 정의되었습니다.

```python
@dataclass
class QAPair:
    question: str
    answer: str
    instruction: str = ""
    source_doc: str = ""
    category: str = ""
    is_augmented: bool = False
```

**필드 설명:**
- `question`: 질문 텍스트
- `answer`: 답변 텍스트
- `instruction`: 지시사항 (Alpaca 형식에서 사용)
- `source_doc`: 원본 문서 ID (근거성 검증에 사용)
- `category`: 질문 카테고리 (예: "이해", "분석", "종합")
- `is_augmented`: 증강된 QA 쌍인지 여부 (DataAugmenter가 생성한 패러프레이즈 쌍은 `True`)

### 11.4 사용 패턴

```python
from slm_factory.models import ParsedDocument, QAPair

# ParsedDocument 생성
doc = ParsedDocument(
    doc_id="report_001",
    title="분기별 보고서",
    content="...",
    tables=[],
    metadata={"date": "2024-01-15"}
)

# QAPair 생성
pair = QAPair(
    question="주요 성과는?",
    answer="...",
    instruction="주요 성과는?",
    source_doc="report_001",
    category="이해"
)
```

---

## 12. converter.py — 형식 변환 모듈 (~265줄)

### 12.1 역할

Alpaca 형식의 QA 쌍을 Student 모델에 맞는 채팅 템플릿으로 변환합니다.

### 12.2 ChatFormatter 클래스

```python
class ChatFormatter:
    def __init__(self, config: SLMConfig):
        self.model_name = config.student.model
        self.max_seq_length = config.student.max_seq_length
        self.system_prompt = config.export.ollama.system_prompt
        self._tokenizer = None
        self.logger = get_logger("formatter")
```

### 12.3 tokenizer (property) — 지연 로딩

```python
@property
def tokenizer(self):
     """토크나이저를 지연 로딩합니다."""
     if self._tokenizer is None:
         self._tokenizer = AutoTokenizer.from_pretrained(
             self.model_name,
             trust_remote_code=True
         )
     return self._tokenizer
```

**이유:**
- 토크나이저 로드는 시간이 걸리므로 실제 사용 시점에 로드
- 여러 번 호출해도 한 번만 로드됨

### 12.4 build_messages() — 메시지 구성

```python
def build_messages(self, pair: QAPair) -> list[dict]:
    """QA 쌍을 채팅 메시지 형식으로 변환합니다."""
    
    messages = [
        {"role": "system", "content": self.system_prompt},
        {"role": "user", "content": pair.question},
        {"role": "assistant", "content": pair.answer}
    ]
    
    return messages
```

**형식:**
```python
[
    {"role": "system", "content": "당신은 도움이 되는 AI 어시스턴트입니다."},
    {"role": "user", "content": "질문 내용"},
    {"role": "assistant", "content": "답변 내용"}
]
```

### 12.5 format_one() — 단일 QA 변환

```python
def format_one(self, pair: QAPair) -> str | None:
    """QA 쌍을 채팅 템플릿 문자열로 변환합니다."""
    
    messages = self.build_messages(pair)
    
    try:
        # 채팅 템플릿 적용
        formatted = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        return formatted
        
    except Exception as e:
        # system 역할 미지원 모델 처리 (Gemma 등)
        self.logger.warning(
            f"채팅 템플릿 적용 실패, system 제거 후 재시도: {e}"
        )
        
        messages_no_system = [
            {"role": "user", "content": pair.question},
            {"role": "assistant", "content": pair.answer}
        ]
        
        try:
            formatted = self.tokenizer.apply_chat_template(
                messages_no_system,
                tokenize=False,
                add_generation_prompt=False
            )
            return formatted
        except Exception as e2:
            self.logger.error(f"채팅 템플릿 적용 최종 실패: {e2}")
            return None
```

**폴백 전략:**
1. system + user + assistant 시도
2. 실패 시 system 제거 후 재시도 (Gemma 등)
3. 최종 실패 시 None 반환

### 12.6 format_batch() — 일괄 변환

```python
def format_batch(self, pairs: list[QAPair]) -> list[dict[str, str]]:
    """QA 쌍 목록을 일괄 변환합니다."""
    
    formatted_data = []
    skipped = 0
    
    for pair in pairs:
        formatted = self.format_one(pair)
        if formatted is None:
            skipped += 1
            continue
        
        # 토큰 수 확인
        token_count = len(self.tokenizer.encode(formatted))
        if token_count > self.max_seq_length:
            self.logger.warning(
                f"시퀀스 길이 초과 ({token_count} > {self.max_seq_length}), "
                f"건너뜀: {pair.question[:50]}..."
            )
            skipped += 1
            continue
        
        formatted_data.append({"text": formatted})
    
    self.logger.info(
        f"변환 완료: {len(formatted_data)}개 성공, {skipped}개 건너뜀"
    )
    
    return formatted_data
```

**검증:**
- 채팅 템플릿 적용 실패 시 건너뜀
- `max_seq_length` 초과 시 건너뜀 (기본 4096 토큰)

### 12.7 save_training_data() — JSONL 저장

```python
def save_training_data(
    self, 
    pairs: list[QAPair], 
    output_path: Path
) -> Path:
    """학습 데이터를 JSONL 형식으로 저장합니다."""
    
    formatted_data = self.format_batch(pairs)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for item in formatted_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    self.logger.info(f"학습 데이터 저장 완료: {output_path}")
    return output_path
```

**JSONL 형식:**
```jsonl
{"text": "<|im_start|>system\n당신은...<|im_end|>\n<|im_start|>user\n질문<|im_end|>\n<|im_start|>assistant\n답변<|im_end|>"}
{"text": "..."}
```

### 12.8 format_from_alpaca_file() — 독립 변환

```python
def format_from_alpaca_file(
    self, 
    input_path: Path, 
    output_path: Path
) -> Path:
    """Alpaca JSON 파일을 직접 JSONL로 변환합니다."""
    
    with open(input_path, encoding="utf-8") as f:
        alpaca_data = json.load(f)
    
    pairs = [
        QAPair(
            question=item["instruction"],
            answer=item["output"],
            instruction=item["instruction"]
        )
        for item in alpaca_data
    ]
    
    return self.save_training_data(pairs, output_path)
```

**사용 사례:**
- 외부에서 생성한 Alpaca 데이터를 변환할 때

---

## 13. trainer/ — 학습 모듈

### 13.1 lora_trainer.py (~315줄)

#### 13.1.1 역할

JSONL 학습 데이터를 로드하고 HuggingFace TRL의 SFTTrainer를 사용하여 LoRA 파인튜닝을 수행합니다. 이전의 `data_loader.py`의 기능이 통합되었습니다.

#### 13.1.2 DataLoader 클래스 (통합)

```python
class DataLoader:
     def __init__(self, train_split: float = 0.9):
         self.train_split = train_split
         self.logger = get_logger("data_loader")
     
     def load_jsonl(self, path: Path) -> Dataset:
         """JSONL 파일을 Dataset으로 로드합니다."""
         dataset = load_dataset("json", data_files=str(path), split="train")
         self.logger.info(f"데이터 로드 완료: {len(dataset)}개 샘플")
         return dataset
     
     def split(self, dataset: Dataset) -> DatasetDict:
         """Dataset을 train/eval로 분할합니다."""
         split_dataset = dataset.train_test_split(
             test_size=1.0 - self.train_split,
             seed=42
         )
         return DatasetDict({
             "train": split_dataset["train"],
             "eval": split_dataset["test"]
         })
     
     def load_and_split(self, path: Path) -> DatasetDict:
         """JSONL 로드 + 분할을 한 번에 수행합니다."""
         dataset = self.load_jsonl(path)
         return self.split(dataset)
```

**기본 분할:**
- train: 90%
- eval: 10%
- seed: 42 (재현성)

#### 13.1.3 LoRATrainer 클래스

```python
class LoRATrainer:
    def __init__(self, config: SLMConfig):
        self.config = config
        self.student_config = config.student
        self.training_config = config.training
        self.lora_config = config.training.lora
        self.output_dir = Path(config.paths.output) / "checkpoints"
        self.logger = get_logger("lora_trainer")
```

#### 13.1.4 _load_model() — 모델 로드

```python
def _load_model(self) -> tuple:
    """Student 모델과 토크나이저를 로드합니다."""
    
    self.logger.info(f"모델 로드 중: {self.student_config.model}")
    
    # 양자화 설정 (선택적)
    quantization_config = None
    if self.training_config.quantization.enabled:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )
    
    # 모델 로드
    model = AutoModelForCausalLM.from_pretrained(
        self.student_config.model,
        device_map="auto",
        torch_dtype=torch.bfloat16 if self.training_config.bf16 else torch.float32,
        quantization_config=quantization_config,
        trust_remote_code=True
    )
    
    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(
        self.student_config.model,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 파라미터 수 로깅
    total_params = sum(p.numel() for p in model.parameters())
    self.logger.info(f"모델 파라미터: {total_params:,}개")
    
    return model, tokenizer
```

**특징:**
- `device_map="auto"`: 자동 GPU 할당
- `bf16`: bfloat16 혼합 정밀도 학습
- 4-bit NF4 양자화 지원 (메모리 절약)
- pad_token 자동 설정

#### 13.1.5 _create_lora_config() — LoRA 설정

```python
def _create_lora_config(self) -> peft.LoraConfig:
    """PEFT LoraConfig를 생성합니다."""
    
    # target_modules 처리
    target_modules = self.lora_config.target_modules
    if target_modules == "auto":
        target_modules = None  # PEFT 자동 감지
    
    return peft.LoraConfig(
        r=self.lora_config.r,
        lora_alpha=self.lora_config.alpha,
        lora_dropout=self.lora_config.dropout,
        target_modules=target_modules,
        use_rslora=self.lora_config.use_rslora,
        task_type=peft.TaskType.CAUSAL_LM,
        bias="none"
    )
```

**파라미터:**
- `r: 16`: LoRA rank (저랭크 행렬 차원)
- `alpha: 32`: 스케일링 파라미터
- `dropout: 0.05`: LoRA 레이어 드롭아웃
- `target_modules: "auto"`: 자동 레이어 선택
- `use_rslora: False`: Rank-Stabilized LoRA

#### 13.1.6 _create_training_args() — 학습 인자

```python
def _create_training_args(self) -> TrainingArguments:
    """HuggingFace TrainingArguments를 생성합니다."""
    
    return TrainingArguments(
        output_dir=str(self.output_dir),
        num_train_epochs=self.training_config.num_epochs,
        per_device_train_batch_size=self.training_config.batch_size,
        gradient_accumulation_steps=self.training_config.gradient_accumulation_steps,
        learning_rate=self.training_config.learning_rate,
        lr_scheduler_type=self.training_config.lr_scheduler,
        warmup_ratio=self.training_config.warmup_ratio,
        bf16=self.training_config.bf16,
        optim=self.training_config.optimizer,
        save_strategy=self.training_config.save_strategy,
        evaluation_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=10,
        report_to="none",
        seed=42
    )
```

**주요 설정:**
- `gradient_accumulation_steps: 4`: 실질적 배치 크기 = 4 × 4 = 16
- `lr_scheduler_type: "cosine"`: 코사인 학습률 스케줄러
- `warmup_ratio: 0.1`: 전체 스텝의 10%를 워밍업
- `load_best_model_at_end: True`: 최고 성능 체크포인트 자동 로드
- `metric_for_best_model: "eval_loss"`: 검증 손실 기준

#### 13.1.7 _create_callbacks() — 콜백 생성

```python
def _create_callbacks(self) -> list:
    """학습 콜백을 생성합니다."""
    
    callbacks = []
    
    if self.training_config.early_stopping.enabled:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=self.training_config.early_stopping.patience,
                early_stopping_threshold=self.training_config.early_stopping.threshold
            )
        )
    
    return callbacks
```

**Early Stopping:**
- `patience: 3`: 3 에폭 동안 개선 없으면 중단
- `threshold: 0.01`: 최소 개선 폭

#### 13.1.8 train() — 학습 실행

```python
def train(self, dataset_dict: DatasetDict) -> Path:
    """LoRA 파인튜닝을 수행합니다."""
    
    self.logger.info("학습 시작")
    
    # 1. 모델 로드
    model, tokenizer = self._load_model()
    
    # 2. LoRA 적용
    lora_config = self._create_lora_config()
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # 3. 학습 인자 및 콜백
    training_args = self._create_training_args()
    callbacks = self._create_callbacks()
    
    # 4. SFTTrainer 생성
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict["eval"],
        processing_class=tokenizer,
        callbacks=callbacks
    )
    
    # 5. 학습 실행
    trainer.train()
    
    # 6. 어댑터 저장
    adapter_path = self.output_dir / "adapter"
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    
    self.logger.info(f"학습 완료: {adapter_path}")
    return adapter_path
```

**출력:**
```
trainable params: 8,388,608 || all params: 3,008,388,608 || trainable%: 0.2788
```

---

## 14. exporter/ — 모델 내보내기 모듈

### 14.1 hf_export.py (155줄)

#### 14.1.1 역할

LoRA 어댑터를 기본 모델에 병합하고 safetensors 형식으로 저장합니다.

#### 14.1.2 HFExporter 클래스

```python
class HFExporter:
    def __init__(self, config: SLMConfig):
        self.student_model = config.student.model
        self.merge_lora = config.export.merge_lora
        self.output_format = config.export.output_format
        self.logger = get_logger("hf_export")
```

#### 14.1.3 merge_and_save() — 병합 및 저장

```python
def merge_and_save(self, adapter_path: Path, output_dir: Path) -> Path:
    """LoRA 어댑터를 기본 모델에 병합하고 저장합니다."""
    
    self.logger.info("LoRA 어댑터 병합 중...")
    
    # 1. 기본 모델 로드
    base_model = AutoModelForCausalLM.from_pretrained(
        self.student_model,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    
    # 2. 어댑터 로드
    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    
    # 3. 병합
    merged_model = model.merge_and_unload()
    
    # 4. 저장
    output_dir.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(
        output_dir,
        safe_serialization=(self.output_format == "safetensors")
    )
    
    # 5. 토크나이저 저장
    tokenizer = AutoTokenizer.from_pretrained(
        adapter_path,
        trust_remote_code=True
    )
    tokenizer.save_pretrained(output_dir)
    
    self.logger.info(f"병합 완료: {output_dir}")
    return output_dir
```

**특징:**
- `merge_and_unload()`: LoRA 가중치를 기본 모델에 통합
- `safe_serialization=True`: safetensors 형식 사용 (보안 + 빠른 로드)

#### 14.1.4 save_adapter_only() — 어댑터만 저장

```python
def save_adapter_only(self, adapter_path: Path, output_dir: Path) -> Path:
    """어댑터만 복사합니다 (PEFT 형식)."""
    
    self.logger.info("어댑터 복사 중...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(adapter_path, output_dir, dirs_exist_ok=True)
    
    self.logger.info(f"어댑터 저장 완료: {output_dir}")
    return output_dir
```

**사용 사례:**
- 어댑터만 배포하여 용량 절약
- PEFT 라이브러리로 런타임에 병합

#### 14.1.5 export() — 내보내기 실행

```python
def export(self, adapter_path: Path, output_dir: Path) -> Path:
    """설정에 따라 병합 또는 어댑터만 저장합니다."""
    
    if self.merge_lora:
        return self.merge_and_save(adapter_path, output_dir)
    else:
        return self.save_adapter_only(adapter_path, output_dir)
```

### 14.2 ollama_export.py (177줄)

#### 14.2.1 역할

Ollama Modelfile을 생성하고 선택적으로 모델을 Ollama에 등록합니다.

#### 14.2.2 OllamaExporter 클래스

```python
class OllamaExporter:
    def __init__(self, config: SLMConfig):
        self.model_name = config.export.ollama.model_name
        self.system_prompt = config.export.ollama.system_prompt
        self.parameters = config.export.ollama.parameters
        self.logger = get_logger("ollama_export")
```

#### 14.2.3 generate_modelfile() — Modelfile 생성

```python
def generate_modelfile(self, model_dir: Path, output_path: Path) -> Path:
    """Ollama Modelfile을 생성합니다."""
    
    lines = [
        f"FROM {model_dir}",
        f'SYSTEM """{self.system_prompt}"""',
    ]
    
    # 파라미터 추가
    for key, value in self.parameters.items():
        lines.append(f"PARAMETER {key} {value}")
    
    # 파일 저장
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    
    self.logger.info(f"Modelfile 생성 완료: {output_path}")
    return output_path
```

**Modelfile 예시:**
```
FROM /path/to/final_model
SYSTEM """당신은 도움이 되는 AI 어시스턴트입니다."""
PARAMETER temperature 0.7
PARAMETER top_p 0.9
```

#### 12.2.4 create_model() — Ollama 등록

```python
def create_model(self, modelfile_path: Path) -> bool:
    """ollama create 명령으로 모델을 등록합니다."""
    
    self.logger.info(f"Ollama 모델 생성 중: {self.model_name}")
    
    try:
        result = subprocess.run(
            ["ollama", "create", self.model_name, "-f", str(modelfile_path)],
            capture_output=True,
            text=True,
            timeout=300  # 5분
        )
        
        if result.returncode == 0:
            self.logger.info(f"Ollama 모델 생성 완료: {self.model_name}")
            return True
        else:
            self.logger.error(f"Ollama 생성 실패: {result.stderr}")
            return False
            
    except FileNotFoundError:
        self.logger.warning("ollama 명령을 찾을 수 없습니다.")
        return False
    except subprocess.TimeoutExpired:
        self.logger.error("Ollama 생성 타임아웃 (5분 초과)")
        return False
```

#### 12.2.5 export() — 내보내기 실행

```python
def export(self, model_dir: Path, output_dir: Path) -> Path:
    """Modelfile 생성 및 Ollama 등록을 수행합니다."""
    
    # 1. Modelfile 생성
    modelfile_path = output_dir / "Modelfile"
    self.generate_modelfile(model_dir, modelfile_path)
    
    # 2. Ollama 감지 및 등록
    if shutil.which("ollama"):
        success = self.create_model(modelfile_path)
        if success:
            self.logger.info(
                f"모델 사용 준비 완료:\n"
                f"  ollama run {self.model_name}"
            )
    else:
        self.logger.info(
            f"Ollama가 설치되지 않았습니다.\n"
            f"수동 등록 명령어:\n"
            f"  ollama create {self.model_name} -f {modelfile_path}"
        )
    
    return modelfile_path
```

**동작:**
1. Modelfile 생성
2. `ollama` 명령 감지
3. 자동 등록 시도 또는 수동 명령어 안내

---

## 15. utils.py — 유틸리티 (~30줄)

### 15.1 역할

Rich 기반 구조화된 로깅을 설정합니다.

### 15.2 setup_logging() — 로깅 초기화

```python
_configured = False

def setup_logging(level: str = "INFO") -> logging.Logger:
    """Rich 핸들러로 로깅을 설정합니다."""
    
    global _configured
    if _configured:
        return logging.getLogger("slm_factory")
    
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True)]
    )
    
    _configured = True
    return logging.getLogger("slm_factory")
```

**특징:**
- `RichHandler`: 컬러 출력 + 트레이스백 포맷팅
- `_configured` 가드: 중복 설정 방지 (여러 모듈에서 호출 가능)

### 15.3 get_logger() — 로거 생성

```python
def get_logger(name: str) -> logging.Logger:
    """네임스페이스가 지정된 로거를 반환합니다."""
    
    return logging.getLogger(f"slm_factory.{name}")
```

**사용 예시:**
```python
from slm_factory.utils import get_logger

logger = get_logger("parsers.pdf")
logger.info("PDF 파싱 시작")
logger.error("파싱 실패", exc_info=True)
```

**출력:**
```
[slm_factory.parsers.pdf] PDF 파싱 시작
[slm_factory.parsers.pdf] 파싱 실패
Traceback (most recent call last):
  ...
```

### 15.4 사용 패턴

모든 모듈에서 파일 상단에 호출합니다:

```python
from slm_factory.utils import get_logger

logger = get_logger("module_name")

class MyClass:
    def __init__(self):
        self.logger = get_logger("module_name.MyClass")
```

**네임스페이스 예시:**
- `slm_factory.pipeline`
- `slm_factory.parsers.pdf`
- `slm_factory.teacher.qa_generator`
- `slm_factory.validator.rules`
- `slm_factory.trainer.lora_trainer`

---

## 16. 모듈 간 상호작용

### 16.1 파이프라인 실행 흐름

```
CLI (cli.py)
   ↓ load_config()
Config (config.py)
   ↓ Pipeline(config)
Pipeline (pipeline.py)
   ↓ step_parse()
ParserRegistry (parsers/__init__.py)
   ↓ parse_directory()
PDFParser/HWPXParser/... (parsers/*.py)
   ↓ ParsedDocument 반환 (models.py에서 정의)
Pipeline.step_generate()
   ↓ QAGenerator(config)
QAGenerator (teacher/qa_generator.py)
   ↓ create_teacher()
OllamaTeacher/OpenAICompatTeacher (teacher/*.py)
   ↓ QAPair 반환 (models.py에서 정의)
Pipeline.step_validate()
   ↓ RuleValidator + GroundednessChecker
Validator (validator/*.py)
   ↓ 검증된 QAPair 반환
Pipeline.step_score() (선택적)
   ↓ QualityScorer
Scorer (scorer.py)
   ↓ 품질 점수 평가 및 필터링
Pipeline.step_augment() (선택적)
   ↓ DataAugmenter
Augmenter (augmenter.py)
   ↓ 데이터 증강 (질문 패러프레이즈)
Pipeline.step_analyze() (선택적)
   ↓ DataAnalyzer
Analyzer (analyzer.py)
   ↓ 분석 보고서 생성
Pipeline.step_convert()
   ↓ ChatFormatter
Formatter (converter.py)
   ↓ JSONL 반환
Pipeline.step_train()
   ↓ DataLoader + LoRATrainer (trainer/lora_trainer.py에 통합)
Trainer (trainer/lora_trainer.py)
   ↓ 어댑터 경로 반환
Pipeline.step_export()
   ↓ HFExporter + OllamaExporter
Exporter (exporter/*.py)
   ↓ 최종 모델 경로 반환
```

### 16.2 설정 전파

모든 모듈은 `SLMConfig` 객체를 통해 설정을 전달받습니다:

```python
# Pipeline
pipeline = Pipeline(config)

# QAGenerator
generator = QAGenerator(config)  # config.teacher, config.questions 사용

# RuleValidator
validator = RuleValidator(config.validation)

# ChatFormatter
formatter = ChatFormatter(config)  # config.student, config.export 사용

# LoRATrainer
trainer = LoRATrainer(config)  # config.student, config.training 사용
```

### 16.3 로깅 통합

모든 모듈은 `utils.logging`을 통해 일관된 로깅을 수행합니다:

```python
from slm_factory.utils import get_logger

logger = get_logger("module_name")
logger.info("작업 시작")
logger.error("에러 발생", exc_info=True)
```

**출력 예시:**
```
[slm_factory.pipeline] Step 1/9: 문서 파싱 시작
[slm_factory.parsers.base] 파싱 중: 5개 파일
[slm_factory.parsers.pdf] report.pdf 파싱 완료
[slm_factory.pipeline] 파싱 완료: 5개 문서
[slm_factory.scorer] 품질 점수 평가 완료: 85/100 통과
[slm_factory.augmenter] 데이터 증강 완료: 원본 85 + 증강 170 = 총 255개
[slm_factory.analyzer] 학습 데이터 분석 보고서 생성 완료
```

---

## 17. 확장 가이드

### 17.1 새로운 파서 추가

```python
# parsers/docx.py
from .base import BaseParser, ParsedDocument, registry

@registry.register
class DOCXParser(BaseParser):
    extensions = [".docx"]
    
    def parse(self, path: Path) -> ParsedDocument:
        # python-docx 사용
        doc = Document(path)
        content = "\n\n".join([p.text for p in doc.paragraphs])
        
        return ParsedDocument(
            doc_id=path.stem,
            title=doc.core_properties.title or path.stem,
            content=content,
            tables=[],
            metadata={}
        )
```

### 17.2 새로운 Teacher 백엔드 추가

```python
# teacher/anthropic.py
from .base import BaseTeacher

class AnthropicTeacher(BaseTeacher):
    def __init__(self, config: TeacherConfig):
        self.api_key = config.api_key
        self.model = config.model
    
    def generate(self, prompt: str, **kwargs) -> str:
        # Anthropic API 호출
        ...
        return response.content[0].text

# teacher/__init__.py
def create_teacher(config: TeacherConfig) -> BaseTeacher:
    if config.backend == "anthropic":
        return AnthropicTeacher(config)
    ...
```

### 17.3 새로운 검증 규칙 추가

```python
# validator/rules.py
class RuleValidator:
    def validate_one(self, pair: QAPair) -> ValidationResult:
        reasons = []
        
        # 기존 검증...
        
        # 새로운 규칙: 특정 키워드 필수 포함
        required_keywords = ["중요", "핵심"]
        if not any(kw in pair.answer for kw in required_keywords):
            reasons.append("필수 키워드 누락")
        
        return ValidationResult(
            passed=len(reasons) == 0,
            reasons=reasons
        )
```

---

## 18. 요약

SLM Factory는 29개 파일, 약 3,900줄의 코드로 구성된 모듈형 파이프라인입니다. 각 모듈은 명확한 책임을 가지며, 설정 시스템을 통해 유연하게 동작을 제어할 수 있습니다. 리팩토링으로 `utils/`와 `converter/` 디렉토리가 최상위 모듈로 통합되었고, 공유 데이터 모델이 `models.py`로 중앙화되었습니다. 품질 점수 평가(scorer.py), 데이터 증강(augmenter.py), 분석(analyzer.py) 모듈이 추가되어 학습 데이터 품질 관리가 강화되었습니다.

**핵심 모듈:**
- **config.py**: 중앙 설정 시스템 (Pydantic 검증)
- **pipeline.py**: 9단계 오케스트레이터
- **models.py**: 공유 데이터 모델 (QAPair, ParsedDocument)
- **converter.py**: 채팅 템플릿 변환 (최상위 모듈)
- **utils.py**: 로깅 유틸리티 (최상위 모듈)
- **scorer.py**: QA 품질 점수 평가 (Teacher LLM 기반)
- **augmenter.py**: QA 데이터 증강 (질문 패러프레이즈)
- **analyzer.py**: 학습 데이터 분석 (통계 보고서)
- **parsers/**: 4개 형식 지원 (PDF, HWPX, HTML, TXT)
- **teacher/**: 2개 백엔드 (Ollama, OpenAI 호환)
- **validator/**: 규칙 + 임베딩 검증
- **trainer/**: LoRA 파인튜닝 (DataLoader 통합)
- **exporter/**: HuggingFace + Ollama 내보내기

**설계 원칙:**
- 모듈 독립성: 각 단계를 개별 실행 가능
- 중간 저장: 디버깅 및 재개 지원
- 에러 격리: 개별 파일 실패가 전체 프로세스 중단하지 않음
- 확장성: 레지스트리 패턴으로 플러그인 추가 용이

---

## 관련 문서

- [README](../README.md) — 프로젝트 소개, 설치, 빠른 시작, CLI 레퍼런스
- [아키텍처 가이드](architecture.md) — 내부 구조, 설계 패턴, 데이터 흐름
- [설정 레퍼런스](configuration.md) — `project.yaml`의 모든 설정 옵션 상세 설명
