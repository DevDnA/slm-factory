"""Pydantic v2 + YAML를 사용한 slm-factory 설정 시스템입니다.

YAML 파일에서 프로젝트 설정을 로드하고 검증합니다.
파싱, 교사 모델, 질문, 검증, 훈련, 내보내기 등 모든 파이프라인 설정에 대한
타입 안전 접근을 제공합니다.
"""

from __future__ import annotations

import importlib.resources
import logging
from pathlib import Path
from typing import Any, Literal, Union

import yaml
from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# 하위 모델
# ---------------------------------------------------------------------------


class ProjectConfig(BaseModel):
    """최상위 프로젝트 메타데이터입니다."""

    name: str = Field("my-project", min_length=1)
    version: str = "1.0.0"
    language: str = "en"


class PathsConfig(BaseModel):
    """입출력 디렉토리 경로입니다 (검증 시 자동 생성됨)."""

    documents: Path = Path("./documents")
    output: Path = Path("./output")

    def ensure_dirs(self) -> None:
        """설정된 디렉토리가 없으면 생성합니다."""
        self.documents.mkdir(parents=True, exist_ok=True)
        self.output.mkdir(parents=True, exist_ok=True)


class PdfOptions(BaseModel):
    """PDF 파싱 옵션입니다."""

    extract_tables: bool = True


class HwpxOptions(BaseModel):
    """HWPX (한글 문서) 파싱 옵션입니다."""

    apply_spacing: bool = True


class ParsingConfig(BaseModel):
    """문서 파싱 설정입니다."""

    formats: list[str] = Field(default_factory=lambda: ["pdf", "txt", "html"])
    pdf: PdfOptions = Field(default_factory=PdfOptions)
    hwpx: HwpxOptions = Field(default_factory=HwpxOptions)


class TeacherConfig(BaseModel):
    """QA 생성을 위한 교사 LLM 설정입니다."""

    backend: Literal["ollama", "openai"] = "ollama"
    model: str = Field("qwen3:8b", min_length=1)
    api_base: str = Field("http://localhost:11434", min_length=1)
    api_key: str | None = None
    temperature: float = 0.3
    timeout: int = 180
    max_context_chars: int = 12000
    max_concurrency: int = 4


class QuestionsConfig(BaseModel):
    """질문 카테고리 및 생성 설정입니다.

    질문은 *categories* (카테고리명을 질문 문자열 리스트로 매핑하는 딕셔너리)를
    통해 인라인으로 정의하거나, *file*을 통해 외부 파일에서 로드할 수 있습니다.
    """

    categories: dict[str, list[str]] = Field(default_factory=dict)
    file: Path | None = None
    system_prompt: str = (
        "You are a helpful assistant that answers questions based strictly on "
        "the provided document. Answer only from the document content. Do not "
        "speculate or fabricate information. Be concise and factual. Include "
        "specific numbers, dates, and names when available. If the document "
        'does not contain relevant information, say "The document does not '
        'contain this information."'
    )
    output_format: str = "alpaca"

    def get_all_questions(self) -> list[str]:
        """모든 카테고리 질문을 단일 리스트로 평탄화합니다.

        *file*이 기존 텍스트 파일을 가리키면, 인라인 카테고리 대신
        파일의 비어있지 않은 줄들을 반환합니다.
        """
        if self.file is not None:
            path = Path(self.file)
            if path.is_file():
                return [
                    line.strip()
                    for line in path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
        return [q for questions in self.categories.values() for q in questions]


class GroundednessConfig(BaseModel):
    """생성된 답변에 대한 의미론적 근거 검증입니다."""

    enabled: bool = False
    model: str = "all-MiniLM-L6-v2"
    threshold: float = 0.3


class ValidationConfig(BaseModel):
    """QA 쌍 검증 및 필터링 규칙입니다."""

    enabled: bool = True
    min_answer_length: int = 20
    max_answer_length: int = 2000
    remove_empty: bool = True
    deduplicate: bool = True
    reject_patterns: list[str] = Field(default_factory=lambda: [
        "(?i)i don't know",
        "(?i)not (available|provided|mentioned|found)",
        "(?i)the document does not contain",
    ])
    groundedness: GroundednessConfig = Field(default_factory=GroundednessConfig)

    @model_validator(mode="after")
    def _check_answer_length_range(self) -> "ValidationConfig":
        """min_answer_length가 max_answer_length보다 작은지 검증합니다."""
        if self.min_answer_length >= self.max_answer_length:
            raise ValueError(
                f"min_answer_length({self.min_answer_length})는 "
                f"max_answer_length({self.max_answer_length})보다 작아야 합니다"
            )
        return self


class ScoringConfig(BaseModel):
    """교사 LLM을 사용한 QA 쌍 품질 점수 설정입니다."""
    enabled: bool = False
    threshold: float = 3.0
    max_concurrency: int = 4

    @model_validator(mode="after")
    def _check_scoring_threshold(self) -> "ScoringConfig":
        """점수 기준값의 유효성을 검증합니다."""
        if self.enabled and not (1.0 <= self.threshold <= 5.0):
            raise ValueError(f"threshold({self.threshold})는 1.0~5.0 범위여야 합니다")
        return self


class AugmentConfig(BaseModel):
    """QA 쌍 데이터 증강 설정입니다."""
    enabled: bool = False
    num_variants: int = 2
    max_concurrency: int = 4


class AnalyzerConfig(BaseModel):
    """학습 데이터 통계 분석 설정입니다."""
    enabled: bool = True
    output_file: str = "data_analysis.json"


class StudentConfig(BaseModel):
    """학생 (미세조정된) 모델 설정입니다."""

    model: str = Field("google/gemma-3-1b-it", min_length=1)
    max_seq_length: int = 4096


class LoraConfig(BaseModel):
    """LoRA 어댑터 하이퍼파라미터입니다."""

    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: Union[str, list[str]] = "auto"
    use_rslora: bool = False


class EarlyStoppingConfig(BaseModel):
    """훈련을 위한 조기 종료 기준입니다."""

    enabled: bool = True
    patience: int = 3
    threshold: float = 0.01


class QuantizationConfig(BaseModel):
    """훈련을 위한 양자화 설정입니다."""

    enabled: bool = False
    bits: int = 4


class TrainingConfig(BaseModel):
    """전체 훈련 하이퍼파라미터 설정입니다."""

    lora: LoraConfig = Field(default_factory=LoraConfig)
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2.0e-5
    lr_scheduler: str = "cosine"
    warmup_ratio: float = 0.1
    num_epochs: int = 20
    early_stopping: EarlyStoppingConfig = Field(default_factory=EarlyStoppingConfig)
    optimizer: str = "adamw_torch_fused"
    bf16: bool = True
    train_split: float = 0.9
    save_strategy: str = "epoch"
    quantization: QuantizationConfig = Field(default_factory=QuantizationConfig)

    @model_validator(mode="after")
    def _check_training_params(self) -> "TrainingConfig":
        """학습 파라미터의 유효성을 검증합니다."""
        if not (0.0 < self.train_split < 1.0):
            raise ValueError(f"train_split({self.train_split})은 0과 1 사이여야 합니다")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate({self.learning_rate})는 양수여야 합니다")
        return self


class OllamaExportConfig(BaseModel):
    """Ollama 특화 내보내기 설정입니다."""

    enabled: bool = True
    model_name: str = Field("my-project-model", min_length=1)
    system_prompt: str = "You are a helpful domain-specific assistant."
    parameters: dict[str, Any] = Field(default_factory=lambda: {
        "temperature": 0.7,
        "top_p": 0.9,
        "num_ctx": 4096,
    })


class ExportConfig(BaseModel):
    """모델 내보내기 및 패키징 설정입니다."""

    merge_lora: bool = True
    output_format: str = "safetensors"
    ollama: OllamaExportConfig = Field(default_factory=OllamaExportConfig)


# ---------------------------------------------------------------------------
# 신규 기능 설정
# ---------------------------------------------------------------------------


class EvalConfig(BaseModel):
    """학습된 모델의 자동 평가 설정입니다."""

    enabled: bool = False
    test_split: float = 0.1
    metrics: list[str] = Field(default_factory=lambda: ["bleu", "rouge"])
    max_samples: int = 50
    output_file: str = "eval_results.json"

    @model_validator(mode="after")
    def _check_eval_params(self) -> "EvalConfig":
        """평가 설정의 유효성을 검증합니다."""
        if not (0.0 < self.test_split < 1.0):
            raise ValueError(
                f"test_split({self.test_split})은 0과 1 사이여야 합니다"
            )
        if self.max_samples < 1:
            raise ValueError(
                f"max_samples({self.max_samples})는 1 이상이어야 합니다"
            )
        return self


class GGUFExportConfig(BaseModel):
    """GGUF 양자화 변환 설정입니다."""

    enabled: bool = False
    quantization_type: str = "q4_k_m"
    llama_cpp_path: str = ""

    @model_validator(mode="after")
    def _check_gguf_params(self) -> "GGUFExportConfig":
        """GGUF 양자화 타입의 유효성을 검증합니다."""
        valid_types = {"q4_0", "q4_1", "q4_k_m", "q4_k_s", "q5_0", "q5_1",
                       "q5_k_m", "q5_k_s", "q8_0", "f16", "f32"}
        if self.quantization_type.lower() not in valid_types:
            raise ValueError(
                f"quantization_type({self.quantization_type})이 올바르지 않습니다. "
                f"지원: {', '.join(sorted(valid_types))}"
            )
        return self


class IncrementalConfig(BaseModel):
    """증분 학습 설정입니다."""

    enabled: bool = False
    hash_file: str = "document_hashes.json"
    merge_strategy: Literal["append", "replace"] = "append"
    resume_adapter: str = ""


class DialogueConfig(BaseModel):
    """멀티턴 대화 생성 설정입니다."""

    enabled: bool = False
    min_turns: int = 2
    max_turns: int = 5
    include_single_qa: bool = True

    @model_validator(mode="after")
    def _check_dialogue_params(self) -> "DialogueConfig":
        """대화 턴 수 범위를 검증합니다."""
        if self.min_turns < 2:
            raise ValueError(
                f"min_turns({self.min_turns})는 2 이상이어야 합니다"
            )
        if self.min_turns > self.max_turns:
            raise ValueError(
                f"min_turns({self.min_turns})는 "
                f"max_turns({self.max_turns})보다 클 수 없습니다"
            )
        return self


class ReviewConfig(BaseModel):
    """QA 수동 리뷰 설정입니다."""

    enabled: bool = False
    auto_open: bool = True
    output_file: str = "qa_reviewed.json"


class CompareConfig(BaseModel):
    """모델 비교 설정입니다."""

    enabled: bool = False
    base_model: str = ""
    finetuned_model: str = ""
    metrics: list[str] = Field(default_factory=lambda: ["bleu", "rouge"])
    max_samples: int = 20
    output_file: str = "compare_results.json"

    @model_validator(mode="after")
    def _check_compare_params(self) -> "CompareConfig":
        """비교 설정의 유효성을 검증합니다."""
        if self.max_samples < 1:
            raise ValueError(
                f"max_samples({self.max_samples})는 1 이상이어야 합니다"
            )
        return self


class DashboardConfig(BaseModel):
    """TUI 대시보드 설정입니다."""

    enabled: bool = False
    refresh_interval: float = 2.0
    theme: str = "dark"


# ---------------------------------------------------------------------------
# 루트 설정
# ---------------------------------------------------------------------------


class SLMConfig(BaseModel):
    """slm-factory 프로젝트의 루트 설정 객체입니다.

    전체 ``project.yaml`` 스키마를 반영합니다. :func:`load_config`를 통해
    또는 딕셔너리/YAML에서 직접 인스턴스화됩니다.
    """

    project: ProjectConfig = Field(default_factory=ProjectConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    parsing: ParsingConfig = Field(default_factory=ParsingConfig)
    teacher: TeacherConfig = Field(default_factory=TeacherConfig)
    questions: QuestionsConfig = Field(default_factory=QuestionsConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    augment: AugmentConfig = Field(default_factory=AugmentConfig)
    analyzer: AnalyzerConfig = Field(default_factory=AnalyzerConfig)
    student: StudentConfig = Field(default_factory=StudentConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    gguf_export: GGUFExportConfig = Field(default_factory=GGUFExportConfig)
    incremental: IncrementalConfig = Field(default_factory=IncrementalConfig)
    dialogue: DialogueConfig = Field(default_factory=DialogueConfig)
    review: ReviewConfig = Field(default_factory=ReviewConfig)
    compare: CompareConfig = Field(default_factory=CompareConfig)
    dashboard: DashboardConfig = Field(default_factory=DashboardConfig)

    @model_validator(mode="before")
    @classmethod
    def _strip_none_sections(cls, values: dict[str, Any]) -> dict[str, Any]:
        """값이 ``None``인 최상위 키를 제거하여 기본값이 적용되도록 합니다."""
        if isinstance(values, dict):
            return {k: v for k, v in values.items() if v is not None}
        return values


# ---------------------------------------------------------------------------
# 공개 API
# ---------------------------------------------------------------------------

_TEMPLATE_PATH = "templates/project.yaml"


def load_config(path: str | Path) -> SLMConfig:
    """프로젝트 YAML 설정 파일을 로드하고 검증합니다.

    ``paths.documents``와 ``paths.output``이 상대 경로인 경우,
    설정 파일이 위치한 디렉토리를 기준으로 절대 경로로 변환합니다.

    매개변수
    ----------
    path:
        ``project.yaml`` 파일의 파일시스템 경로입니다.

    반환값
    -------
    SLMConfig
        완전히 검증된 설정 객체입니다.

    예외
    ------
    FileNotFoundError
        *path*가 존재하지 않으면 발생합니다.
    yaml.YAMLError
        파일이 유효한 YAML이 아니면 발생합니다.
    pydantic.ValidationError
        YAML 내용이 예상 스키마와 일치하지 않으면 발생합니다.
    """
    filepath = Path(path).resolve()
    if not filepath.is_file():
        raise FileNotFoundError(f"Config file not found: {filepath}")

    raw = yaml.safe_load(filepath.read_text(encoding="utf-8")) or {}
    config = SLMConfig.model_validate(raw)

    config_dir = filepath.parent
    if not config.paths.documents.is_absolute():
        config.paths.documents = (config_dir / config.paths.documents).resolve()
    if not config.paths.output.is_absolute():
        config.paths.output = (config_dir / config.paths.output).resolve()

    return config


def create_default_config() -> str:
    """기본 YAML 프로젝트 템플릿을 문자열로 반환합니다.

    패키지와 함께 제공되는 ``templates/project.yaml``을 읽습니다.
    개발 환경 (편집 가능 설치)과 빌드된 wheel 모두에서 작동하도록
    패키지 리소스 조회로 폴백합니다.
    """
    # 먼저 형제 경로 시도 (소스 / 편집 가능 설치에서 작동)
    pkg_root = Path(__file__).resolve().parent.parent.parent
    template = pkg_root / _TEMPLATE_PATH
    if template.is_file():
        return template.read_text(encoding="utf-8")

    # 폴백: importlib.resources (설치된 wheel)
    try:
        ref = importlib.resources.files("slm_factory").joinpath(
            "../../" + _TEMPLATE_PATH
        )
        return ref.read_text(encoding="utf-8")
    except Exception as e:
        logging.getLogger("slm_factory.config").debug(
            "importlib.resources에서 템플릿 로드 실패: %s", e
        )

    # 최후의 수단: 최소한의 내장 기본값 반환
    return SLMConfig().model_dump_json(indent=2)
