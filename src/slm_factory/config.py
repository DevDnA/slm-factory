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
# 언어별 기본 프롬프트
# ---------------------------------------------------------------------------

_EN_DEFAULT_QA_SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions based strictly on "
    "the provided document. Answer only from the document content. Do not "
    "speculate or fabricate information. Be concise and factual. Include "
    "specific numbers, dates, and names when available. If the document "
    'does not contain relevant information, say "The document does not '
    'contain this information."'
)

_KO_DEFAULT_QA_SYSTEM_PROMPT = (
    "당신은 제공된 문서를 기반으로 질문에 답변하는 도우미입니다. "
    "반드시 문서에 포함된 내용만으로 답변하세요. "
    "추측하거나 정보를 지어내지 마세요. "
    "간결하고 사실에 기반하여 답변하세요. "
    "가능한 경우 구체적인 수치, 날짜, 이름을 포함하세요. "
    '문서에 관련 정보가 없으면 "해당 정보는 문서에 포함되어 있지 않습니다."라고 '
    "답변하세요."
)

_EN_DEFAULT_OLLAMA_SYSTEM_PROMPT = "You are a helpful domain-specific assistant."

_KO_DEFAULT_OLLAMA_SYSTEM_PROMPT = (
    "당신은 도메인 전문 지식을 바탕으로 사용자의 질문에 "
    "정확하고 유용하게 답변하는 어시스턴트입니다."
)


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

    formats: list[str] = Field(
        default_factory=lambda: ["pdf", "txt", "html", "md", "hwpx", "hwp", "docx"]
    )
    pdf: PdfOptions = Field(default_factory=PdfOptions)
    hwpx: HwpxOptions = Field(default_factory=HwpxOptions)


class TeacherConfig(BaseModel):
    """QA 생성을 위한 교사 LLM 설정입니다."""

    backend: Literal["ollama", "openai"] = "ollama"
    model: str = Field("qwen3.5:9b", min_length=1)
    api_base: str = Field("http://localhost:11434", min_length=1)
    api_key: str | None = None
    temperature: float = 0.3
    timeout: int = 300
    max_context_chars: int = 12000
    max_concurrency: int = 2


class QuestionsConfig(BaseModel):
    """질문 카테고리 및 생성 설정입니다.

    질문은 *categories* (카테고리명을 질문 문자열 리스트로 매핑하는 딕셔너리)를
    통해 인라인으로 정의하거나, *file*을 통해 외부 파일에서 로드할 수 있습니다.

    ``auto_generate=True``이면 Teacher LLM이 각 청크를 분석하여 질문-답변 쌍을
    자동으로 생성합니다. 기존 ``categories`` 고정질문이 있으면 자동생성 QA가
    **추가**됩니다. ``categories``가 비어있으면 자동생성만 수행합니다.
    """

    categories: dict[str, list[str]] = Field(default_factory=dict)
    file: Path | None = None
    auto_generate: bool = False
    questions_per_chunk: int | Literal["auto"] = Field(default="auto")
    system_prompt: str = _EN_DEFAULT_QA_SYSTEM_PROMPT
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
            raise FileNotFoundError(
                f"질문 파일을 찾을 수 없습니다: {path}. "
                f"questions.file 경로를 확인하세요."
            )
        return [q for questions in self.categories.values() for q in questions]

    def get_questions_with_categories(self) -> list[tuple[str, str]]:
        """(카테고리명, 질문) 튜플 리스트를 반환합니다.

        카테고리 정보를 유지하면서 모든 질문을 반환합니다.
        파일에서 로드한 질문은 카테고리가 빈 문자열입니다.
        """
        if self.file is not None:
            path = Path(self.file)
            if path.is_file():
                return [
                    ("", line.strip())
                    for line in path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
            raise FileNotFoundError(
                f"질문 파일을 찾을 수 없습니다: {path}. "
                f"questions.file 경로를 확인하세요."
            )
        return [
            (cat_name, q)
            for cat_name, questions in self.categories.items()
            for q in questions
        ]


class GroundednessConfig(BaseModel):
    """생성된 답변에 대한 의미론적 근거 검증입니다."""

    enabled: bool = True
    model: str = "all-MiniLM-L6-v2"
    threshold: float = 0.3


class ValidationConfig(BaseModel):
    """QA 쌍 검증 및 필터링 규칙입니다."""

    enabled: bool = True
    min_answer_length: int = 20
    max_answer_length: int = 2000
    remove_empty: bool = True
    deduplicate: bool = True
    reject_patterns: list[str] = Field(
        default_factory=lambda: [
            "(?i)i don't know",
            "(?i)not (available|provided|mentioned|found)",
            "(?i)the document does not contain",
            "(?:알|확인할|파악할|판단할) 수 없",
            "정보.{0,8}(?:없|부족|찾을 수 없)",
            "(?:문서|자료|내용)에서?.{0,12}(?:찾을 수 없|포함되어 있지 않|언급되지 않|다루고 있지 않)",
        ]
    )
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

    enabled: bool = True
    threshold: float = 3.0
    max_concurrency: int = 3

    @model_validator(mode="after")
    def _check_scoring_threshold(self) -> "ScoringConfig":
        """점수 기준값의 유효성을 검증합니다."""
        if self.enabled and not (1.0 <= self.threshold <= 5.0):
            raise ValueError(f"threshold({self.threshold})는 1.0~5.0 범위여야 합니다")
        return self


class AugmentConfig(BaseModel):
    """QA 쌍 데이터 증강 설정입니다."""

    enabled: bool = True
    num_variants: int = 2
    max_concurrency: int = 3
    min_similarity: float = 0.3
    """패러프레이즈된 질문과 원본 질문 간 최소 토큰 겹침 비율입니다.
    이 값 미만이면 의미가 변질된 것으로 간주하여 제거합니다 (0.0~1.0)."""


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

    r: int = 8
    alpha: int = 16
    dropout: float = 0.1
    target_modules: Union[str, list[str]] = "auto"
    use_rslora: bool = False


class EarlyStoppingConfig(BaseModel):
    """훈련을 위한 조기 종료 기준입니다."""

    enabled: bool = True
    patience: int = 3
    threshold: float = 0.01


class QuantizationConfig(BaseModel):
    """훈련을 위한 양자화 설정입니다."""

    enabled: bool = True
    bits: int = 4


class TrainingConfig(BaseModel):
    """전체 훈련 하이퍼파라미터 설정입니다."""

    lora: LoraConfig = Field(default_factory=LoraConfig)
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2.0e-4
    lr_scheduler: str = "cosine"
    warmup_ratio: float = 0.1
    num_epochs: int | Literal["auto"] = "auto"
    """에포크 수입니다. ``"auto"``이면 학습 데이터 양에 따라 자동 결정합니다."""
    early_stopping: EarlyStoppingConfig = Field(default_factory=EarlyStoppingConfig)
    optimizer: str = "adamw_torch_fused"  # 디바이스 자동 감지로 런타임에 조정됨
    bf16: bool = True  # 디바이스 자동 감지로 런타임에 조정됨
    train_split: float = 0.9
    save_strategy: str = "epoch"
    quantization: QuantizationConfig = Field(default_factory=QuantizationConfig)
    neftune_noise_alpha: float | None = None
    """NEFTune 임베딩 노이즈 강도입니다. 설정 시 학습 중 임베딩에 노이즈를
    주입하여 일반화 성능을 5~15% 향상시킵니다. 권장값: 5.0 (2B 이하), 10.0 (4B 이상)."""

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
    system_prompt: str = _EN_DEFAULT_OLLAMA_SYSTEM_PROMPT
    parameters: dict[str, Any] = Field(
        default_factory=lambda: {
            "temperature": 0.7,
            "top_p": 0.9,
            "num_ctx": 4096,
        }
    )

    @model_validator(mode="after")
    def _sanitize_model_name(self) -> "OllamaExportConfig":
        from pathlib import PurePosixPath

        if "/" in self.model_name:
            sanitized = PurePosixPath(self.model_name).name
            if sanitized:
                self.model_name = sanitized
        return self


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

    enabled: bool = True
    test_split: float = 0.1
    metrics: list[str] = Field(default_factory=lambda: ["bleu", "rouge", "llm_judge"])
    max_samples: int = 20
    max_tokens: int = 512
    """Ollama 생성 최대 토큰 수. 평가 시 응답 길이를 제한하여 무한 생성을 방지합니다."""
    output_file: str = "eval_results.json"
    quality_gate: bool = True
    """품질 게이트 활성화 여부입니다. True이면 평가 결과가 임계값 미달 시 경고합니다."""
    quality_thresholds: dict[str, float] = Field(
        default_factory=lambda: {"bleu": 0.1, "rougeL": 0.2}
    )
    """메트릭별 최소 통과 임계값입니다. 평균 점수가 이 값 미만이면 품질 게이트 실패입니다."""
    llm_judge_model: str = ""
    """LLM-as-Judge에 사용할 모델입니다. 빈 문자열이면 teacher.model을 사용합니다."""

    @model_validator(mode="after")
    def _check_eval_params(self) -> "EvalConfig":
        """평가 설정의 유효성을 검증합니다."""
        if not (0.0 < self.test_split < 1.0):
            raise ValueError(f"test_split({self.test_split})은 0과 1 사이여야 합니다")
        if self.max_samples < 1:
            raise ValueError(f"max_samples({self.max_samples})는 1 이상이어야 합니다")
        return self


class RefinementConfig(BaseModel):
    """Iterative Refinement 설정입니다.

    학습 후 평가에서 약점이 발견된 QA에 대해
    Teacher LLM으로 추가 학습 데이터를 생성하고 재학습합니다.
    """

    enabled: bool = False
    """Refinement 활성화 여부입니다. 기본 비활성."""

    max_rounds: int = 1
    """최대 Refinement 반복 횟수입니다."""

    llm_judge_threshold: float = 0.6
    """이 점수 미만인 QA를 약점으로 판단합니다 (0.0~1.0)."""

    @model_validator(mode="after")
    def _check_refinement_params(self) -> "RefinementConfig":
        """Refinement 설정의 유효성을 검증합니다."""
        if self.max_rounds < 1:
            raise ValueError(f"max_rounds({self.max_rounds})는 1 이상이어야 합니다")
        if not (0.0 <= self.llm_judge_threshold <= 1.0):
            raise ValueError(
                f"llm_judge_threshold({self.llm_judge_threshold})는 "
                "0.0~1.0 범위여야 합니다"
            )
        return self


class RagConfig(BaseModel):
    """RAG 서비스 구축 설정입니다.

    ``corpus.parquet``을 ChromaDB에 임베딩하여 적재하고,
    Ollama SLM과 연동하는 RAG API 서버를 실행합니다.
    """

    embedding_model: str = "BAAI/bge-m3"
    """임베딩 모델 이름 (sentence-transformers 호환)."""

    vector_db_path: str = "chroma_db"
    """ChromaDB 저장 경로. ``paths.output`` 하위에 생성됩니다."""

    collection_name: str = "corpus"
    """ChromaDB 컬렉션 이름."""

    top_k: int = 5
    """검색 시 반환할 최대 문서 청크 수."""

    batch_size: int = 64
    """ChromaDB 인덱싱 시 배치 크기."""

    server_host: str = "0.0.0.0"
    """RAG API 서버 바인드 호스트."""

    server_port: int = 8000
    """RAG API 서버 포트."""

    workers: int = 1
    """uvicorn 워커 프로세스 수."""

    log_level: str = "info"
    """uvicorn 로그 레벨."""

    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
    """CORS 허용 오리진 목록."""

    request_timeout: float = 120.0
    """RAG 질의 타임아웃 (초)."""

    max_tokens: int = 512
    """Ollama 생성 최대 토큰 수. 응답 길이를 제한하여 불필요한 반복 생성을 방지합니다."""

    ollama_model: str = ""
    """Ollama 모델명. 빈 문자열이면 ``export.ollama.model_name``을 사용합니다."""

    reranker_enabled: bool = False
    """검색 결과를 cross-encoder로 재정렬하여 정확도를 높입니다."""

    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    """Reranker 모델명 (sentence-transformers CrossEncoder 호환)."""

    hybrid_search: bool = False
    """벡터 검색과 키워드 검색(BM25)을 결합하여 검색 재현율을 높입니다."""

    query_rewriting: bool = False
    """짧은 질의를 LLM으로 확장하여 검색 품질을 향상합니다."""

    @model_validator(mode="after")
    def _check_rag_params(self) -> "RagConfig":
        """RAG 설정의 유효성을 검증합니다."""
        if self.top_k < 1:
            raise ValueError(f"top_k({self.top_k})는 1 이상이어야 합니다")
        if self.server_port < 1 or self.server_port > 65535:
            raise ValueError(
                f"server_port({self.server_port})는 1~65535 범위여야 합니다"
            )
        if self.batch_size < 1:
            raise ValueError(f"batch_size({self.batch_size})는 1 이상이어야 합니다")
        if self.workers < 1:
            raise ValueError(f"workers({self.workers})는 1 이상이어야 합니다")
        if self.request_timeout <= 0:
            raise ValueError(
                f"request_timeout({self.request_timeout})은 0보다 커야 합니다"
            )
        return self


class AutoRAGExportConfig(BaseModel):
    """AutoRAG 연동을 위한 데이터 내보내기 설정입니다.

    slm-factory의 파싱된 문서와 QA 쌍을 AutoRAG 평가용
    ``corpus.parquet`` + ``qa.parquet`` 형식으로 변환합니다.
    """

    enabled: bool = True
    output_dir: str = "autorag"
    chunk_size: int = 512
    overlap_chars: int = 64

    @model_validator(mode="after")
    def _check_autorag_export_params(self) -> "AutoRAGExportConfig":
        """AutoRAG 내보내기 설정의 유효성을 검증합니다."""
        if self.chunk_size < 100:
            raise ValueError(f"chunk_size({self.chunk_size})는 100 이상이어야 합니다")
        if self.overlap_chars >= self.chunk_size:
            raise ValueError(
                f"overlap_chars({self.overlap_chars})는 "
                f"chunk_size({self.chunk_size})보다 작아야 합니다"
            )
        return self


class IncrementalConfig(BaseModel):
    """증분 학습 설정입니다."""

    enabled: bool = True
    hash_file: str = "document_hashes.json"
    merge_strategy: Literal["append", "replace"] = "append"
    resume_adapter: str = ""


class ReviewConfig(BaseModel):
    """QA 수동 리뷰 설정입니다."""

    enabled: bool = True
    auto_open: bool = True
    output_file: str = "qa_reviewed.json"


class CompareConfig(BaseModel):
    """모델 비교 설정입니다."""

    enabled: bool = False
    base_model: str = ""
    finetuned_model: str = ""
    metrics: list[str] = Field(default_factory=lambda: ["bleu", "rouge"])
    max_samples: int = 20
    max_tokens: int = 512
    """Ollama 생성 최대 토큰 수. 비교 시 응답 길이를 제한하여 무한 생성을 방지합니다."""
    output_file: str = "compare_results.json"

    @model_validator(mode="after")
    def _check_compare_params(self) -> "CompareConfig":
        """비교 설정의 유효성을 검증합니다."""
        if self.max_samples < 1:
            raise ValueError(f"max_samples({self.max_samples})는 1 이상이어야 합니다")
        if self.enabled and not self.base_model:
            raise ValueError("compare.enabled가 true일 때 base_model은 필수입니다")
        if self.enabled and not self.finetuned_model:
            raise ValueError("compare.enabled가 true일 때 finetuned_model은 필수입니다")
        return self


class EvolveConfig(BaseModel):
    """자동 진화 파이프라인 설정입니다.

    ``tool evolve`` 명령의 동작을 제어합니다.
    증분 업데이트 → 전체 재학습 → 품질 게이트 → 조건부 배포를
    단일 명령으로 실행합니다.
    """

    quality_gate: bool = True
    """품질 게이트 활성화 여부입니다. True이면 이전 모델보다 나은 경우에만 배포합니다."""

    gate_metric: str = "rougeL"
    """품질 비교에 사용할 메트릭입니다 (bleu, rouge1, rouge2, rougeL)."""

    gate_min_improvement: float = 0.0
    """최소 개선율(%)입니다. 0이면 어떤 개선이든 통과합니다."""

    version_format: str = "date"
    """버전 형식입니다. 현재 ``date`` (vYYYYMMDD) 형식만 지원합니다."""

    history_file: str = "evolve_history.json"
    """진화 히스토리 파일명입니다 (출력 디렉토리에 저장)."""

    keep_previous_versions: int = 3
    """보관할 이전 버전 수입니다. 초과분은 ``ollama rm``으로 제거됩니다."""

    @model_validator(mode="after")
    def _check_evolve_params(self) -> "EvolveConfig":
        """진화 설정의 유효성을 검증합니다."""
        valid_metrics = {"bleu", "rouge1", "rouge2", "rougeL"}
        if self.gate_metric not in valid_metrics:
            raise ValueError(
                f"gate_metric({self.gate_metric})이 올바르지 않습니다. "
                f"지원: {', '.join(sorted(valid_metrics))}"
            )
        if self.keep_previous_versions < 0:
            raise ValueError(
                f"keep_previous_versions({self.keep_previous_versions})는 "
                "0 이상이어야 합니다"
            )
        return self


class DashboardConfig(BaseModel):
    """TUI 대시보드 설정입니다."""

    enabled: bool = True
    refresh_interval: float = 2.0
    theme: str = "dark"


class ChunkingConfig(BaseModel):
    """문서 청킹 설정입니다.

    긴 문서를 중첩(overlap)이 있는 청크로 분할하여 QA 생성 품질을 향상합니다.
    ``max_context_chars``보다 짧은 문서는 청킹하지 않습니다.
    """

    enabled: bool = True
    chunk_size: int | Literal["auto"] = "auto"
    """각 청크의 최대 문자 수입니다. ``"auto"``이면 문서 분석으로 자동 결정합니다."""

    overlap_chars: int = 500
    """연속된 청크 간 중첩 문자 수입니다. 문맥 연속성을 유지합니다."""

    @model_validator(mode="after")
    def _check_chunking_params(self) -> "ChunkingConfig":
        """청킹 파라미터의 유효성을 검증합니다."""
        if self.overlap_chars < 0:
            raise ValueError(
                f"overlap_chars({self.overlap_chars})는 0 이상이어야 합니다"
            )
        if self.chunk_size == "auto":
            return self
        if self.chunk_size < 1000:
            raise ValueError(f"chunk_size({self.chunk_size})는 1000 이상이어야 합니다")
        if self.overlap_chars >= self.chunk_size:
            raise ValueError(
                f"overlap_chars({self.overlap_chars})는 "
                f"chunk_size({self.chunk_size})보다 작아야 합니다"
            )
        return self


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
    refinement: RefinementConfig = Field(default_factory=RefinementConfig)
    autorag_export: AutoRAGExportConfig = Field(default_factory=AutoRAGExportConfig)
    rag: RagConfig = Field(default_factory=RagConfig)
    incremental: IncrementalConfig = Field(default_factory=IncrementalConfig)
    review: ReviewConfig = Field(default_factory=ReviewConfig)
    compare: CompareConfig = Field(default_factory=CompareConfig)
    evolve: EvolveConfig = Field(default_factory=EvolveConfig)
    dashboard: DashboardConfig = Field(default_factory=DashboardConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)

    @model_validator(mode="before")
    @classmethod
    def _strip_none_sections(cls, values: dict[str, Any]) -> dict[str, Any]:
        """값이 ``None``인 최상위 키를 제거하여 기본값이 적용되도록 합니다."""
        if isinstance(values, dict):
            return {k: v for k, v in values.items() if v is not None}
        return values

    @model_validator(mode="after")
    def _apply_language_defaults(self) -> "SLMConfig":
        if self.project.language != "ko":
            return self
        if self.questions.system_prompt == _EN_DEFAULT_QA_SYSTEM_PROMPT:
            self.questions.system_prompt = _KO_DEFAULT_QA_SYSTEM_PROMPT
        if self.export.ollama.system_prompt == _EN_DEFAULT_OLLAMA_SYSTEM_PROMPT:
            self.export.ollama.system_prompt = _KO_DEFAULT_OLLAMA_SYSTEM_PROMPT
        return self


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

    # 최후의 수단: 최소한의 내장 기본값 반환 (YAML 형식)
    return yaml.dump(
        SLMConfig().model_dump(),
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=False,
    )
