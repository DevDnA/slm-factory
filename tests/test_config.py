"""설정 시스템(config.py)의 단위 테스트입니다."""

from pathlib import Path

import pytest

from slm_factory.config import (
    EarlyStoppingConfig,
    ExportConfig,
    GroundednessConfig,
    HwpxOptions,
    LoraConfig,
    OllamaExportConfig,
    ParsingConfig,
    PathsConfig,
    PdfOptions,
    ProjectConfig,
    QuantizationConfig,
    QuestionsConfig,
    SLMConfig,
    StudentConfig,
    TeacherConfig,
    TrainingConfig,
    ValidationConfig,
    create_default_config,
    load_config,
)


# ---------------------------------------------------------------------------
# 개별 Config 클래스 기본값 검증
# ---------------------------------------------------------------------------


class TestProjectConfig:
    """ProjectConfig 기본값 테스트입니다."""

    def test_기본값(self):
        """name, version, language의 기본값을 검증합니다."""
        cfg = ProjectConfig()
        assert cfg.name == "my-project"
        assert cfg.version == "1.0.0"
        assert cfg.language == "en"


class TestPathsConfig:
    """PathsConfig 기본값 및 ensure_dirs 테스트입니다."""

    def test_기본값(self):
        """documents와 output의 기본 경로를 검증합니다."""
        cfg = PathsConfig()
        assert cfg.documents == Path("./documents")
        assert cfg.output == Path("./output")

    def test_ensure_dirs_디렉토리_생성(self, tmp_path):
        """ensure_dirs 호출 시 디렉토리가 실제로 생성되는지 확인합니다."""
        cfg = PathsConfig(
            documents=tmp_path / "docs",
            output=tmp_path / "out",
        )
        cfg.ensure_dirs()
        assert (tmp_path / "docs").is_dir()
        assert (tmp_path / "out").is_dir()


class TestPdfOptions:
    """PdfOptions 기본값 테스트입니다."""

    def test_기본값(self):
        """extract_tables의 기본값이 True인지 확인합니다."""
        cfg = PdfOptions()
        assert cfg.extract_tables is True


class TestHwpxOptions:
    """HwpxOptions 기본값 테스트입니다."""

    def test_기본값(self):
        """apply_spacing의 기본값이 True인지 확인합니다."""
        cfg = HwpxOptions()
        assert cfg.apply_spacing is True


class TestParsingConfig:
    """ParsingConfig 기본값 테스트입니다."""

    def test_기본값(self):
        """formats, pdf, hwpx의 기본값을 검증합니다."""
        cfg = ParsingConfig()
        assert cfg.formats == ["pdf"]
        assert isinstance(cfg.pdf, PdfOptions)
        assert isinstance(cfg.hwpx, HwpxOptions)


class TestTeacherConfig:
    """TeacherConfig 기본값 테스트입니다."""

    def test_기본값(self):
        """교사 설정의 기본값을 검증합니다."""
        cfg = TeacherConfig()
        assert cfg.backend == "ollama"
        assert cfg.model == "qwen3:8b"
        assert cfg.api_base == "http://localhost:11434"
        assert cfg.api_key is None
        assert cfg.temperature == 0.3
        assert cfg.timeout == 180
        assert cfg.max_context_chars == 12000
        assert cfg.max_concurrency == 4


class TestQuestionsConfig:
    """QuestionsConfig 기본값 및 메서드 테스트입니다."""

    def test_기본값(self):
        """categories, file, output_format의 기본값을 검증합니다."""
        cfg = QuestionsConfig()
        assert cfg.categories == {}
        assert cfg.file is None
        assert cfg.output_format == "alpaca"
        assert "helpful assistant" in cfg.system_prompt

    def test_get_all_questions_카테고리에서_추출(self):
        """categories 딕셔너리에서 모든 질문을 평탄화하여 반환하는지 확인합니다."""
        cfg = QuestionsConfig(
            categories={
                "factual": ["질문1", "질문2"],
                "reasoning": ["질문3"],
            }
        )
        questions = cfg.get_all_questions()
        assert len(questions) == 3
        assert "질문1" in questions
        assert "질문3" in questions

    def test_get_all_questions_파일에서_로드(self, tmp_path):
        """file 경로가 지정되면 파일에서 질문을 로드하는지 확인합니다."""
        qfile = tmp_path / "questions.txt"
        qfile.write_text("파일 질문1\n파일 질문2\n\n파일 질문3\n", encoding="utf-8")
        cfg = QuestionsConfig(file=qfile)
        questions = cfg.get_all_questions()
        assert questions == ["파일 질문1", "파일 질문2", "파일 질문3"]

    def test_get_all_questions_빈_카테고리(self):
        """카테고리가 비어있으면 빈 리스트를 반환하는지 확인합니다."""
        cfg = QuestionsConfig()
        assert cfg.get_all_questions() == []


class TestValidationConfig:
    """ValidationConfig 기본값 테스트입니다."""

    def test_기본값(self, sample_validation_config):
        """검증 설정의 기본값을 검증합니다."""
        cfg = sample_validation_config
        assert cfg.enabled is True
        assert cfg.min_answer_length == 20
        assert cfg.max_answer_length == 2000
        assert cfg.remove_empty is True
        assert cfg.deduplicate is True
        assert len(cfg.reject_patterns) == 3
        assert isinstance(cfg.groundedness, GroundednessConfig)


class TestGroundednessConfig:
    """GroundednessConfig 기본값 테스트입니다."""

    def test_기본값(self):
        """근거 검증 설정의 기본값을 검증합니다."""
        cfg = GroundednessConfig()
        assert cfg.enabled is False
        assert cfg.model == "all-MiniLM-L6-v2"
        assert cfg.threshold == 0.3


class TestStudentConfig:
    """StudentConfig 기본값 테스트입니다."""

    def test_기본값(self):
        """학생 모델 설정의 기본값을 검증합니다."""
        cfg = StudentConfig()
        assert cfg.model == "google/gemma-3-1b-it"
        assert cfg.max_seq_length == 4096


class TestLoraConfig:
    """LoraConfig 기본값 테스트입니다."""

    def test_기본값(self):
        """LoRA 하이퍼파라미터 기본값을 검증합니다."""
        cfg = LoraConfig()
        assert cfg.r == 16
        assert cfg.alpha == 32
        assert cfg.dropout == 0.05
        assert cfg.target_modules == "auto"
        assert cfg.use_rslora is False


class TestTrainingConfig:
    """TrainingConfig 기본값 테스트입니다."""

    def test_기본값(self):
        """훈련 설정의 기본값을 검증합니다."""
        cfg = TrainingConfig()
        assert cfg.batch_size == 4
        assert cfg.gradient_accumulation_steps == 4
        assert cfg.learning_rate == 2e-5
        assert cfg.lr_scheduler == "cosine"
        assert cfg.warmup_ratio == 0.1
        assert cfg.num_epochs == 20
        assert cfg.optimizer == "adamw_torch_fused"
        assert cfg.bf16 is True
        assert cfg.train_split == 0.9
        assert cfg.save_strategy == "epoch"
        assert isinstance(cfg.lora, LoraConfig)
        assert isinstance(cfg.early_stopping, EarlyStoppingConfig)
        assert isinstance(cfg.quantization, QuantizationConfig)


class TestExportConfig:
    """ExportConfig 기본값 테스트입니다."""

    def test_기본값(self):
        """내보내기 설정의 기본값을 검증합니다."""
        cfg = ExportConfig()
        assert cfg.merge_lora is True
        assert cfg.output_format == "safetensors"
        assert isinstance(cfg.ollama, OllamaExportConfig)


class TestOllamaExportConfig:
    """OllamaExportConfig 기본값 테스트입니다."""

    def test_기본값(self):
        """Ollama 내보내기 설정의 기본값을 검증합니다."""
        cfg = OllamaExportConfig()
        assert cfg.enabled is True
        assert cfg.model_name == "my-project-model"
        assert "helpful" in cfg.system_prompt


class TestEarlyStoppingConfig:
    """EarlyStoppingConfig 기본값 테스트입니다."""

    def test_기본값(self):
        """조기 종료 설정의 기본값을 검증합니다."""
        cfg = EarlyStoppingConfig()
        assert cfg.enabled is True
        assert cfg.patience == 3
        assert cfg.threshold == 0.01


class TestQuantizationConfig:
    """QuantizationConfig 기본값 테스트입니다."""

    def test_기본값(self):
        """양자화 설정의 기본값을 검증합니다."""
        cfg = QuantizationConfig()
        assert cfg.enabled is False
        assert cfg.bits == 4


# ---------------------------------------------------------------------------
# SLMConfig (루트 설정)
# ---------------------------------------------------------------------------


class TestSLMConfig:
    """SLMConfig 루트 설정의 테스트입니다."""

    def test_기본값(self, default_config):
        """SLMConfig의 모든 하위 설정이 기본값으로 초기화되는지 확인합니다."""
        cfg = default_config
        assert isinstance(cfg.project, ProjectConfig)
        assert isinstance(cfg.paths, PathsConfig)
        assert isinstance(cfg.parsing, ParsingConfig)
        assert isinstance(cfg.teacher, TeacherConfig)
        assert isinstance(cfg.questions, QuestionsConfig)
        assert isinstance(cfg.validation, ValidationConfig)
        assert isinstance(cfg.student, StudentConfig)
        assert isinstance(cfg.training, TrainingConfig)
        assert isinstance(cfg.export, ExportConfig)

    def test_strip_none_sections_None_섹션_제거(self):
        """_strip_none_sections가 None 값인 최상위 키를 제거하는지 확인합니다."""
        cfg = SLMConfig(**{"project": None, "teacher": None})
        # None 섹션이 제거되고 기본값이 적용되어야 합니다
        assert cfg.project.name == "my-project"
        assert cfg.teacher.backend == "ollama"

    def test_make_config_오버라이드(self, make_config):
        """make_config fixture로 특정 필드를 오버라이드할 수 있는지 확인합니다."""
        cfg = make_config(project={"name": "custom-project", "language": "ko"})
        assert cfg.project.name == "custom-project"
        assert cfg.project.language == "ko"


# ---------------------------------------------------------------------------
# 공개 API
# ---------------------------------------------------------------------------


class TestLoadConfig:
    """load_config 함수의 테스트입니다."""

    def test_정상_YAML_로드(self, tmp_yaml_config):
        """유효한 YAML 파일을 로드하여 SLMConfig를 반환하는지 확인합니다."""
        cfg = load_config(tmp_yaml_config)
        assert isinstance(cfg, SLMConfig)
        assert cfg.project.name == "test-project"
        assert cfg.project.language == "ko"
        assert cfg.teacher.backend == "ollama"

    def test_존재하지_않는_파일(self, tmp_path):
        """존재하지 않는 파일 경로가 FileNotFoundError를 발생시키는지 확인합니다."""
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nonexistent.yaml")


class TestCreateDefaultConfig:
    """create_default_config 함수의 테스트입니다."""

    def test_문자열_반환(self):
        """create_default_config가 비어있지 않은 문자열을 반환하는지 확인합니다."""
        result = create_default_config()
        assert isinstance(result, str)
        assert len(result) > 0
