"""파이프라인 오케스트레이터 — 모든 모듈을 순차적으로 연결합니다.

각 단계는 독립적으로 또는 전체 파이프라인의 일부로 실행될 수 있습니다.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import SLMConfig
    from .models import ParsedDocument, QAPair

from .utils import get_logger

logger = get_logger("pipeline")


class Pipeline:
    """전체 slm-factory 파이프라인 오케스트레이터입니다."""

    def __init__(self, config: SLMConfig) -> None:
        self.config = config
        self.output_dir = Path(config.paths.output)

    # ------------------------------------------------------------------
    # 단계 1: 문서 파싱
    # ------------------------------------------------------------------

    def step_parse(self) -> list[ParsedDocument]:
        """설정된 디렉토리의 모든 문서를 파싱합니다.

        반환값
        -------
        list[ParsedDocument]
            텍스트, 표, 메타데이터가 포함된 파싱된 문서입니다.

        예외
        ------
        RuntimeError
            문서를 찾지 못했거나 파싱에 실패한 경우 발생합니다.
        """
        from .parsers import registry

        # 설정에서 확장자 목록 구성 (예: ["pdf"] → [".pdf"])
        extensions = [
            ext if ext.startswith(".") else f".{ext}"
            for ext in self.config.parsing.formats
        ]

        logger.info(
            "Parsing documents in %s (formats: %s)",
            self.config.paths.documents,
            extensions,
        )

        docs = registry.parse_directory(
            self.config.paths.documents, formats=extensions
        )

        if not docs:
            raise RuntimeError(
                f"No documents found in {self.config.paths.documents} "
                f"with formats {extensions}. "
                "Check that the directory exists and contains supported files."
            )

        # 파싱된 문서를 JSON으로 저장 (디버깅 및 재개용)
        debug_path = self.output_dir / "parsed_documents.json"
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        debug_data = [asdict(doc) for doc in docs]
        debug_path.write_text(
            json.dumps(debug_data, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        logger.info(
            "Parsed %d documents — saved debug output to %s",
            len(docs),
            debug_path,
        )

        return docs

    # ------------------------------------------------------------------
    # 단계 2: QA 쌍 생성
    # ------------------------------------------------------------------

    def step_generate(self, docs: list[ParsedDocument]) -> list[QAPair]:
        """교사 LLM을 사용하여 파싱된 문서에서 QA 쌍을 생성합니다.

        매개변수
        ----------
        docs:
            :meth:`step_parse`에서 얻은 파싱된 문서입니다.

        반환값
        -------
        list[QAPair]
            생성된 질문-답변 쌍입니다.
        """
        from .teacher.qa_generator import QAGenerator

        logger.info("Generating QA pairs from %d documents...", len(docs))

        generator = QAGenerator(self.config)
        pairs = asyncio.run(generator.generate_all_async(docs))

        # 중간 출력을 위해 Alpaca JSON 저장
        alpaca_path = self.output_dir / "qa_alpaca.json"
        generator.save_alpaca(pairs, alpaca_path)

        logger.info("Generated %d QA pairs", len(pairs))
        return pairs

    # ------------------------------------------------------------------
    # 단계 3: QA 쌍 검증
    # ------------------------------------------------------------------

    def step_validate(
        self,
        pairs: list[QAPair],
        docs: list[ParsedDocument] | None = None,
    ) -> list[QAPair]:
        """규칙 및 선택적 근거 검증을 사용하여 QA 쌍을 검증하고 필터링합니다.

        매개변수
        ----------
        pairs:
            :meth:`step_generate`에서 얻은 QA 쌍입니다.
        docs:
            근거 검증을 위한 선택적 파싱된 문서입니다.

        반환값
        -------
        list[QAPair]
            수락된 (검증된) QA 쌍입니다.
        """
        from .validator.rules import RuleValidator

        if not self.config.validation.enabled:
            logger.info("Validation disabled — returning all %d pairs", len(pairs))
            return pairs

        logger.info("Validating %d QA pairs...", len(pairs))

        validator = RuleValidator(self.config.validation)
        accepted, rejected = validator.validate_batch(pairs)

        logger.info(
            "Rule validation: %d accepted, %d rejected",
            len(accepted),
            len(rejected),
        )

        # 선택적 근거 검증
        if self.config.validation.groundedness.enabled and docs is not None:
            from .validator.similarity import GroundednessChecker

            logger.info("Running groundedness check...")

            checker = GroundednessChecker(self.config.validation.groundedness)

            # source_texts 딕셔너리 구성: doc_id → 내용
            source_texts = {doc.doc_id: doc.content for doc in docs}

            accepted, ungrounded = checker.check_batch(accepted, source_texts)

            logger.info(
                "Groundedness: %d grounded, %d ungrounded",
                len(accepted),
                len(ungrounded),
            )

        logger.info("Validation complete — %d pairs accepted", len(accepted))
        return accepted

    # ------------------------------------------------------------------
    # 단계 3a: 품질 점수 평가
    # ------------------------------------------------------------------

    def step_score(self, pairs: list[QAPair]) -> list[QAPair]:
        """교사 LLM을 사용하여 QA 쌍의 품질을 점수 평가하고 필터링합니다.

        매개변수
        ----------
        pairs:
            점수 평가할 QA 쌍입니다.

        반환값
        -------
        list[QAPair]
            threshold 이상의 점수를 받은 QA 쌍입니다.
        """
        if not self.config.scoring.enabled:
            logger.info("Scoring disabled — skipping")
            return pairs

        from .teacher import create_teacher
        from .scorer import QualityScorer

        logger.info("Scoring %d QA pairs...", len(pairs))

        teacher = create_teacher(self.config.teacher)
        scorer = QualityScorer(teacher, self.config.scoring, self.config.teacher)
        accepted, filtered = asyncio.run(scorer.score_all(pairs))

        logger.info(
            "Scoring complete: %d accepted, %d filtered",
            len(accepted), len(filtered),
        )
        return accepted

    # ------------------------------------------------------------------
    # 단계 3b: 데이터 증강
    # ------------------------------------------------------------------

    def step_augment(self, pairs: list[QAPair]) -> list[QAPair]:
        """교사 LLM을 사용하여 QA 쌍을 패러프레이즈 증강합니다.

        매개변수
        ----------
        pairs:
            증강할 QA 쌍입니다.

        반환값
        -------
        list[QAPair]
            원본 + 증강된 QA 쌍입니다.
        """
        if not self.config.augment.enabled:
            logger.info("Augmentation disabled — skipping")
            return pairs

        from .teacher import create_teacher
        from .augmenter import DataAugmenter

        logger.info("Augmenting %d QA pairs...", len(pairs))

        teacher = create_teacher(self.config.teacher)
        augmenter = DataAugmenter(teacher, self.config.augment, self.config.teacher)
        augmented = asyncio.run(augmenter.augment_all(pairs))

        logger.info("Augmentation complete: %d → %d pairs", len(pairs), len(augmented))
        return augmented

    # ------------------------------------------------------------------
    # 단계 3c: 데이터 분석
    # ------------------------------------------------------------------

    def step_analyze(self, pairs: list[QAPair]) -> None:
        """QA 쌍의 통계를 분석하고 보고서를 저장합니다.

        매개변수
        ----------
        pairs:
            분석할 QA 쌍입니다.
        """
        if not self.config.analyzer.enabled:
            logger.info("Analyzer disabled — skipping")
            return

        from .analyzer import DataAnalyzer

        logger.info("Analyzing %d QA pairs...", len(pairs))

        analyzer = DataAnalyzer()
        report = analyzer.analyze(pairs)

        report_path = self.output_dir / self.config.analyzer.output_file
        analyzer.save_report(report, report_path)
        analyzer.print_summary(report)

        logger.info("Analysis complete — report saved to %s", report_path)

    # ------------------------------------------------------------------
    # 단계 4: 훈련 형식으로 변환
    # ------------------------------------------------------------------

    def step_convert(self, pairs: list[QAPair]) -> Path:
        """QA 쌍을 채팅 템플릿 형식의 JSONL 훈련 데이터로 변환합니다.

        매개변수
        ----------
        pairs:
            :meth:`step_validate`에서 얻은 검증된 QA 쌍입니다.

        반환값
        -------
        Path
            생성된 JSONL 훈련 데이터 파일의 경로입니다.
        """
        from .converter import ChatFormatter

        logger.info("Converting %d QA pairs to training format...", len(pairs))

        formatter = ChatFormatter(self.config)
        jsonl_path = formatter.save_training_data(
            pairs, self.output_dir / "training_data.jsonl"
        )

        logger.info("Training data saved to %s", jsonl_path)
        return jsonl_path

    # ------------------------------------------------------------------
    # 단계 5: LoRA 어댑터 훈련
    # ------------------------------------------------------------------

    def step_train(self, training_data_path: Path) -> Path:
        """준비된 훈련 데이터에서 LoRA 어댑터를 훈련합니다.

        매개변수
        ----------
        training_data_path:
            :meth:`step_convert`에서 얻은 JSONL 훈련 데이터의 경로입니다.

        반환값
        -------
        Path
            저장된 LoRA 어댑터 디렉토리의 경로입니다.
        """
        from .trainer import DataLoader, LoRATrainer

        logger.info("Loading training data from %s", training_data_path)

        loader = DataLoader(self.config.training.train_split)
        dataset_dict = loader.load_and_split(training_data_path)

        logger.info("Starting LoRA training...")

        trainer = LoRATrainer(self.config)
        adapter_path = trainer.train(dataset_dict)

        logger.info("Training complete — adapter at %s", adapter_path)
        return adapter_path

    # ------------------------------------------------------------------
    # 단계 6: 모델 내보내기
    # ------------------------------------------------------------------

    def step_export(self, adapter_path: Path) -> Path:
        """훈련된 모델을 내보냅니다 (LoRA 병합 + 선택적 Ollama Modelfile).

        매개변수
        ----------
        adapter_path:
            :meth:`step_train`에서 얻은 LoRA 어댑터의 경로입니다.

        반환값
        -------
        Path
            병합/내보낸 모델 디렉토리의 경로입니다.
        """
        from .exporter import HFExporter, OllamaExporter

        logger.info("Exporting model from adapter at %s", adapter_path)

        hf_exporter = HFExporter(self.config)
        model_dir = hf_exporter.export(adapter_path)

        if self.config.export.ollama.enabled:
            logger.info("Generating Ollama Modelfile...")
            ollama_exporter = OllamaExporter(self.config)
            ollama_exporter.export(model_dir)

        logger.info("Export complete — model at %s", model_dir)
        return model_dir

    # ------------------------------------------------------------------
    # 전체 파이프라인
    # ------------------------------------------------------------------

    def run(self) -> Path:
        """전체 파이프라인을 엔드-투-엔드로 실행합니다.

        단계: 파싱 → 생성 → 검증 → 변환 → 훈련 → 내보내기.

        반환값
        -------
        Path
            최종 내보낸 모델 디렉토리의 경로입니다.
        """
        start = time.time()

        try:
            logger.info(
                "Starting slm-factory pipeline for project '%s'",
                self.config.project.name,
            )

            # 출력 디렉토리가 존재하는지 확인
            self.config.paths.ensure_dirs()

            # 단계 1: 파싱
            docs = self.step_parse()

            # 단계 2: QA 생성
            pairs = self.step_generate(docs)

            # 단계 3: 검증
            pairs = self.step_validate(pairs, docs=docs)

            # 단계 3a: 품질 점수 평가
            pairs = self.step_score(pairs)

            # 단계 3b: 데이터 증강
            pairs = self.step_augment(pairs)

            # 단계 3c: 데이터 분석
            self.step_analyze(pairs)

            # 단계 4: 변환
            training_data_path = self.step_convert(pairs)

            # 단계 5: 훈련
            adapter_path = self.step_train(training_data_path)

            # 단계 6: 내보내기
            model_dir = self.step_export(adapter_path)

            elapsed = time.time() - start
            logger.info(
                "Pipeline complete in %.1fs — model at %s",
                elapsed,
                model_dir,
            )
            return model_dir

        except Exception:
            elapsed = time.time() - start
            logger.exception("Pipeline failed after %.1fs", elapsed)
            raise
