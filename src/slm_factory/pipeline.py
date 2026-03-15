"""파이프라인 오케스트레이터 — 모든 모듈을 순차적으로 연결합니다.

각 단계는 독립적으로 또는 전체 파이프라인의 일부로 실행될 수 있습니다.
"""

from __future__ import annotations

import dataclasses
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import SLMConfig
    from .models import CompareResult, EvalResult, ParsedDocument, QAPair
    from .teacher.base import BaseTeacher

from .utils import get_logger, run_async

logger = get_logger("pipeline")


class Pipeline:
    """전체 slm-factory 파이프라인 오케스트레이터입니다."""

    def __init__(self, config: SLMConfig) -> None:
        self.config = config
        self.output_dir = Path(config.paths.output)

    # ------------------------------------------------------------------
    # 유틸리티 헬퍼
    # ------------------------------------------------------------------

    @staticmethod
    def _ollama_model_exists(model_name: str) -> bool:
        import subprocess

        try:
            result = subprocess.run(
                ["ollama", "show", model_name],
                capture_output=True,
                text=True,
                timeout=3,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def _save_pairs(self, pairs: list[QAPair], path: Path) -> None:
        """QA 쌍을 JSON 파일로 저장합니다."""
        data = [asdict(p) for p in pairs]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _load_pairs(self, path: Path) -> list[QAPair]:
        """JSON 파일에서 QA 쌍을 로드합니다."""
        from dataclasses import fields as dc_fields

        from .models import QAPair

        valid_fields = {f.name for f in dc_fields(QAPair)}
        data = json.loads(path.read_text(encoding="utf-8"))
        pairs: list[QAPair] = []
        for item in data:
            if "question" in item:
                filtered = {k: v for k, v in item.items() if k in valid_fields}
                pairs.append(QAPair(**filtered))
            elif "instruction" in item:
                pairs.append(
                    QAPair(
                        question=item.get("instruction", ""),
                        answer=item.get("output", ""),
                        instruction=item.get("instruction", ""),
                        source_doc=item.get("source_doc", ""),
                        category=item.get("category", ""),
                    )
                )
        return pairs

    # ------------------------------------------------------------------
    # 단계 1: 문서 파싱
    # ------------------------------------------------------------------

    def step_parse(
        self,
        files: list[Path] | None = None,
    ) -> list[ParsedDocument]:
        """설정된 디렉토리의 모든 문서를 파싱합니다.

        매개변수
        ----------
        files:
            파싱할 파일 목록입니다. 지정 시 디렉토리 스캔 대신
            이 목록의 파일만 파싱합니다.

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
            self.config.paths.documents,
            formats=extensions,
            files=files,
        )

        if not docs:
            raise RuntimeError(
                f"{self.config.paths.documents} 디렉토리에서 "
                f"{extensions} 형식의 문서를 찾을 수 없습니다. "
                "디렉토리가 존재하고 지원되는 파일이 있는지 확인하세요."
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

    def step_generate(
        self,
        docs: list[ParsedDocument],
    ) -> list[QAPair]:
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
        pairs = run_async(
            generator.generate_all_async(docs),
        )

        alpaca_path = self.output_dir / "qa_alpaca.json"
        generator.save_alpaca(pairs, alpaca_path)

        logger.info("Generated %d QA pairs", len(pairs))

        if not pairs:
            raise RuntimeError(
                "QA 쌍이 하나도 생성되지 않았습니다. "
                "문서 내용이 충분한지, Teacher LLM이 정상 응답하는지 확인하세요."
            )
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

    def step_score(
        self,
        pairs: list[QAPair],
        docs: list[ParsedDocument] | None = None,
    ) -> list[QAPair]:
        """교사 LLM을 사용하여 QA 쌍의 품질을 점수 평가하고 필터링합니다."""
        if not self.config.scoring.enabled:
            logger.info("Scoring disabled — skipping")
            return pairs

        from .teacher import create_teacher
        from .scorer import QualityScorer

        logger.info("Scoring %d QA pairs...", len(pairs))

        source_texts: dict[str, str] | None = None
        if docs:
            source_texts = {doc.doc_id: doc.content for doc in docs}

        teacher = create_teacher(self.config.teacher)
        scorer = QualityScorer(teacher, self.config.scoring, self.config.teacher)
        accepted, filtered = run_async(
            scorer.score_all(pairs, source_texts=source_texts)
        )

        if self.config.scoring.regenerate and filtered and docs:
            accepted = self._regenerate_low_quality(
                accepted,
                filtered,
                docs,
                teacher,
                scorer,
            )

        scored_path = self.output_dir / "qa_scored.json"
        self._save_pairs(accepted, scored_path)
        logger.info(
            "Scoring complete: %d accepted, %d filtered — saved to %s",
            len(accepted),
            len(filtered),
            scored_path,
        )
        return accepted

    def _regenerate_low_quality(
        self,
        accepted: list[QAPair],
        filtered: list[tuple[QAPair, int, str]],
        docs: list[ParsedDocument],
        teacher: BaseTeacher,
        scorer: QualityScorer,
    ) -> list[QAPair]:
        """낮은 점수의 QA 쌍을 비동기 배치로 재생성하여 품질을 높입니다."""
        from .teacher.qa_generator import QAGenerator

        generator = QAGenerator(self.config)
        doc_map = {doc.doc_id: doc for doc in docs}

        max_rounds = self.config.scoring.max_regenerate_rounds
        remaining = filtered

        for round_num in range(1, max_rounds + 1):
            if not remaining:
                break

            logger.info(
                "Regeneration round %d/%d: %d pairs to improve",
                round_num,
                max_rounds,
                len(remaining),
            )

            regen_pairs = run_async(
                self._regenerate_round(
                    remaining,
                    doc_map,
                    generator,
                )
            )

            if not regen_pairs:
                break

            new_accepted, new_filtered = run_async(scorer.score_all(regen_pairs))
            accepted.extend(new_accepted)
            remaining = new_filtered

            logger.info(
                "Regeneration round %d: %d recovered, %d still below threshold",
                round_num,
                len(new_accepted),
                len(new_filtered),
            )

        return accepted

    async def _regenerate_round(
        self,
        remaining: list[tuple[QAPair, int, str]],
        doc_map: dict[str, ParsedDocument],
        generator: object,
    ) -> list[QAPair]:
        """한 라운드의 재생성을 비동기 배치로 실행합니다."""
        import asyncio

        from rich.progress import (
            BarColumn,
            Progress,
            SpinnerColumn,
            TextColumn,
            TimeRemainingColumn,
        )

        from .utils import run_bounded

        semaphore = asyncio.Semaphore(1)

        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeRemainingColumn(),
        )

        items = []
        for pair, score, reason in remaining:
            doc = doc_map.get(pair.source_doc)
            if doc is not None:
                items.append((pair, score, reason, doc))

        if not items:
            return []

        results: list[QAPair] = []
        with progress:
            task_id = progress.add_task(
                "저품질 QA 재생성 중...",
                total=len(items),
            )
            tasks = [
                run_bounded(
                    semaphore,
                    self._regenerate_one(pair, score, reason, doc, generator),
                    progress,
                    task_id,
                )
                for pair, score, reason, doc in items
            ]
            gathered = await asyncio.gather(*tasks, return_exceptions=True)

        for result in gathered:
            if isinstance(result, BaseException):
                logger.error("재생성 실패: %s", result)
                continue
            if result is not None:
                results.append(result)

        return results

    async def _regenerate_one(
        self,
        pair: QAPair,
        score: int,
        reason: str,
        doc: ParsedDocument,
        generator: object,
    ) -> QAPair | None:
        """단일 QA 쌍을 비동기로 재생성합니다."""
        from .models import QAPair

        score_guidance = {
            1: "이전 답변이 완전히 잘못되었습니다. 문서 내용만을 근거로 정확한 답변을 작성하세요.",
            2: "이전 답변에 심각한 오류가 있었습니다. 오류를 수정하고 문서에 근거한 답변을 작성하세요.",
            3: "이전 답변이 불완전했습니다. 누락된 세부 정보를 보완하여 완전한 답변을 작성하세요.",
        }
        guidance = score_guidance.get(
            score,
            "이전 답변의 품질이 부족했습니다. 더 정확하고 완전한 답변을 작성하세요.",
        )
        enhanced_system = (
            f"{self.config.questions.system_prompt}\n\n"
            f"[재생성 지시] 이전 답변 점수: {score}/5점\n"
            f"평가 사유: {reason}\n"
            f"개선 방향: {guidance}"
        )

        chunk_content = self._find_relevant_chunk(doc, pair.question, pair.answer)

        prompt = generator.build_prompt(
            doc_title=doc.title,
            content=chunk_content,
            question=pair.question,
            tables=doc.tables if doc.tables else None,
            system_prompt=enhanced_system,
        )

        try:
            response = await generator.teacher.agenerate(
                prompt, format="json", think=False
            )
            parsed = generator.parse_response(response)
            if parsed:
                return QAPair(
                    question=pair.question,
                    answer=parsed["output"],
                    instruction=parsed["instruction"],
                    source_doc=pair.source_doc,
                    category=pair.category,
                )
        except Exception as e:
            logger.error(
                "Regeneration failed for '%s': %s",
                pair.question[:40],
                e,
            )
        return None

    def _find_relevant_chunk(
        self, doc: ParsedDocument, question: str, answer: str
    ) -> str:
        """질문/답변과 가장 관련 있는 문서 청크를 찾아 반환합니다."""
        from .calibration import auto_chunk_size, section_aware_chunk
        from .teacher.qa_generator import chunk_document

        max_context = self.config.teacher.max_context_chars
        if len(doc.content) <= max_context:
            return doc.content

        chunk_size = self.config.chunking.chunk_size
        if chunk_size == "auto":
            cs = auto_chunk_size(doc.content, max_context)
            chunks = section_aware_chunk(doc.content, cs)
        else:
            overlap = min(self.config.chunking.overlap_chars, chunk_size // 5)
            chunks = chunk_document(doc.content, chunk_size, overlap)

        if not chunks:
            return doc.content

        search_text = f"{question} {answer}".lower()
        best_chunk = chunks[0]
        best_score = 0
        for chunk in chunks:
            words = set(search_text.split())
            chunk_lower = chunk.lower()
            score = sum(1 for w in words if w in chunk_lower)
            if score > best_score:
                best_score = score
                best_chunk = chunk

        return best_chunk

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
        augmented = run_async(augmenter.augment_all(pairs))

        augmented_path = self.output_dir / "qa_augmented.json"
        self._save_pairs(augmented, augmented_path)
        logger.info(
            "Augmentation complete: %d → %d pairs — saved to %s",
            len(pairs),
            len(augmented),
            augmented_path,
        )
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

        num_epochs = self.config.training.num_epochs
        if num_epochs == "auto":
            from .calibration import auto_num_epochs

            num_examples = sum(1 for _ in open(training_data_path, encoding="utf-8"))
            num_epochs = auto_num_epochs(num_examples)
            logger.info(
                "Auto num_epochs: %d examples → %d epochs",
                num_examples,
                num_epochs,
            )
            self.config.training.num_epochs = num_epochs

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
            modelfile_path = ollama_exporter.export(model_dir)
            ollama_model = (
                self.config.export.ollama.model_name
                or self.config.student.model.split("/")[-1]
            )
            logger.info(
                "Ollama Modelfile: %s — 모델 등록 실패 시: ollama create %s -f %s",
                modelfile_path,
                ollama_model,
                modelfile_path,
            )

        logger.info("Export complete — model at %s", model_dir)
        return model_dir

    # ------------------------------------------------------------------
    # 단계 7: 자동 평가
    # ------------------------------------------------------------------

    def step_eval(
        self,
        pairs: list[QAPair],
        model_name: str,
    ) -> list[EvalResult]:
        """학습된 모델의 품질을 QA 쌍으로 자동 평가합니다.

        매개변수
        ----------
        pairs:
            평가에 사용할 QA 쌍입니다.
        model_name:
            평가 대상 Ollama 모델 이름입니다.

        반환값
        -------
        list[EvalResult]
            각 QA 쌍에 대한 평가 결과입니다.
        """
        from .evaluator import ModelEvaluator
        from .models import EvalResult

        if not self.config.eval.enabled:
            logger.info("Eval disabled — skipping")
            return []

        logger.info("Evaluating model '%s' with %d pairs...", model_name, len(pairs))

        evaluator = ModelEvaluator(self.config)
        results = evaluator.evaluate(pairs, model_name)

        eval_path = self.output_dir / self.config.eval.output_file
        evaluator.save_results(results, eval_path)
        evaluator.print_summary(results)

        if self.config.eval.quality_gate and results:
            passed, averages = evaluator.check_quality_gate(
                results,
                self.config.eval.quality_thresholds,
            )
            if not passed:
                logger.warning(
                    "⚠ 품질 게이트 미달 — 메트릭 평균: %s / 임계값: %s",
                    {k: f"{v:.4f}" for k, v in averages.items()},
                    self.config.eval.quality_thresholds,
                )

        logger.info(
            "Evaluation complete: %d results — saved to %s", len(results), eval_path
        )
        return results

    # ------------------------------------------------------------------
    # RAG 벡터 인덱싱
    # ------------------------------------------------------------------

    def step_rag_index(self, corpus_path: Path) -> Path:
        """corpus.parquet을 ChromaDB에 임베딩하여 적재합니다.

        매개변수
        ----------
        corpus_path:
            ``corpus.parquet`` 파일 경로입니다.

        반환값
        -------
        Path
            ChromaDB 저장 경로. 의존성 미설치 시 빈 ``Path()``를 반환합니다.
        """
        try:
            from .rag.indexer import RAGIndexer
        except ImportError:
            logger.warning(
                "RAG 의존성 미설치 — rag_index 건너뜀 "
                "(uv sync --extra rag --extra validation)"
            )
            return Path()

        if not corpus_path or not corpus_path.is_file():
            logger.warning(
                "corpus.parquet을 찾을 수 없음: %s — rag_index 건너뜀",
                corpus_path,
            )
            return Path()

        logger.info("ChromaDB 인덱싱 시작: %s", corpus_path)
        indexer = RAGIndexer(self.config)
        db_path = indexer.index(corpus_path)
        logger.info("ChromaDB 인덱싱 완료: %s", db_path)
        return db_path

    # ------------------------------------------------------------------
    # AutoRAG 데이터 내보내기
    # ------------------------------------------------------------------

    def step_autorag_export(
        self,
        parsed_docs: list[dict],
        qa_pairs: list[dict],
    ) -> tuple[Path, Path]:
        """파싱된 문서와 QA 쌍을 AutoRAG 평가용 parquet으로 변환합니다.

        매개변수
        ----------
        parsed_docs:
            :meth:`step_parse` 결과의 딕셔너리 리스트입니다.
        qa_pairs:
            QA 쌍 딕셔너리 리스트입니다.

        반환값
        -------
        tuple[Path, Path]
            ``(corpus.parquet 경로, qa.parquet 경로)``
        """
        from .exporter.autorag_export import AutoRAGExporter

        if not self.config.autorag_export.enabled:
            logger.info("AutoRAG export disabled — skipping")
            return Path(), Path()

        logger.info("Exporting data for AutoRAG evaluation...")

        exporter = AutoRAGExporter(self.config)
        corpus_path, qa_path = exporter.export(parsed_docs, qa_pairs)

        logger.info("AutoRAG export complete: %s, %s", corpus_path, qa_path)
        return corpus_path, qa_path

    # ------------------------------------------------------------------
    # Iterative Refinement
    # ------------------------------------------------------------------

    def step_refine(
        self,
        eval_results: list[EvalResult],
        docs: list[ParsedDocument],
        model_name: str,
    ) -> list[EvalResult]:
        """Student 약점을 분석하고 타겟 QA를 재생성하여 재학습합니다.

        매개변수
        ----------
        eval_results:
            :meth:`step_eval`에서 얻은 평가 결과입니다.
        docs:
            원본 파싱된 문서입니다.
        model_name:
            Ollama 모델 이름입니다.

        반환값
        -------
        list[EvalResult]
            최종 평가 결과입니다.
        """
        if not self.config.refinement.enabled:
            return eval_results

        from .teacher.qa_generator import QAGenerator

        threshold = self.config.refinement.llm_judge_threshold

        for round_num in range(1, self.config.refinement.max_rounds + 1):
            weak = [r for r in eval_results if r.scores.get("llm_judge", 0) < threshold]
            if not weak:
                logger.info("Refinement round %d: 약점 없음, 완료", round_num)
                break

            logger.info(
                "Refinement round %d: %d개 약점 타겟 QA 생성",
                round_num,
                len(weak),
            )

            generator = QAGenerator(self.config)
            refinement_pairs = run_async(generator.generate_refinement_qa(weak, docs))

            if not refinement_pairs:
                logger.info("Refinement round %d: 생성된 QA 없음, 중단", round_num)
                break

            refinement_pairs = self.step_validate(refinement_pairs, docs=docs)
            refinement_pairs = self.step_score(refinement_pairs, docs=docs)

            if not refinement_pairs:
                logger.info("Refinement round %d: 검증 통과 QA 없음, 중단", round_num)
                break

            refinement_pairs = self.step_augment(refinement_pairs)

            training_path = self.step_convert(refinement_pairs)
            adapter_path = self.step_train(training_path)
            self.step_export(adapter_path)

            eval_results = self.step_eval(refinement_pairs, model_name)

            logger.info("Refinement round %d 완료", round_num)

        return eval_results

    # ------------------------------------------------------------------
    # 단계 9: 모델 비교
    # ------------------------------------------------------------------

    def step_compare(self, pairs: list[QAPair]) -> list[CompareResult]:
        """Base 모델과 Fine-tuned 모델의 답변을 비교합니다.

        매개변수
        ----------
        pairs:
            비교에 사용할 QA 쌍입니다.

        반환값
        -------
        list[CompareResult]
            각 QA 쌍에 대한 비교 결과입니다.
        """
        from .comparator import ModelComparator
        from .models import CompareResult

        if not self.config.compare.enabled:
            logger.info("Model comparison disabled — skipping")
            return []

        logger.info("Comparing models...")

        comparator = ModelComparator(self.config)
        results = comparator.compare(pairs)

        compare_path = self.output_dir / self.config.compare.output_file
        comparator.save_results(results, compare_path)
        comparator.print_summary(results)

        logger.info(
            "Comparison complete: %d results — saved to %s", len(results), compare_path
        )
        return results

    # ------------------------------------------------------------------
    # 전체 파이프라인
    # ------------------------------------------------------------------

    def run(self) -> Path:
        """전체 파이프라인을 엔드-투-엔드로 실행합니다.

        단계: 파싱 → 생성 → 검증 → 변환 → 훈련 → 내보내기 → 평가 → RAG.

        반환값
        -------
        Path
            최종 내보낸 모델 디렉토리의 경로입니다.
        """
        start = time.time()

        try:
            logger.info(
                "프로젝트 '%s' 파이프라인을 시작합니다",
                self.config.project.name,
            )

            self.config.paths.ensure_dirs()

            logger.info("━━━ [1/13] 문서 파싱 ━━━")
            docs = self.step_parse()

            logger.info("━━━ [2/13] QA 쌍 생성 ━━━")
            pairs = self.step_generate(docs)

            logger.info("━━━ [3/13] QA 검증 ━━━")
            pairs = self.step_validate(pairs, docs=docs)

            logger.info("━━━ [4/13] 품질 점수 평가 ━━━")
            pairs = self.step_score(pairs, docs=docs)

            logger.info("━━━ [5/13] 데이터 증강 ━━━")
            pairs = self.step_augment(pairs)

            logger.info("━━━ [6/13] 데이터 분석 ━━━")
            self.step_analyze(pairs)

            logger.info("━━━ [7/13] 훈련 데이터 변환 ━━━")
            training_data_path = self.step_convert(pairs)

            logger.info("━━━ [8/13] LoRA 학습 ━━━")
            adapter_path = self.step_train(training_data_path)

            logger.info("━━━ [9/13] 모델 내보내기 ━━━")
            model_dir = self.step_export(adapter_path)

            logger.info("━━━ [10/13] 모델 평가 ━━━")
            ollama_cfg = self.config.export.ollama
            eval_results: list = []
            if ollama_cfg.enabled and ollama_cfg.model_name and pairs:
                if self._ollama_model_exists(ollama_cfg.model_name):
                    eval_results = self.step_eval(pairs, ollama_cfg.model_name)
                else:
                    logger.warning(
                        "Ollama 모델 '%s'을(를) 찾을 수 없어 평가를 건너뜁니다. "
                        "모델 등록 후 수동 평가: slm-factory eval model --model %s",
                        ollama_cfg.model_name,
                        ollama_cfg.model_name,
                    )

            logger.info("━━━ [11/13] Iterative Refinement ━━━")
            if self.config.refinement.enabled and eval_results:
                eval_results = self.step_refine(
                    eval_results, docs, ollama_cfg.model_name
                )

            logger.info("━━━ [12/13] RAG 데이터 내보내기 ━━━")
            doc_dicts = [asdict(d) if dataclasses.is_dataclass(d) else d for d in docs]
            pair_dicts = [
                asdict(p) if dataclasses.is_dataclass(p) else p for p in pairs
            ]
            corpus_path, _qa_path = self.step_autorag_export(
                doc_dicts,
                pair_dicts,
            )

            logger.info("━━━ [13/13] RAG 벡터 인덱싱 ━━━")
            self.step_rag_index(corpus_path)

            elapsed = time.time() - start
            logger.info(
                "파이프라인 완료 (%.1f초) — 모델 위치: %s",
                elapsed,
                model_dir,
            )
            return model_dir

        except Exception:
            elapsed = time.time() - start
            logger.exception("파이프라인이 %.1f초 후 실패했습니다", elapsed)
            raise
