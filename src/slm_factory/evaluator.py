"""학습된 모델의 자동 평가 모듈입니다."""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn
from rich.table import Table

if TYPE_CHECKING:
    from .config import EvalConfig, SLMConfig

from .models import EvalResult, QAPair
from .utils import get_logger, ollama_generate, run_async, run_bounded

logger = get_logger("evaluator")


_bleu_metric = None
_rouge_metric = None
_korean_tokenizer = None
_korean_tokenizer_checked = False


def _load_bleu():
    global _bleu_metric
    if _bleu_metric is None:
        import evaluate
        _bleu_metric = evaluate.load("bleu")
    return _bleu_metric


def _load_rouge():
    global _rouge_metric
    if _rouge_metric is None:
        import evaluate
        _rouge_metric = evaluate.load("rouge")
    return _rouge_metric


def _load_korean_tokenizer():
    """한국어 형태소 분석 토크나이저를 로드합니다 (kiwipiepy 필요).

    ``kiwipiepy``가 설치되어 있으면 형태소 분석 함수를 반환하고,
    설치되지 않았으면 ``None``을 반환합니다.

    반환값
    -------
    callable | None
        텍스트를 받아 형태소 리스트를 반환하는 함수, 또는 ``None``.
    """
    global _korean_tokenizer, _korean_tokenizer_checked
    if _korean_tokenizer_checked:
        return _korean_tokenizer
    _korean_tokenizer_checked = True
    try:
        from kiwipiepy import Kiwi

        kiwi = Kiwi()

        def tokenize(text: str) -> list[str]:
            return [token.form for token in kiwi.tokenize(text)]

        _korean_tokenizer = tokenize
        logger.info("한국어 형태소 분석 토크나이저 로드됨 (kiwipiepy)")
    except ImportError:
        logger.debug("kiwipiepy 미설치 — 기본 공백 토크나이저 사용")
        _korean_tokenizer = None
    return _korean_tokenizer


def _preprocess_for_metrics(text: str, korean_tokenizer=None) -> str:
    """메트릭 계산을 위해 텍스트를 전처리합니다.

    한국어 토크나이저가 제공되면 형태소 단위로 분리하여
    공백으로 연결합니다. 이를 통해 BLEU/ROUGE가 형태소 단위로
    계산되어 한국어 평가의 정확도가 향상됩니다.

    매개변수
    ----------
    text:
        전처리할 텍스트.
    korean_tokenizer:
        형태소 분석 함수. ``None``이면 원본 텍스트를 반환합니다.

    반환값
    -------
    str
        전처리된 텍스트.
    """
    if korean_tokenizer is not None:
        return " ".join(korean_tokenizer(text))
    return text


class ModelEvaluator:
    """Ollama 모델을 QA 쌍으로 평가합니다."""

    def __init__(self, config: SLMConfig) -> None:
        self.config = config
        self.eval_config: EvalConfig = config.eval
        self.api_base = config.teacher.api_base.rstrip("/")
        self.timeout = config.teacher.timeout
        self.max_concurrency = config.teacher.max_concurrency

    async def _generate(self, client: httpx.AsyncClient, model_name: str, question: str) -> str:
        return await ollama_generate(client, self.api_base, model_name, question, self.timeout)

    def _compute_scores(self, reference: str, generated: str) -> dict[str, float]:
        """설정된 메트릭으로 점수를 계산합니다.

        한국어 프로젝트(``project.language == "ko"``)에서는 형태소 분석
        토크나이저로 텍스트를 전처리한 뒤 BLEU/ROUGE를 계산합니다.
        이를 통해 한국어의 교착어 특성이 반영된 정확한 점수를 얻습니다.
        """
        scores: dict[str, float] = {}
        metrics = self.eval_config.metrics

        # 한국어 프로젝트일 때 형태소 분석 토크나이저 사용
        tokenizer = None
        if self.config.project.language == "ko":
            tokenizer = _load_korean_tokenizer()

        ref = _preprocess_for_metrics(reference, tokenizer)
        gen = _preprocess_for_metrics(generated, tokenizer)

        if "bleu" in metrics:
            bleu = _load_bleu()
            result = bleu.compute(
                predictions=[gen], references=[[ref]],
            )
            scores["bleu"] = round(result["bleu"], 4)

        if "rouge" in metrics:
            rouge = _load_rouge()
            result = rouge.compute(
                predictions=[gen], references=[ref],
            )
            scores["rouge1"] = round(result["rouge1"], 4)
            scores["rouge2"] = round(result["rouge2"], 4)
            scores["rougeL"] = round(result["rougeL"], 4)

        return scores

    async def _evaluate_one(
        self,
        client: httpx.AsyncClient,
        model_name: str,
        pair: QAPair,
    ) -> EvalResult:
        generated = await self._generate(client, model_name, pair.question)
        scores = self._compute_scores(pair.answer, generated)
        return EvalResult(
            question=pair.question,
            reference_answer=pair.answer,
            generated_answer=generated,
            scores=scores,
        )

    async def _evaluate_async(
        self, qa_pairs: list[QAPair], model_name: str,
    ) -> list[EvalResult]:
        semaphore = asyncio.Semaphore(self.max_concurrency)
        results: list[EvalResult] = []

        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeRemainingColumn(),
        )

        async with httpx.AsyncClient() as client:
            with progress:
                task_id = progress.add_task("모델 평가 중...", total=len(qa_pairs))

                tasks = [
                    run_bounded(semaphore, self._evaluate_one(client, model_name, pair), progress, task_id)
                    for pair in qa_pairs
                ]
                gathered = await asyncio.gather(*tasks, return_exceptions=True)

        for item in gathered:
            if isinstance(item, BaseException):
                logger.error("평가 실패: %s", item)
                continue
            results.append(item)

        logger.info("평가 완료: %d/%d 성공", len(results), len(qa_pairs))
        return results

    def evaluate(self, qa_pairs: list[QAPair], model_name: str) -> list[EvalResult]:
        """QA 쌍으로 모델을 평가합니다."""
        if not qa_pairs:
            logger.warning("평가할 QA 쌍이 없습니다")
            return []

        samples = qa_pairs[: self.eval_config.max_samples]
        return run_async(self._evaluate_async(samples, model_name))

    def save_results(self, results: list[EvalResult], path: Path) -> None:
        """평가 결과를 JSON 파일로 저장합니다."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = [asdict(r) for r in results]
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("평가 결과 저장: %s (%d건)", path, len(results))

    def check_quality_gate(
        self,
        results: list[EvalResult],
        thresholds: dict[str, float],
    ) -> tuple[bool, dict[str, float]]:
        """평가 결과가 품질 임계값을 충족하는지 확인합니다."""
        if not results:
            return False, {}

        averages: dict[str, float] = {}
        all_keys: list[str] = []
        for r in results:
            for k in r.scores:
                if k not in all_keys:
                    all_keys.append(k)

        for key in all_keys:
            vals = [r.scores[key] for r in results if key in r.scores]
            if vals:
                averages[key] = sum(vals) / len(vals)

        passed = True
        for metric, threshold in thresholds.items():
            avg = averages.get(metric, 0.0)
            if avg < threshold:
                logger.warning(
                    "품질 게이트 실패: %s 평균 %.4f < 임계값 %.4f",
                    metric, avg, threshold,
                )
                passed = False

        if passed:
            logger.info("품질 게이트 통과: %s", {k: f"{v:.4f}" for k, v in averages.items()})

        return passed, averages

    def print_summary(self, results: list[EvalResult]) -> None:
        """평가 결과 요약을 출력합니다."""
        from rich.console import Console

        con = Console()

        if not results:
            con.print("[yellow]평가 결과가 없습니다[/yellow]")
            return

        all_keys: list[str] = []
        for r in results:
            for k in r.scores:
                if k not in all_keys:
                    all_keys.append(k)

        table = Table(title="평가 결과 요약")
        table.add_column("메트릭", style="cyan")
        table.add_column("평균", justify="right", style="bold")
        table.add_column("최소", justify="right")
        table.add_column("최대", justify="right")

        for key in all_keys:
            vals = [r.scores[key] for r in results if key in r.scores]
            if vals:
                table.add_row(
                    key,
                    f"{sum(vals) / len(vals):.4f}",
                    f"{min(vals):.4f}",
                    f"{max(vals):.4f}",
                )

        con.print(table)
        con.print(f"\n총 {len(results)}건 평가 완료")
