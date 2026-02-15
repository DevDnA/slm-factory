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
from .utils import get_logger

logger = get_logger("evaluator")


def _load_bleu():
    import evaluate

    return evaluate.load("bleu")


def _load_rouge():
    import evaluate

    return evaluate.load("rouge")


class ModelEvaluator:
    """Ollama 모델을 QA 쌍으로 평가합니다."""

    def __init__(self, config: SLMConfig) -> None:
        self.config = config
        self.eval_config: EvalConfig = config.eval
        self.api_base = config.teacher.api_base.rstrip("/")
        self.timeout = config.teacher.timeout
        self.max_concurrency = config.teacher.max_concurrency

    async def _generate(self, client: httpx.AsyncClient, model_name: str, question: str) -> str:
        """Ollama API로 답변을 생성합니다."""
        resp = await client.post(
            f"{self.api_base}/api/generate",
            json={"model": model_name, "prompt": question, "stream": False},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json().get("response", "")

    def _compute_scores(self, reference: str, generated: str) -> dict[str, float]:
        """설정된 메트릭으로 점수를 계산합니다."""
        scores: dict[str, float] = {}
        metrics = self.eval_config.metrics

        if "bleu" in metrics:
            bleu = _load_bleu()
            result = bleu.compute(
                predictions=[generated], references=[[reference]],
            )
            scores["bleu"] = round(result["bleu"], 4)

        if "rouge" in metrics:
            rouge = _load_rouge()
            result = rouge.compute(
                predictions=[generated], references=[reference],
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

                async def _bounded(pair: QAPair) -> EvalResult:
                    async with semaphore:
                        result = await self._evaluate_one(client, model_name, pair)
                        progress.advance(task_id)
                        return result

                tasks = [_bounded(pair) for pair in qa_pairs]
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
        return asyncio.run(self._evaluate_async(samples, model_name))

    def save_results(self, results: list[EvalResult], path: Path) -> None:
        """평가 결과를 JSON 파일로 저장합니다."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = [asdict(r) for r in results]
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("평가 결과 저장: %s (%d건)", path, len(results))

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
