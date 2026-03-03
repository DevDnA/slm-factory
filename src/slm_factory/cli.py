"""slm-factory용 명령줄 인터페이스입니다."""

from __future__ import annotations

import enum
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import typer
from rich.console import Console

from . import __version__

if TYPE_CHECKING:
    from .models import QAPair
    from .pipeline import Pipeline


class PipelineStep(str, enum.Enum):
    """run --until에 사용되는 파이프라인 단계입니다."""
    parse = "parse"
    generate = "generate"
    validate = "validate"
    score = "score"
    augment = "augment"
    analyze = "analyze"


app = typer.Typer(
    name="slm-factory",
    rich_markup_mode="rich",
)
console = Console()

eval_app = typer.Typer(help="모델 평가 및 비교")
tool_app = typer.Typer(help="유틸리티 도구 모음")

app.add_typer(eval_app, name="eval", rich_help_panel="📊 평가")
app.add_typer(tool_app, name="tool", rich_help_panel="🔧 도구")


@app.callback(invoke_without_command=True)
def main_callback(
    ctx: typer.Context,
    verbose: bool = typer.Option(False, "--verbose", "-v", help="디버그 로그를 표시합니다"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="경고와 에러만 표시합니다"),
) -> None:
    """Teacher-Student Knowledge Distillation framework for domain-specific SLMs."""
    if verbose:
        import logging

        logging.getLogger("slm_factory").setLevel(logging.DEBUG)
    elif quiet:
        import logging

        logging.getLogger("slm_factory").setLevel(logging.WARNING)
    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())


# ---------------------------------------------------------------------------
# 공통 옵션 상수
# ---------------------------------------------------------------------------

_CONFIG_HELP = "프로젝트 설정 파일 경로입니다. 현재 디렉토리부터 상위까지 자동 탐색합니다."
_RESUME_HELP = "중간 저장 파일에서 재개합니다"


# ---------------------------------------------------------------------------
# 헬퍼
# ---------------------------------------------------------------------------


def _get_error_hints(error: Exception) -> list[str]:
    """에러 유형에 따라 해결 힌트를 반환합니다."""
    error_str = str(error).lower()
    error_type = type(error).__name__.lower()

    if isinstance(error, FileNotFoundError):
        return ["설정 파일을 찾을 수 없습니다. `slm-factory init`으로 프로젝트를 생성하세요"]

    if (
        isinstance(error, ConnectionError)
        or "connect" in error_str
        or "connect" in error_type
        or "ollama" in error_str
        or "httpx" in error_type
    ):
        return [
            "Ollama가 실행 중인지 확인하세요: `ollama serve`",
            "모델이 다운로드되었는지 확인하세요: `ollama pull qwen3:8b`",
        ]

    if isinstance(error, RuntimeError) and (
        "no documents" in error_str or "no parseable" in error_str
    ):
        return ["documents 디렉토리에 문서(PDF, TXT 등)를 추가하세요"]

    if "cuda" in error_str or "mps" in error_str or "out of memory" in error_str or "oom" in error_str:
        return [
            "GPU 메모리가 부족합니다. 다음을 시도하세요:",
            "  training.batch_size를 줄이세요 (예: 2 또는 1)",
            "  lora.r 값을 줄이세요 (예: 8)",
            "  gradient_accumulation_steps를 늘리세요",
            "  Apple Silicon: PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 환경변수 설정",
        ]

    if "model" in error_str and ("not found" in error_str or "does not exist" in error_str):
        return [
            "모델을 찾을 수 없습니다:",
            "  Ollama 모델 확인: `ollama list`",
            "  모델 다운로드: `ollama pull <모델명>`",
            "  설정 파일의 teacher.model 또는 student.model을 확인하세요",
        ]

    if "permission" in error_str or "access denied" in error_str:
        return [
            "파일/디렉토리 접근 권한이 없습니다:",
            "  출력 디렉토리의 쓰기 권한을 확인하세요",
            "  `sudo` 없이 실행 중인지 확인하세요",
        ]

    if "disk" in error_str or "no space" in error_str:
        return [
            "디스크 공간이 부족합니다:",
            "  `df -h`로 디스크 사용량을 확인하세요",
            "  불필요한 체크포인트를 삭제하세요: `slm-factory clean`",
        ]

    return ["--verbose 옵션으로 상세 로그를 확인하세요"]


def _print_error(
    title: str,
    error: Exception | str,
    hints: list[str] | None = None,
    resume_cmd: str | None = None,
) -> None:
    """사용자 친화적 에러 메시지를 Rich Panel로 출력합니다."""
    from rich.panel import Panel

    msg = f"[red]✗[/red] {title}\n\n[dim]{error}[/dim]"
    if hints:
        hint_lines = "\n".join(f"  → {hint}" for hint in hints)
        msg += f"\n\n[yellow]해결 방법:[/yellow]\n{hint_lines}"
    if resume_cmd:
        msg += f"\n\n[blue]재개하려면:[/blue]\n  [cyan]{resume_cmd}[/cyan]"
    console.print(Panel(msg, title="[red]오류[/red]", border_style="red"))


def _find_config(config_path: str) -> str:
    p = Path(config_path)
    if p.is_file():
        return config_path

    if p.name == "project.yaml":
        for parent in [Path.cwd()] + list(Path.cwd().parents):
            candidate = parent / "project.yaml"
            if candidate.is_file():
                return str(candidate)
            if parent == Path.cwd().parent.parent:
                break

    return config_path


def _load_qa_data(
    pipeline: Pipeline,
    data: str | None,
    extra_candidates: list[str] | None = None,
) -> list[QAPair]:
    """QA 데이터를 로드합니다 (명시 경로 또는 출력 디렉토리 자동 감지).

    ``--data`` 옵션이 있으면 해당 파일을, 없으면 출력 디렉토리에서
    qa_augmented → qa_scored → qa_alpaca 순으로 탐색합니다.
    """
    from .models import QAPair

    if data is not None:
        data_path = Path(data)
        if not data_path.is_file():
            _print_error(
                "QA 데이터 파일 미발견",
                f"파일을 찾을 수 없음: {data_path}",
                ["--data 옵션의 경로를 확인하세요"],
            )
            raise typer.Exit(code=1)
        return pipeline._load_pairs(data_path)

    output_dir = pipeline.output_dir
    candidate_names = list(extra_candidates or []) + [
        "qa_augmented.json",
        "qa_scored.json",
        "qa_alpaca.json",
    ]
    for name in candidate_names:
        candidate = output_dir / name
        if candidate.is_file():
            console.print(f"[yellow]자동 감지:[/yellow] {candidate}")
            return pipeline._load_pairs(candidate)

    _print_error(
        "QA 데이터 파일 미발견",
        "출력 디렉토리에 QA 데이터 파일이 없습니다",
        ["generate 명령을 먼저 실행하거나 --data 옵션으로 경로를 지정하세요"],
    )
    raise typer.Exit(code=1)


def _load_pipeline(config_path: str) -> Pipeline:
    from pydantic import ValidationError

    from .config import load_config
    from .pipeline import Pipeline
    from .utils import setup_logging

    config_path = _find_config(config_path)
    setup_logging()

    try:
        config = load_config(config_path)
    except ValidationError as e:
        from rich.table import Table

        table = Table(title="설정 검증 오류", show_lines=True)
        table.add_column("위치", style="cyan")
        table.add_column("오류", style="red")
        table.add_column("입력값", style="yellow")
        for err in e.errors():
            loc = " → ".join(str(loc_part) for loc_part in err["loc"])
            table.add_row(loc, err["msg"], str(err.get("input", "")))
        console.print(table)
        console.print("\n[dim]ℹ 설정 파일을 확인하세요[/dim]")
        raise typer.Exit(code=1)

    return Pipeline(config)


def _try_load_parsed_docs(pipeline: Pipeline) -> list | None:
    """parsed_documents.json이 존재하면 ParsedDocument 리스트를 반환합니다.

    Resume 시 groundedness 검증을 위해 파싱된 문서를 재로드합니다.
    파일이 없으면 None을 반환하며, step_validate는 docs=None을 정상 처리합니다.
    """
    from .models import ParsedDocument

    parsed_path = pipeline.output_dir / "parsed_documents.json"
    if not parsed_path.is_file():
        return None
    try:
        raw = json.loads(parsed_path.read_text(encoding="utf-8"))
        return [ParsedDocument(**item) for item in raw]
    except (json.JSONDecodeError, TypeError, KeyError):
        import logging
        logging.getLogger("slm_factory.cli").warning("파싱된 문서 파일 로드 실패: %s", parsed_path)
        return None


def _find_resume_point(pipeline: Pipeline) -> tuple[str, list]:
    """중간 저장 파일에서 가장 최근의 재개 지점을 탐색합니다."""
    from .models import ParsedDocument, QAPair

    output_dir = pipeline.output_dir

    augmented = output_dir / "qa_augmented.json"
    if augmented.is_file():
        pairs = pipeline._load_pairs(augmented)
        console.print(
            f"[yellow]재개 지점:[/yellow] qa_augmented.json ({len(pairs)}개 쌍)"
            " → analyze 단계부터"
        )
        return "analyze", pairs

    scored = output_dir / "qa_scored.json"
    if scored.is_file():
        pairs = pipeline._load_pairs(scored)
        console.print(
            f"[yellow]재개 지점:[/yellow] qa_scored.json ({len(pairs)}개 쌍)"
            " → augment 단계부터"
        )
        return "augment", pairs

    alpaca = output_dir / "qa_alpaca.json"
    if alpaca.is_file():
        raw = json.loads(alpaca.read_text(encoding="utf-8"))
        if raw and "question" in raw[0]:
            pairs = [QAPair(**item) for item in raw]
        elif raw and "instruction" in raw[0]:
            pairs = [
                QAPair(
                    question=item.get("instruction", ""),
                    answer=item.get("output", ""),
                    instruction=item.get("instruction", ""),
                    source_doc=item.get("source_doc", ""),
                    category=item.get("category", ""),
                )
                for item in raw
            ]
        else:
            pairs = []
        if pairs:
            console.print(
                f"[yellow]재개 지점:[/yellow] qa_alpaca.json ({len(pairs)}개 쌍)"
                " → validate 단계부터"
            )
            return "validate", pairs

    parsed = output_dir / "parsed_documents.json"
    if parsed.is_file():
        raw = json.loads(parsed.read_text(encoding="utf-8"))
        docs = [ParsedDocument(**item) for item in raw]
        if docs:
            console.print(
                f"[yellow]재개 지점:[/yellow] parsed_documents.json ({len(docs)}개 문서)"
                " → generate 단계부터"
            )
            return "generate", docs

    console.print("[yellow]재개 지점 없음 — 처음부터 실행합니다[/yellow]")
    return "start", []


# ---------------------------------------------------------------------------
# 명령어
# ---------------------------------------------------------------------------


@app.command(rich_help_panel="🚀 시작하기")
def init(
    name: str = typer.Argument(help="프로젝트 이름"),
    path: str = typer.Option(".", "--path", help="프로젝트를 생성할 상위 디렉토리입니다"),
) -> None:
    """새로운 slm-factory 프로젝트를 초기화합니다."""
    from .config import create_default_config

    from rich.prompt import Confirm

    project_dir = Path(path) / name
    documents_dir = project_dir / "documents"
    output_dir = project_dir / "output"
    config_path = project_dir / "project.yaml"

    if config_path.is_file():
        console.print(f"\n[yellow]⚠[/yellow] 설정 파일이 이미 존재합니다: [cyan]{config_path}[/cyan]")
        if not Confirm.ask("  덮어쓰시겠습니까?", default=False):
            console.print("  [dim]취소됨[/dim]")
            raise typer.Exit(code=0)

    project_dir.mkdir(parents=True, exist_ok=True)
    documents_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    template = create_default_config()
    config_content = template.replace('name: "my-project"', f'name: "{name}"')
    config_content = config_content.replace(
        'model_name: "my-project-model"', f'model_name: "{name}-model"'
    )

    config_path.write_text(config_content, encoding="utf-8")

    console.print(f"\n[green]✓[/green] 프로젝트 '{name}'가 생성되었습니다: [cyan]{project_dir}[/cyan]\n")
    console.print("프로젝트 구조:")
    console.print(f"  {project_dir}/")
    console.print(f"  {documents_dir}/")
    console.print(f"  {output_dir}/")
    console.print(f"  {config_path}")
    console.print(f"\n[bold]사전 준비:[/bold]")
    console.print(f"  1. [cyan]{documents_dir}[/cyan] 디렉토리에 학습할 문서(PDF, TXT 등)를 추가하세요")
    console.print(f"  2. 별도 터미널에서 Ollama를 실행하세요: [cyan]ollama serve[/cyan]")
    console.print(f"  3. Teacher 모델을 다운로드하세요: [cyan]ollama pull qwen3:8b[/cyan]")
    console.print(f"\n[bold]실행:[/bold]")
    console.print(f"  4. 환경 점검: [cyan]slm-factory check --config {config_path}[/cyan]")
    console.print(f"  5. wizard 실행: [cyan]slm-factory tool wizard --config {config_path}[/cyan]\n")


_STEP_ORDER = ["parse", "generate", "validate", "score", "augment", "analyze"]

_RESUME_TO_STEP_IDX: dict[str, int] = {
    "generate": 1,
    "validate": 2,
    "augment": 4,
    "analyze": 5,
}


def _run_until_step(pipeline: Pipeline, target: str, resume: bool) -> None:
    target_idx = _STEP_ORDER.index(target)

    docs = None
    pairs: list[QAPair] = []
    start_idx = 0

    if resume:
        resume_step, resume_data = _find_resume_point(pipeline)
        if resume_step != "start":
            start_idx = _RESUME_TO_STEP_IDX.get(resume_step, 0)
            if resume_step == "generate":
                docs = resume_data
            else:
                pairs = resume_data

    if start_idx > target_idx:
        console.print(
            f"\n[bold green]{target} 단계는 이미 완료되었습니다[/bold green]\n"
        )
        return

    for idx in range(start_idx, target_idx + 1):
        step = _STEP_ORDER[idx]

        if step == "parse":
            docs = pipeline.step_parse()
            console.print(f"  [green]✓[/green] {len(docs)}개 문서 파싱 완료")

        elif step == "generate":
            if docs is None:
                docs = pipeline.step_parse()
            pairs = pipeline.step_generate(docs)
            console.print(f"  [green]✓[/green] {len(pairs)}개 QA 쌍 생성 완료")

        elif step == "validate":
            before = len(pairs)
            pairs = pipeline.step_validate(pairs, docs=docs) if docs else pipeline.step_validate(pairs)
            console.print(
                f"  [green]✓[/green] 검증 완료: {len(pairs)}개 수락, {before - len(pairs)}개 거부"
            )

        elif step == "score":
            before = len(pairs)
            pairs = pipeline.step_score(pairs)
            console.print(
                f"  [green]✓[/green] 점수 평가: {len(pairs)}개 통과, {before - len(pairs)}개 제거"
            )

        elif step == "augment":
            before = len(pairs)
            pairs = pipeline.step_augment(pairs)
            console.print(f"  [green]✓[/green] 데이터 증강: {before}개 → {len(pairs)}개")

        elif step == "analyze":
            pipeline.step_analyze(pairs)
            console.print(f"  [green]✓[/green] {len(pairs)}개 QA 쌍 분석 완료")


@app.command(rich_help_panel="⚙️ 파이프라인")
def run(
    config: str = typer.Option("project.yaml", "--config", help=_CONFIG_HELP),
    resume: bool = typer.Option(
        False, "--resume", "-r", help=_RESUME_HELP
    ),
    until: Optional[PipelineStep] = typer.Option(
        None, "--until", help="지정된 단계까지만 실행 (parse|generate|validate|score|augment|analyze)"
    ),
) -> None:
    """파이프라인을 실행합니다. --until로 단계를 지정하면 해당 단계까지만 실행합니다."""
    try:
        pipeline = _load_pipeline(config)
        pipeline.config.paths.ensure_dirs()

        if until is not None:
            console.print(
                f"\n[bold blue]slm-factory[/bold blue] — {until.value} 단계까지 실행 중...\n"
            )
            _run_until_step(pipeline, until.value, resume)
            console.print(
                f"\n[bold green]{until.value} 단계까지 완료![/bold green]\n"
            )
            return

        console.print(
            "\n[bold blue]slm-factory[/bold blue] — 전체 파이프라인 시작 중...\n"
        )

        if not resume:
            model_dir = pipeline.run()
        else:
            step, data = _find_resume_point(pipeline)

            if step == "start":
                model_dir = pipeline.run()
            else:
                if step == "generate":
                    docs = data
                    pairs = pipeline.step_generate(docs)
                    pairs = pipeline.step_validate(pairs, docs=docs)
                    pairs = pipeline.step_score(pairs)
                    pairs = pipeline.step_augment(pairs)
                elif step == "validate":
                    # groundedness 검증을 위해 파싱된 문서 로드 시도
                    docs = _try_load_parsed_docs(pipeline)
                    pairs = pipeline.step_validate(data, docs=docs)
                    pairs = pipeline.step_score(pairs)
                    pairs = pipeline.step_augment(pairs)
                elif step == "augment":
                    pairs = pipeline.step_augment(data)
                else:
                    pairs = data

                pipeline.step_analyze(pairs)
                training_data_path = pipeline.step_convert(pairs)
                adapter_path = pipeline.step_train(training_data_path)
                model_dir = pipeline.step_export(adapter_path)

        console.print(
            f"\n[bold green]파이프라인 완료![/bold green] 모델 저장 위치: [cyan]{model_dir}[/cyan]\n"
        )

    except FileNotFoundError as e:
        _print_error("설정 파일 오류", e, hints=_get_error_hints(e))
        raise typer.Exit(code=1)
    except Exception as e:
        _print_error("파이프라인 실패", e, hints=_get_error_hints(e))
        raise typer.Exit(code=1)


@app.command(rich_help_panel="⚙️ 파이프라인")
def train(
    config: str = typer.Option("project.yaml", "--config", help=_CONFIG_HELP),
    data: Optional[str] = typer.Option(
        None, "--data", help="사전 생성된 training_data.jsonl 파일 경로입니다"
    ),
    resume: bool = typer.Option(
        False, "--resume", "-r", help=_RESUME_HELP
    ),
) -> None:
    """훈련 단계를 실행합니다 (선택적으로 사전 생성된 데이터 사용)."""
    try:
        pipeline = _load_pipeline(config)
        pipeline.config.paths.ensure_dirs()

        if data is not None:
            training_data_path = Path(data)
            if not training_data_path.is_file():
                _print_error("훈련 데이터 미발견", f"파일을 찾을 수 없음: {training_data_path}", ["--data 옵션의 경로를 확인하세요"])
                raise typer.Exit(code=1)
        elif resume:
            step, loaded = _find_resume_point(pipeline)
            if step == "generate":
                docs = loaded
                pairs = pipeline.step_generate(docs)
                pairs = pipeline.step_validate(pairs, docs=docs)
                pairs = pipeline.step_score(pairs)
                pairs = pipeline.step_augment(pairs)
            elif step == "validate":
                docs = _try_load_parsed_docs(pipeline)
                pairs = pipeline.step_validate(loaded, docs=docs)
                pairs = pipeline.step_score(pairs)
                pairs = pipeline.step_augment(pairs)
            elif step == "augment":
                pairs = pipeline.step_augment(loaded)
            elif step == "analyze":
                pairs = loaded
            else:
                docs = pipeline.step_parse()
                pairs = pipeline.step_generate(docs)
                pairs = pipeline.step_validate(pairs, docs=docs)
                pairs = pipeline.step_score(pairs)
                pairs = pipeline.step_augment(pairs)
            pipeline.step_analyze(pairs)
            training_data_path = pipeline.step_convert(pairs)
        else:
            docs = pipeline.step_parse()
            pairs = pipeline.step_generate(docs)
            pairs = pipeline.step_validate(pairs, docs=docs)
            pairs = pipeline.step_score(pairs)
            pairs = pipeline.step_augment(pairs)
            pipeline.step_analyze(pairs)
            training_data_path = pipeline.step_convert(pairs)

        adapter_path = pipeline.step_train(training_data_path)

        console.print(
            f"\n[bold green]훈련 완료![/bold green] "
            f"어댑터 저장 위치: [cyan]{adapter_path}[/cyan]\n"
        )

    except FileNotFoundError as e:
        _print_error("설정 파일 오류", e, hints=_get_error_hints(e))
        raise typer.Exit(code=1)
    except Exception as e:
        _print_error("훈련 실패", e, hints=_get_error_hints(e))
        raise typer.Exit(code=1)


@app.command(rich_help_panel="🚀 시작하기")
def check(
    config: str = typer.Option("project.yaml", "--config", help=_CONFIG_HELP),
) -> None:
    """프로젝트 설정과 환경을 사전 점검합니다."""
    from rich.table import Table

    table = Table(title="slm-factory 환경 점검")
    table.add_column("항목", style="cyan")
    table.add_column("상태", style="bold")
    table.add_column("상세", style="dim")

    all_ok = True

    try:
        from .config import load_config

        resolved = _find_config(config)
        cfg = load_config(resolved)
        table.add_row("설정 파일", "[green]OK[/green]", str(resolved))
    except Exception as e:
        cfg = None
        table.add_row("설정 파일", "[red]FAIL[/red]", str(e))
        all_ok = False

    if cfg is None:
        console.print()
        console.print(table)
        raise typer.Exit(code=1)

    doc_dir = Path(cfg.paths.documents)
    if doc_dir.is_dir():
        doc_files = [f for f in doc_dir.iterdir() if f.is_file()]
        if doc_files:
            table.add_row(
                "문서 디렉토리",
                "[green]OK[/green]",
                f"{len(doc_files)}개 파일 ({doc_dir})",
            )
        else:
            table.add_row(
                "문서 디렉토리",
                "[yellow]WARN[/yellow]",
                f"디렉토리는 있으나 파일 없음 ({doc_dir})",
            )
            all_ok = False
    else:
        table.add_row(
            "문서 디렉토리", "[red]FAIL[/red]", f"디렉토리 없음: {doc_dir}"
        )
        all_ok = False

    out_dir = Path(cfg.paths.output)
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        probe = out_dir / ".check_write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        table.add_row(
            "출력 디렉토리", "[green]OK[/green]", f"쓰기 가능 ({out_dir})"
        )
    except Exception as e:
        table.add_row("출력 디렉토리", "[red]FAIL[/red]", str(e))
        all_ok = False

    if cfg.teacher.backend == "ollama":
        import httpx

        api_base = cfg.teacher.api_base.rstrip("/")
        try:
            resp = httpx.get(f"{api_base}/api/version", timeout=5)
            if resp.status_code == 200:
                ver = resp.json().get("version", "unknown")
                table.add_row(
                    "Ollama 연결",
                    "[green]OK[/green]",
                    f"v{ver} ({api_base})",
                )
            else:
                table.add_row(
                    "Ollama 연결",
                    "[red]FAIL[/red]",
                    f"HTTP {resp.status_code}",
                )
                all_ok = False
        except Exception as e:
            table.add_row("Ollama 연결", "[red]FAIL[/red]", f"연결 불가: {e}")
            all_ok = False

        try:
            resp = httpx.get(f"{api_base}/api/tags", timeout=5)
            if resp.status_code == 200:
                models = [m["name"] for m in resp.json().get("models", [])]
                target = cfg.teacher.model
                if any(target in m for m in models):
                    table.add_row("모델 사용 가능", "[green]OK[/green]", target)
                else:
                    available = ", ".join(models[:5]) if models else "없음"
                    table.add_row(
                        "모델 사용 가능",
                        "[yellow]WARN[/yellow]",
                        f"'{target}' 미발견 (사용 가능: {available})",
                    )
                    all_ok = False
            else:
                table.add_row(
                    "모델 사용 가능",
                    "[red]FAIL[/red]",
                    f"모델 목록 조회 실패 (HTTP {resp.status_code})",
                )
                all_ok = False
        except Exception:
            table.add_row(
                "모델 사용 가능",
                "[yellow]WARN[/yellow]",
                "모델 목록 조회 불가 (Ollama 연결 필요)",
            )
    else:
        table.add_row(
            "LLM 백엔드",
            "[green]OK[/green]",
            f"{cfg.teacher.backend} ({cfg.teacher.api_base})",
        )

    # ── 컴퓨팅 디바이스 감지 ─────────────────────────────────────
    try:
        from .device import detect_device

        device = detect_device()

        type_label = {
            "cuda": "[green]NVIDIA GPU (CUDA)[/green]",
            "mps": "[green]Apple Silicon GPU (MPS)[/green]",
            "cpu": "[yellow]CPU[/yellow]",
        }
        table.add_row(
            "컴퓨팅 디바이스",
            type_label.get(device.type, device.type),
            device.name,
        )

        precision = {
            "bfloat16": "bfloat16 (bf16)",
            "float16": "float16 (fp16)",
            "float32": "float32",
        }
        table.add_row(
            "학습 정밀도",
            "[green]OK[/green]" if device.is_gpu else "[yellow]WARN[/yellow]",
            precision.get(device.dtype_name, device.dtype_name),
        )

        if device.type == "cuda":
            bnb_status = "[green]OK[/green]" if device.quantization_available else "[yellow]WARN[/yellow]"
            bnb_detail = "사용 가능" if device.quantization_available else "미설치 (pip install bitsandbytes)"
            table.add_row("4bit 양자화", bnb_status, bnb_detail)
        elif device.type == "mps":
            table.add_row(
                "4bit 양자화",
                "[dim]N/A[/dim]",
                "Apple Silicon — Unified Memory로 대체",
            )

        if not device.is_gpu:
            all_ok = False
    except Exception:
        table.add_row(
            "컴퓨팅 디바이스",
            "[yellow]WARN[/yellow]",
            "PyTorch 미설치 (학습 시 필요)",
        )

    console.print()
    console.print(table)
    if all_ok:
        console.print("\n[bold green]모든 점검 통과![/bold green]")
        console.print(f"  wizard 실행: [cyan]slm-factory tool wizard --config {resolved}[/cyan]\n")
    else:
        console.print("\n[bold yellow]일부 항목에 주의가 필요합니다.[/bold yellow]")
        console.print("\n[dim]일반적인 해결 방법:[/dim]")
        console.print("  문서 추가  → documents/ 디렉토리에 PDF, TXT 등을 넣으세요")
        console.print("  Ollama 실행 → 별도 터미널에서 [cyan]ollama serve[/cyan]")
        console.print(f"  모델 다운로드 → [cyan]ollama pull {cfg.teacher.model}[/cyan]\n")
        raise typer.Exit(code=1)


@app.command(rich_help_panel="ℹ️ 정보")
def status(
    config: str = typer.Option("project.yaml", "--config", help=_CONFIG_HELP),
) -> None:
    """파이프라인 진행 상태를 확인합니다."""
    from rich.table import Table

    from .config import load_config

    try:
        cfg = load_config(_find_config(config))
    except Exception as e:
        _print_error("설정 로드 실패", e, hints=_get_error_hints(e))
        raise typer.Exit(code=1)

    output_dir = Path(cfg.paths.output)

    stages: list[tuple[str, str, str]] = [
        ("parse", "parsed_documents.json", "문서"),
        ("generate", "qa_alpaca.json", "쌍"),
        ("score", "qa_scored.json", "쌍"),
        ("augment", "qa_augmented.json", "쌍"),
        ("analyze", "data_analysis.json", "항목"),
        ("convert", "training_data.jsonl", "줄"),
        ("train", "checkpoints/adapter/", ""),
        ("export", "merged_model/", ""),
    ]

    table = Table(title="파이프라인 진행 상태")
    table.add_column("단계", style="cyan")
    table.add_column("파일", style="dim")
    table.add_column("상태", style="bold")
    table.add_column("건수")

    for stage_name, filename, unit in stages:
        filepath = output_dir / filename.rstrip("/")
        if filename.endswith("/"):
            if filepath.is_dir():
                table.add_row(stage_name, filename, "[green]존재[/green]", "디렉토리")
            else:
                table.add_row(stage_name, filename, "[red]없음[/red]", "-")
        elif filename.endswith(".jsonl"):
            if filepath.is_file():
                with filepath.open(encoding="utf-8") as fh:
                    line_count = sum(1 for _ in fh)
                table.add_row(
                    stage_name, filename, "[green]존재[/green]",
                    f"{line_count}개 {unit}",
                )
            else:
                table.add_row(stage_name, filename, "[red]없음[/red]", "-")
        elif filepath.is_file():
            try:
                data = json.loads(filepath.read_text(encoding="utf-8"))
                count = len(data) if isinstance(data, list) else 1
                table.add_row(
                    stage_name, filename, "[green]존재[/green]",
                    f"{count}개 {unit}",
                )
            except Exception:
                table.add_row(stage_name, filename, "[green]존재[/green]", "?")
        else:
            table.add_row(stage_name, filename, "[red]없음[/red]", "-")

    console.print()
    console.print(table)

    if (output_dir / "qa_augmented.json").is_file():
        resume_stage = "analyze"
    elif (output_dir / "qa_scored.json").is_file():
        resume_stage = "augment"
    elif (output_dir / "qa_alpaca.json").is_file():
        resume_stage = "validate"
    elif (output_dir / "parsed_documents.json").is_file():
        resume_stage = "generate"
    else:
        resume_stage = "parse"

    all_complete = all(
        (output_dir / fn.rstrip("/")).exists() for _, fn, _ in stages
    )
    if all_complete:
        console.print("\n[bold green]모든 단계가 완료되었습니다[/bold green]\n")
    else:
        console.print(
            f"\n다음 [cyan]--resume[/cyan] 실행 시 "
            f"[bold]{resume_stage}[/bold]부터 재개됩니다\n"
        )


@app.command(rich_help_panel="ℹ️ 정보")
def clean(
    config: str = typer.Option("project.yaml", "--config", help=_CONFIG_HELP),
    all_files: bool = typer.Option(False, "--all", help="모든 출력 파일을 삭제합니다"),
) -> None:
    """중간 생성 파일을 정리합니다."""
    import shutil

    from rich.table import Table

    from .config import load_config

    try:
        cfg = load_config(_find_config(config))
    except Exception as e:
        _print_error("설정 로드 실패", e, hints=_get_error_hints(e))
        raise typer.Exit(code=1)

    output_dir = Path(cfg.paths.output)

    if all_files:
        targets = list(output_dir.iterdir()) if output_dir.is_dir() else []
    else:
        intermediate_names = [
            "qa_scored.json",
            "qa_augmented.json",
            "data_analysis.json",
        ]
        targets = [
            output_dir / name
            for name in intermediate_names
            if (output_dir / name).exists()
        ]

    if not targets:
        console.print("\n삭제할 파일이 없습니다.\n")
        return

    console.print("\n[bold]삭제 대상:[/bold]")
    for t in targets:
        console.print(f"  {t}")

    typer.confirm("\n삭제하시겠습니까?", abort=True)

    deleted: list[Path] = []
    for t in targets:
        if t.is_dir():
            shutil.rmtree(t)
            deleted.append(t)
        elif t.is_file():
            t.unlink()
            deleted.append(t)

    table = Table(title="삭제 결과")
    table.add_column("파일", style="cyan")
    table.add_column("상태", style="bold")
    for d in deleted:
        table.add_row(str(d), "[green]삭제됨[/green]")

    console.print()
    console.print(table)
    console.print(f"\n[bold green]{len(deleted)}개 항목 삭제 완료[/bold green]\n")


@tool_app.command(name="convert")
def convert(
    config: str = typer.Option("project.yaml", "--config", help=_CONFIG_HELP),
    data: Optional[str] = typer.Option(
        None, "--data", help="QA 데이터 파일 경로 (qa_alpaca.json 또는 qa_augmented.json)"
    ),
) -> None:
    """QA 데이터를 훈련용 JSONL 형식으로 변환합니다."""
    try:
        pipeline = _load_pipeline(config)
        pipeline.config.paths.ensure_dirs()

        pairs = _load_qa_data(pipeline, data)

        training_data_path = pipeline.step_convert(pairs)

        console.print(
            f"\n[bold green]변환 완료![/bold green] "
            f"훈련 데이터: [cyan]{training_data_path}[/cyan] ({len(pairs)}개 쌍)\n"
        )

    except FileNotFoundError as e:
        _print_error("설정 파일 오류", e, hints=_get_error_hints(e))
        raise typer.Exit(code=1)
    except Exception as e:
        _print_error("변환 실패", e, hints=_get_error_hints(e))
        raise typer.Exit(code=1)


@app.command(name="export", rich_help_panel="⚙️ 파이프라인")
def export_model(
    config: str = typer.Option("project.yaml", "--config", help=_CONFIG_HELP),
    adapter: Optional[str] = typer.Option(
        None, "--adapter", help="어댑터 디렉토리 경로"
    ),
) -> None:
    """훈련된 모델을 내보냅니다 (LoRA 병합 + Ollama Modelfile)."""
    try:
        pipeline = _load_pipeline(config)
        pipeline.config.paths.ensure_dirs()

        if adapter is not None:
            adapter_path = Path(adapter)
            if not adapter_path.is_dir():
                _print_error("어댑터 디렉토리 미발견", f"디렉토리를 찾을 수 없음: {adapter_path}", ["--adapter 옵션의 경로를 확인하세요"])
                raise typer.Exit(code=1)
        else:
            adapter_path = pipeline.output_dir / "checkpoints" / "adapter"
            if not adapter_path.is_dir():
                _print_error("어댑터 디렉토리 미발견", f"디렉토리를 찾을 수 없음: {adapter_path}", ["--adapter 옵션으로 경로를 지정하거나 train 명령을 먼저 실행하세요"])
                raise typer.Exit(code=1)

        model_dir = pipeline.step_export(adapter_path)

        console.print(
            f"\n[bold green]내보내기 완료![/bold green] "
            f"모델 저장 위치: [cyan]{model_dir}[/cyan]\n"
        )

    except FileNotFoundError as e:
        _print_error("설정 파일 오류", e, hints=_get_error_hints(e))
        raise typer.Exit(code=1)
    except Exception as e:
        _print_error("내보내기 실패", e, hints=_get_error_hints(e))
        raise typer.Exit(code=1)


@app.command(rich_help_panel="ℹ️ 정보")
def version() -> None:
    """slm-factory 버전을 표시합니다."""
    console.print(f"slm-factory [bold]{__version__}[/bold]")


@tool_app.command(name="wizard")
def wizard(
    config: str = typer.Option("project.yaml", "--config", help=_CONFIG_HELP),
    resume: bool = typer.Option(False, "--resume", "-r", help="이전 실행의 중간 결과에서 재개합니다"),
) -> None:
    """대화형 파이프라인 — 단계별로 확인하며 실행합니다."""
    from rich.panel import Panel
    from rich.prompt import Confirm, Prompt
    from rich.table import Table

    console.print()
    console.print(
        Panel(
            "[bold cyan]slm-factory 대화형 파이프라인[/bold cyan]\n"
            "[dim]각 단계를 확인하며 진행합니다[/dim]",
            expand=False,
        )
    )

    # ── 컴퓨팅 디바이스 감지 ──────────────────────────────────────
    _wizard_device = None
    try:
        from .device import detect_device, print_device_summary

        _wizard_device = detect_device()
        print_device_summary(_wizard_device)
    except Exception:
        console.print("  [yellow]⚠ 디바이스 감지 실패 (PyTorch 미설치?)[/yellow]")

    # ── Step 1: 설정 파일 ─────────────────────────────────────────
    console.print("\n[bold]━━━ [1/12] 설정 파일 ━━━[/bold]")
    resolved = _find_config(config)
    if not Path(resolved).is_file():
        resolved = Prompt.ask("  설정 파일 경로를 입력하세요", default="project.yaml")
        resolved = _find_config(resolved)

    try:
        pipeline = _load_pipeline(resolved)
        console.print(f"  [green]✓[/green] [cyan]{resolved}[/cyan]")
        console.print(f"    프로젝트: {pipeline.config.project.name}")
        console.print(f"    Teacher : {pipeline.config.teacher.model}")
        console.print(f"    Student : {pipeline.config.student.model}")
    except Exception as e:
        _print_error("설정 로드 실패", e, hints=_get_error_hints(e))
        raise typer.Exit(code=1)

    pipeline.config.paths.ensure_dirs()
    _resume_cmd = f"slm-factory tool wizard --resume --config {resolved}"

    # ── 재개 지점 감지 ────────────────────────────────────────────
    skip_to_step = 1
    docs = None
    pairs: Any = []
    if resume:
        _resume_step, _resume_data = _find_resume_point(pipeline)
        if _resume_step == "generate":
            docs = _resume_data
            skip_to_step = 4
        elif _resume_step == "validate":
            pairs = _resume_data
            skip_to_step = 5
        elif _resume_step == "augment":
            pairs = _resume_data
            skip_to_step = 7
        elif _resume_step == "analyze":
            pairs = _resume_data
            skip_to_step = 8
        if _resume_step != "start":
            console.print(f"  [blue]ℹ[/blue] 이전 결과를 감지하여 재개합니다")

    # ── Step 2: 문서 선택 ─────────────────────────────────────────
    console.print("\n[bold]━━━ [2/12] 문서 선택 (필수) ━━━[/bold]")
    selected_files: list[Path] | None = None
    if skip_to_step > 2:
        console.print("  [yellow]⏭ 건너뜀 (이전 결과 사용)[/yellow]")
    else:
        doc_dir = pipeline.config.paths.documents
        if not doc_dir.is_dir():
            _print_error("문서 디렉토리 없음", f"디렉토리를 찾을 수 없음: {doc_dir}", ["documents 디렉토리를 생성하고 문서를 추가하세요"])
            raise typer.Exit(code=1)

        extensions = [
            ext if ext.startswith(".") else f".{ext}"
            for ext in pipeline.config.parsing.formats
        ]
        all_files = sorted(
            f for f in doc_dir.iterdir()
            if f.is_file() and f.suffix.lower() in extensions
        )

        if not all_files:
            _print_error("지원되는 문서 없음", f"디렉토리에 지원되는 문서가 없습니다: {doc_dir}", [f"지원 형식: {', '.join(extensions)}"])
            raise typer.Exit(code=1)

        file_table = Table(show_header=True, title=f"문서 목록 ({doc_dir})")
        file_table.add_column("#", style="dim", width=4)
        file_table.add_column("파일명", style="cyan")
        file_table.add_column("크기", justify="right")
        for i, f in enumerate(all_files, 1):
            size = f.stat().st_size
            if size < 1024:
                size_str = f"{size}B"
            elif size < 1024 * 1024:
                size_str = f"{size / 1024:.1f}KB"
            else:
                size_str = f"{size / 1024 / 1024:.1f}MB"
            file_table.add_row(str(i), f.name, size_str)
        console.print(file_table)

        use_all = Confirm.ask(
            f"  {len(all_files)}개 문서를 모두 사용하시겠습니까?", default=True,
        )
        if not use_all:
            while True:
                selection = Prompt.ask(
                    "  사용할 문서 번호 (쉼표 구분)",
                    default=",".join(str(i) for i in range(1, len(all_files) + 1)),
                )
                indices = []
                for part in selection.split(","):
                    s = part.strip()
                    if s.isdigit():
                        idx = int(s) - 1
                        if 0 <= idx < len(all_files):
                            indices.append(idx)
                        else:
                            console.print(f"  [yellow]⚠ 번호 {s}은(는) 범위 밖입니다 (1~{len(all_files)})[/yellow]")
                    elif s:
                        console.print(f"  [yellow]⚠ '{s}'은(는) 유효한 번호가 아닙니다[/yellow]")
                if indices:
                    break
                console.print(f"  [red]✗ 유효한 문서가 선택되지 않았습니다. 1~{len(all_files)} 범위의 번호를 입력하세요.[/red]")
            selected_files = [all_files[i] for i in indices]
            console.print(f"  [green]✓[/green] {len(selected_files)}개 문서 선택됨")

    # ── Step 3: 파싱 ──────────────────────────────────────────────
    console.print("\n[bold]━━━ [3/12] 문서 파싱 (필수) ━━━[/bold]")
    if skip_to_step > 3:
        console.print("  [yellow]⏭ 건너뜀 (이전 결과 사용)[/yellow]")
    else:
        try:
            docs = pipeline.step_parse(files=selected_files)
            console.print(f"  [green]✓[/green] {len(docs)}개 문서 파싱 완료")
        except Exception as e:
            _print_error("파싱 실패", e, hints=_get_error_hints(e), resume_cmd=_resume_cmd)
            raise typer.Exit(code=1)

    # ── Step 4: QA 생성 ───────────────────────────────────────────
    console.print("\n[bold]━━━ [4/12] QA 쌍 생성 (필수) ━━━[/bold]")
    if skip_to_step > 4:
        console.print("  [yellow]⏭ 건너뜀 (이전 결과 사용)[/yellow]")
    else:
        console.print(
            f"  Teacher: {pipeline.config.teacher.model} "
            f"({pipeline.config.teacher.backend})"
        )
        console.print("  [dim]Teacher LLM으로 문서 기반 질문-답변 쌍을 생성합니다. Ollama 실행이 필요합니다.[/dim]")
        if pipeline.config.teacher.backend == "ollama":
            import httpx

            try:
                resp = httpx.get(f"{pipeline.config.teacher.api_base}/api/tags", timeout=5)
                models = [m["name"] for m in resp.json().get("models", [])]
                teacher_model = pipeline.config.teacher.model
                model_found = any(
                    teacher_model == m or teacher_model == m.split(":")[0]
                    for m in models
                )
                if not model_found:
                    console.print(f"  [yellow]⚠ Teacher 모델 '{teacher_model}'이(가) Ollama에 없습니다[/yellow]")
                    console.print(f"  [dim]다운로드: ollama pull {teacher_model}[/dim]")
            except Exception:
                console.print("  [yellow]⚠ Ollama 서버에 연결할 수 없습니다[/yellow]")
                console.print("  [dim]실행: ollama serve[/dim]")
                if not Confirm.ask("  계속 진행하시겠습니까?", default=False):
                    raise typer.Exit(code=0)
        if not Confirm.ask("  QA 쌍을 생성하시겠습니까?", default=True):
            console.print("  [yellow]⏭ 건너뜀[/yellow]")
            console.print(Panel(
                f"[bold yellow]파이프라인 중단[/bold yellow]\n\n"
                f"  파싱된 문서: [cyan]{len(docs) if docs else 0}[/cyan]개\n"
                f"  파싱 결과: [cyan]{pipeline.output_dir / 'parsed_documents.json'}[/cyan]",
                expand=False,
            ))
            return

        assert docs is not None
        try:
            pairs = pipeline.step_generate(docs)
            console.print(f"  [green]✓[/green] {len(pairs)}개 QA 쌍 생성 완료")
        except Exception as e:
            _print_error("QA 생성 실패", e, hints=_get_error_hints(e), resume_cmd=_resume_cmd)
            raise typer.Exit(code=1)

    # ── Step 5: 검증 ──────────────────────────────────────────────
    console.print("\n[bold]━━━ [5/12] QA 검증 (필수) ━━━[/bold]")
    if skip_to_step > 5:
        console.print("  [yellow]⏭ 건너뜀 (이전 결과 사용)[/yellow]")
    else:
        try:
            total_before = len(pairs)
            if docs is not None:
                pairs = pipeline.step_validate(pairs, docs=docs)
            else:
                pairs = pipeline.step_validate(pairs)
            rejected = total_before - len(pairs)
            console.print(
                f"  [green]✓[/green] {len(pairs)}개 수락, {rejected}개 거부"
            )
        except Exception as e:
            _print_error("검증 실패", e, hints=_get_error_hints(e), resume_cmd=_resume_cmd)
            raise typer.Exit(code=1)

    # ── Step 6: 품질 평가 ─────────────────────────────────────────
    console.print("\n[bold]━━━ [6/12] 품질 점수 평가 (선택) ━━━[/bold]")
    if skip_to_step > 6:
        console.print("  [yellow]⏭ 건너뜀 (이전 결과 사용)[/yellow]")
    else:
        score_default = pipeline.config.scoring.enabled
        console.print("  [dim]Teacher LLM이 각 QA 쌍을 1~5점으로 평가하여 저품질 데이터를 제거합니다.[/dim]")
        console.print(
            f"  [dim]설정: scoring.enabled = {str(score_default).lower()}[/dim]"
        )
        if Confirm.ask("  품질 점수 평가를 하시겠습니까?", default=score_default):
            pipeline.config.scoring.enabled = True
            try:
                before = len(pairs)
                pairs = pipeline.step_score(pairs)
                console.print(
                    f"  [green]✓[/green] {len(pairs)}개 통과, "
                    f"{before - len(pairs)}개 제거"
                )
            except Exception as e:
                _print_error("점수 평가 실패", e, hints=_get_error_hints(e), resume_cmd=_resume_cmd)
                raise typer.Exit(code=1)
        else:
            console.print("  [yellow]⏭ 건너뜀[/yellow]")

    # ── Step 7: 데이터 증강 ───────────────────────────────────────
    console.print("\n[bold]━━━ [7/12] 데이터 증강 (선택) ━━━[/bold]")
    if skip_to_step > 7:
        console.print("  [yellow]⏭ 건너뜀 (이전 결과 사용)[/yellow]")
    else:
        augment_default = pipeline.config.augment.enabled
        console.print(
            f"  [dim]질문을 다양한 표현으로 변형하여 학습 데이터를 늘립니다 (설정: {pipeline.config.augment.num_variants}배).[/dim]"
        )
        console.print(
            f"  [dim]설정: augment.enabled = {str(augment_default).lower()}[/dim]"
        )
        if Confirm.ask("  데이터 증강을 하시겠습니까?", default=augment_default):
            pipeline.config.augment.enabled = True
            try:
                before = len(pairs)
                pairs = pipeline.step_augment(pairs)
                console.print(
                    f"  [green]✓[/green] {before}개 → {len(pairs)}개 "
                    f"({len(pairs) - before}개 증강)"
                )
            except Exception as e:
                _print_error("증강 실패", e, hints=_get_error_hints(e), resume_cmd=_resume_cmd)
                raise typer.Exit(code=1)
        else:
            console.print("  [yellow]⏭ 건너뜀[/yellow]")

    # ── 분석 (자동) ───────────────────────────────────────────────
    if skip_to_step < 8:
        pipeline.step_analyze(pairs)

    # ── Step 8: 학습 ──────────────────────────────────────────────
    console.print("\n[bold]━━━ [8/12] 모델 학습 (필수) ━━━[/bold]")
    console.print(f"  Student: {pipeline.config.student.model}")
    if _wizard_device is not None:
        _dev_label = {"cuda": "NVIDIA GPU (CUDA)", "mps": "Apple Silicon GPU (MPS)", "cpu": "CPU"}
        console.print(f"  디바이스: [bold]{_dev_label.get(_wizard_device.type, _wizard_device.type)}[/bold] — {_wizard_device.name}")
    console.print("  [dim]Student 모델에 LoRA 어댑터를 적용하여 파인튜닝합니다.[/dim]")
    try:
        training_data_path = pipeline.step_convert(pairs)
        console.print(
            f"  [green]✓[/green] 학습 데이터 변환 완료 ({len(pairs)}개 쌍)"
        )
    except Exception as e:
        _print_error("변환 실패", e, hints=_get_error_hints(e), resume_cmd=_resume_cmd)
        raise typer.Exit(code=1)

    if not Confirm.ask("  LoRA 학습을 진행하시겠습니까?", default=True):
        console.print("  [yellow]⏭ 건너뜀[/yellow]")
        console.print(Panel(
            f"[bold yellow]파이프라인 중단 (학습 건너뜀)[/bold yellow]\n\n"
            f"  총 QA 쌍: [cyan]{len(pairs)}[/cyan]개\n"
            f"  학습 데이터: [cyan]{training_data_path}[/cyan]\n\n"
            f"[bold]나중에 실행:[/bold]\n"
            f"  [cyan]slm-factory train --config {resolved}"
            f" --data {training_data_path}[/cyan]",
            expand=False,
        ))
        return

    try:
        adapter_path = pipeline.step_train(training_data_path)
        console.print(f"  [green]✓[/green] 학습 완료: [cyan]{adapter_path}[/cyan]")
    except Exception as e:
        _print_error("학습 실패", e, hints=_get_error_hints(e), resume_cmd=_resume_cmd)
        raise typer.Exit(code=1)

    # ── Step 9: 내보내기 ──────────────────────────────────────────
    console.print("\n[bold]━━━ [9/12] 모델 내보내기 (필수) ━━━[/bold]")
    console.print("  [dim]LoRA 어댑터를 기본 모델에 병합하고 Ollama 모델로 등록합니다.[/dim]")
    if not Confirm.ask("  모델을 내보내시겠습니까?", default=True):
        console.print("  [yellow]⏭ 건너뜀[/yellow]")
        console.print(Panel(
            f"[bold yellow]파이프라인 중단 (내보내기 건너뜀)[/bold yellow]\n\n"
            f"  총 QA 쌍: [cyan]{len(pairs)}[/cyan]개\n"
            f"  어댑터: [cyan]{adapter_path}[/cyan]\n\n"
            f"[bold]나중에 실행:[/bold]\n"
            f"  [cyan]slm-factory export --config {resolved}"
            f" --adapter {adapter_path}[/cyan]",
            expand=False,
        ))
        return

    try:
        model_dir = pipeline.step_export(adapter_path)
        console.print(
            f"  [green]✓[/green] 내보내기 완료: [cyan]{model_dir}[/cyan]"
        )
    except Exception as e:
        _print_error("내보내기 실패", e, hints=_get_error_hints(e), resume_cmd=_resume_cmd)
        raise typer.Exit(code=1)

    # ── Step 10: 멀티턴 대화 생성 (선택) ────────────────────────────
    console.print("\n[bold]━━━ [10/12] 멀티턴 대화 생성 (선택) ━━━[/bold]")
    console.print("  [dim]QA 쌍을 멀티턴 대화 형식으로 확장합니다. Ollama 실행 필요.[/dim]")
    if Confirm.ask("  멀티턴 대화를 생성하시겠습니까?", default=False):
        try:
            from .utils import run_async

            from .teacher import create_teacher
            from .teacher.dialogue_generator import DialogueGenerator

            teacher = create_teacher(pipeline.config.teacher)
            gen = DialogueGenerator(teacher, pipeline.config.dialogue, pipeline.config.teacher)
            dialogues = run_async(gen.generate_all(pairs))
            dialogue_path = pipeline.output_dir / "dialogues.json"
            gen.save_dialogues(dialogues, dialogue_path)
            console.print(f"  [green]✓[/green] {len(dialogues)}개 대화 생성 → [cyan]{dialogue_path}[/cyan]")
        except Exception as e:
            _print_error("대화 생성 실패", e, hints=_get_error_hints(e))
            console.print("  [yellow]⏭ 건너뛰고 계속합니다[/yellow]")
    else:
        console.print("  [yellow]⏭ 건너뜀[/yellow]")

    # ── Step 11: GGUF 변환 (선택) ─────────────────────────────────
    console.print("\n[bold]━━━ [11/12] GGUF 변환 (선택) ━━━[/bold]")
    console.print(f"  [dim]모델을 GGUF 양자화 형식으로 변환합니다 (llama.cpp). 양자화: {pipeline.config.gguf_export.quantization_type}[/dim]")
    if Confirm.ask("  GGUF 변환을 하시겠습니까?", default=False):
        try:
            from .exporter.gguf_export import GGUFExporter

            exporter = GGUFExporter(pipeline.config)
            gguf_path = exporter.export(model_dir)
            console.print(f"  [green]✓[/green] GGUF 변환 완료 → [cyan]{gguf_path}[/cyan]")
        except Exception as e:
            _print_error("GGUF 변환 실패", e, hints=_get_error_hints(e))
            console.print("  [yellow]⏭ 건너뛰고 계속합니다[/yellow]")
    else:
        console.print("  [yellow]⏭ 건너뜀[/yellow]")

    # ── Step 12: 모델 평가 (선택) ─────────────────────────────────
    console.print("\n[bold]━━━ [12/12] 모델 평가 (선택) ━━━[/bold]")
    console.print(f"  [dim]학습된 모델의 품질을 QA 데이터로 평가합니다 (BLEU/ROUGE). Ollama 실행 필요.[/dim]")
    if Confirm.ask("  모델 평가를 하시겠습니까?", default=False):
        eval_model_name = Prompt.ask(
            "  평가할 모델 이름",
            default=pipeline.config.export.ollama.model_name,
        )
        try:
            from .evaluator import ModelEvaluator

            evaluator = ModelEvaluator(pipeline.config)
            results = evaluator.evaluate(pairs, eval_model_name)
            eval_path = pipeline.output_dir / pipeline.config.eval.output_file
            evaluator.save_results(results, eval_path)
            evaluator.print_summary(results)
            console.print(f"  [green]✓[/green] 평가 완료 ({len(results)}건) → [cyan]{eval_path}[/cyan]")
        except Exception as e:
            _print_error("평가 실패", e, hints=_get_error_hints(e))
            console.print("  [yellow]⏭ 건너뛰고 계속합니다[/yellow]")
    else:
        console.print("  [yellow]⏭ 건너뜀[/yellow]")

    # ── 완료 ──────────────────────────────────────────────────────
    console.print()
    summary = (
        f"[bold green]파이프라인 완료![/bold green]\n\n"
        f"  총 QA 쌍: [cyan]{len(pairs)}[/cyan]개\n"
        f"  Student 모델: [cyan]{pipeline.config.student.model}[/cyan]\n"
        f"  모델 출력: [cyan]{model_dir}[/cyan]\n\n"
        f"[bold]모델 실행:[/bold]\n"
        f"  [cyan]ollama run {pipeline.config.export.ollama.model_name}[/cyan]"
    )
    console.print(Panel(summary, expand=False))


@eval_app.command(name="run")
def eval_model(
    config: str = typer.Option("project.yaml", "--config", help=_CONFIG_HELP),
    model: str = typer.Option(..., "--model", help="평가할 Ollama 모델 이름입니다"),
    data: Optional[str] = typer.Option(
        None, "--data", help="QA 데이터 파일 경로 (미지정 시 출력 디렉토리에서 자동 감지)"
    ),
) -> None:
    """학습된 모델을 QA 데이터로 평가합니다."""
    try:
        from .evaluator import ModelEvaluator

        pipeline = _load_pipeline(config)
        pipeline.config.paths.ensure_dirs()

        pairs = _load_qa_data(pipeline, data)

        evaluator = ModelEvaluator(pipeline.config)
        results = evaluator.evaluate(pairs, model)

        eval_output = pipeline.output_dir / pipeline.config.eval.output_file
        evaluator.save_results(results, eval_output)
        evaluator.print_summary(results)

        console.print(
            f"\n[bold green]평가 완료![/bold green] "
            f"결과: [cyan]{eval_output}[/cyan] ({len(results)}건)\n"
        )

    except FileNotFoundError as e:
        _print_error("설정 파일 오류", e, hints=_get_error_hints(e))
        raise typer.Exit(code=1)
    except Exception as e:
        _print_error("평가 실패", e, hints=_get_error_hints(e))
        raise typer.Exit(code=1)


@tool_app.command(name="gguf")
def export_gguf(
    config: str = typer.Option("project.yaml", "--config", help=_CONFIG_HELP),
    model_dir: Optional[str] = typer.Option(
        None, "--model-dir", help="병합된 모델 디렉토리 경로 (기본값: output/merged_model)"
    ),
) -> None:
    """병합된 모델을 GGUF 형식으로 변환합니다 (llama.cpp 사용)."""
    try:
        from .exporter.gguf_export import GGUFExporter

        pipeline = _load_pipeline(config)
        pipeline.config.paths.ensure_dirs()

        if model_dir is not None:
            resolved_model_dir = Path(model_dir)
        else:
            resolved_model_dir = pipeline.config.paths.output / "merged_model"

        if not resolved_model_dir.is_dir():
            _print_error(
                "모델 디렉토리 미발견",
                f"디렉토리를 찾을 수 없음: {resolved_model_dir}",
                ["--model-dir 옵션으로 경로를 지정하거나 export 명령을 먼저 실행하세요"],
            )
            raise typer.Exit(code=1)

        exporter = GGUFExporter(pipeline.config)
        gguf_path = exporter.export(resolved_model_dir)

        console.print(
            f"\n[bold green]GGUF 변환 완료![/bold green] "
            f"파일: [cyan]{gguf_path}[/cyan]\n"
        )

    except FileNotFoundError as e:
        _print_error("파일 오류", e, hints=_get_error_hints(e))
        raise typer.Exit(code=1)
    except Exception as e:
        _print_error("GGUF 변환 실패", e, hints=_get_error_hints(e))
        raise typer.Exit(code=1)


@tool_app.command(name="evolve")
def evolve(
    config: str = typer.Option("project.yaml", "--config", help=_CONFIG_HELP),
    force_update: bool = typer.Option(
        False, "--force-update", help="변경 감지를 무시하고 전체 문서를 재처리합니다",
    ),
    skip_gate: bool = typer.Option(
        False, "--skip-gate", help="품질 게이트를 건너뛰고 무조건 배포합니다",
    ),
) -> None:
    """증분 업데이트 → 재학습 → 품질 게이트 → 버전된 모델 배포를 단일 명령으로 실행합니다."""
    from rich.panel import Panel

    try:
        from .evolve_history import EvolveHistory
        from .incremental import IncrementalTracker

        pipeline = _load_pipeline(config)
        pipeline.config.paths.ensure_dirs()

        history = EvolveHistory(pipeline.config)
        tracker = IncrementalTracker(pipeline.config)

        # ── 1. 증분 업데이트 ──────────────────────────────────────
        console.print("\n[bold]━━━ [1/5] 증분 업데이트 ━━━[/bold]")

        if force_update:
            console.print("  [yellow]--force-update: 전체 문서 재처리[/yellow]")
            docs = pipeline.step_parse()
            pairs = pipeline.step_generate(docs)
        else:
            changed_files = tracker.get_changed_files(pipeline.config.paths.documents)
            if not changed_files:
                console.print("  변경된 문서가 없습니다")
                console.print(
                    "\n[dim]힌트: --force-update 옵션으로 전체 재처리할 수 있습니다[/dim]\n"
                )
                return

            console.print(f"  {len(changed_files)}개 변경 문서 감지")
            docs = pipeline.step_parse(files=changed_files)
            pairs = pipeline.step_generate(docs)

        existing_path = pipeline.output_dir / "qa_alpaca.json"
        if existing_path.is_file():
            existing_pairs = pipeline._load_pairs(existing_path)
        else:
            existing_pairs = []

        strategy = pipeline.config.incremental.merge_strategy
        merged = tracker.merge_qa_pairs(existing_pairs, pairs, strategy)
        pipeline._save_pairs(merged, existing_path)
        tracker.commit_hashes()  # 처리 성공 후 해시 확정
        console.print(
            f"  [green]✓[/green] 새 QA {len(pairs)}개, 전체 {len(merged)}개 (전략: {strategy})"
        )

        # ── 2. 검증 + 점수 + 증강 ────────────────────────────────
        console.print("\n[bold]━━━ [2/5] 검증 · 점수 · 증강 ━━━[/bold]")
        pairs = pipeline.step_validate(merged, docs=docs)
        console.print(f"  [green]✓[/green] 검증: {len(pairs)}개 통과")

        pairs = pipeline.step_score(pairs)
        pairs = pipeline.step_augment(pairs)
        pipeline.step_analyze(pairs)
        console.print(f"  [green]✓[/green] 최종 학습 데이터: {len(pairs)}개")

        # ── 3. 변환 + 학습 + 내보내기 ────────────────────────────
        console.print("\n[bold]━━━ [3/5] 학습 · 내보내기 ━━━[/bold]")
        training_data_path = pipeline.step_convert(pairs)
        adapter_path = pipeline.step_train(training_data_path)

        from .exporter import HFExporter

        hf_exporter = HFExporter(pipeline.config)
        model_dir = hf_exporter.export(adapter_path)
        console.print(f"  [green]✓[/green] 모델 병합: {model_dir}")

        # ── 4. 버전 생성 + 품질 게이트 ───────────────────────────
        console.print("\n[bold]━━━ [4/5] 품질 게이트 ━━━[/bold]")
        version = history.generate_version_name()
        versioned_name = history.generate_model_name(version)
        is_first = history.is_first_run()

        if is_first or skip_gate:
            reason = "첫 실행 (비교 대상 없음)" if is_first else "--skip-gate"
            console.print(f"  [yellow]품질 게이트 건너뜀:[/yellow] {reason}")

            if pipeline.config.export.ollama.enabled:
                from .exporter import OllamaExporter

                ollama_exporter = OllamaExporter(pipeline.config)
                modelfile_path = ollama_exporter.generate_modelfile(model_dir)
                success = ollama_exporter.create_model(
                    modelfile_path, model_name_override=versioned_name,
                )
                if success:
                    console.print(f"  [green]✓[/green] Ollama 모델 생성: {versioned_name}")
                else:
                    console.print(f"  [yellow]⚠[/yellow] Ollama 모델 생성 실패 (수동: ollama create {versioned_name} -f {modelfile_path})")

            history.record_version(
                version, versioned_name, qa_count=len(pairs), promoted=True,
            )
        else:
            previous_name = history.get_current_model_name()
            if previous_name is None:
                console.print("  [yellow]이전 모델 정보 없음 — 무조건 배포[/yellow]")
                gate_passed = True
                gate_scores: dict[str, float] = {}
            else:
                console.print(
                    f"  비교: {previous_name} vs {versioned_name}"
                )

                if pipeline.config.export.ollama.enabled:
                    from .exporter import OllamaExporter

                    ollama_exporter = OllamaExporter(pipeline.config)
                    modelfile_path = ollama_exporter.generate_modelfile(model_dir)
                    ollama_exporter.create_model(
                        modelfile_path, model_name_override=versioned_name,
                    )

                pipeline.config.compare.enabled = True
                pipeline.config.compare.base_model = previous_name
                pipeline.config.compare.finetuned_model = versioned_name

                try:
                    compare_results = pipeline.step_compare(pairs)
                    gate_passed, gate_scores = history.check_quality_gate(compare_results)
                except Exception as e:
                    console.print(f"  [yellow]⚠[/yellow] 품질 비교 실패: {e}")
                    console.print("  [yellow]모델은 생성되었으나 품질 검증 미완료[/yellow]")
                    history.record_version(
                        version, versioned_name, qa_count=len(pairs), promoted=False,
                    )
                    console.print(
                        Panel(
                            f"[bold yellow]진화 부분 완료[/bold yellow]\n\n"
                            f"  버전: [cyan]{version}[/cyan]\n"
                            f"  모델: [cyan]{versioned_name}[/cyan]\n"
                            f"  상태: 품질 검증 보류\n\n"
                            f"Ollama 실행 후 수동 비교:\n"
                            f"  [cyan]slm-factory eval compare --base-model {previous_name} --ft {versioned_name}[/cyan]",
                            expand=False,
                        )
                    )
                    return

            if gate_passed:
                improvement = gate_scores.get("improvement_pct", 0)
                console.print(
                    f"  [green]✓[/green] 품질 게이트 통과 ({improvement:+.1f}%)"
                )
                history.record_version(
                    version, versioned_name, scores=gate_scores,
                    qa_count=len(pairs), promoted=True,
                )
            else:
                improvement = gate_scores.get("improvement_pct", 0)
                console.print(
                    f"  [red]✗[/red] 품질 게이트 실패 ({improvement:+.1f}%)"
                )
                history.record_version(
                    version, versioned_name, scores=gate_scores,
                    qa_count=len(pairs), promoted=False,
                )
                # 품질 게이트 실패 시 생성된 Ollama 모델 정리
                EvolveHistory._ollama_rm(versioned_name)
                console.print(
                    Panel(
                        f"[bold red]진화 중단 — 품질 미달[/bold red]\n\n"
                        f"  버전: [cyan]{version}[/cyan]\n"
                        f"  모델: [cyan]{versioned_name}[/cyan] (미배포)\n"
                        f"  메트릭: {pipeline.config.evolve.gate_metric}\n"
                        f"  개선율: {improvement:+.1f}% "
                        f"(최소 {pipeline.config.evolve.gate_min_improvement}% 필요)\n\n"
                        f"현재 활성 모델: [green]{previous_name}[/green]",
                        expand=False,
                    )
                )
                return

        # ── 5. 정리 + 완료 ───────────────────────────────────────
        console.print("\n[bold]━━━ [5/5] 완료 ━━━[/bold]")
        removed = history.cleanup_old_versions()
        if removed:
            console.print(f"  이전 버전 정리: {', '.join(removed)}")

        console.print(
            Panel(
                f"[bold green]진화 완료![/bold green]\n\n"
                f"  버전: [cyan]{version}[/cyan]\n"
                f"  모델: [cyan]{versioned_name}[/cyan]\n"
                f"  QA 쌍: [cyan]{len(pairs)}[/cyan]개\n"
                f"  모델 경로: [cyan]{model_dir}[/cyan]\n\n"
                f"모델 실행:\n"
                f"  [cyan]ollama run {versioned_name}[/cyan]",
                expand=False,
            )
        )

    except FileNotFoundError as e:
        _print_error("설정 파일 오류", e, hints=_get_error_hints(e))
        raise typer.Exit(code=1)
    except Exception as e:
        _print_error("진화 실패", e, hints=_get_error_hints(e))
        raise typer.Exit(code=1)


@tool_app.command(name="update")
def update(
    config: str = typer.Option("project.yaml", "--config", help=_CONFIG_HELP),
) -> None:
    """변경된 문서만 증분 처리합니다."""
    try:
        from .incremental import IncrementalTracker

        pipeline = _load_pipeline(config)
        pipeline.config.paths.ensure_dirs()

        tracker = IncrementalTracker(pipeline.config)
        changed_files = tracker.get_changed_files(pipeline.config.paths.documents)

        if not changed_files:
            console.print("\n변경된 문서가 없습니다\n")
            return

        console.print(
            f"\n[bold blue]증분 업데이트:[/bold blue] {len(changed_files)}개 변경 문서 감지\n"
        )

        docs = pipeline.step_parse(files=changed_files)
        pairs = pipeline.step_generate(docs)

        existing_path = pipeline.output_dir / "qa_alpaca.json"
        if existing_path.is_file():
            existing_pairs = pipeline._load_pairs(existing_path)
        else:
            existing_pairs = []

        strategy = pipeline.config.incremental.merge_strategy
        merged = tracker.merge_qa_pairs(existing_pairs, pairs, strategy)

        pipeline._save_pairs(merged, existing_path)
        tracker.commit_hashes()  # 처리 성공 후 해시 확정

        console.print(
            f"\n[bold green]증분 업데이트 완료![/bold green] "
            f"변경 문서: {len(changed_files)}개, "
            f"새 QA: {len(pairs)}개, "
            f"전체 QA: {len(merged)}개 "
            f"(전략: {strategy})\n"
        )

    except FileNotFoundError as e:
        _print_error("설정 파일 오류", e, hints=_get_error_hints(e))
        raise typer.Exit(code=1)
    except Exception as e:
        _print_error("증분 업데이트 실패", e, hints=_get_error_hints(e))
        raise typer.Exit(code=1)


@tool_app.command(name="dialogue")
def generate_dialogue(
    config: str = typer.Option("project.yaml", "--config", help=_CONFIG_HELP),
    data: Optional[str] = typer.Option(
        None, "--data", help="QA 데이터 파일 경로 (qa_alpaca.json 또는 qa_augmented.json)"
    ),
) -> None:
    """QA 쌍에서 멀티턴 대화 데이터를 생성합니다."""
    try:
        from .utils import run_async

        pipeline = _load_pipeline(config)
        pipeline.config.paths.ensure_dirs()

        pairs = _load_qa_data(pipeline, data)

        from .teacher import create_teacher
        from .teacher.dialogue_generator import DialogueGenerator

        teacher = create_teacher(pipeline.config.teacher)
        generator = DialogueGenerator(
            teacher, pipeline.config.dialogue, pipeline.config.teacher
        )
        dialogues = run_async(generator.generate_all(pairs))

        dialogue_path = pipeline.output_dir / "dialogues.json"
        generator.save_dialogues(dialogues, dialogue_path)

        console.print(
            f"\n[bold green]대화 생성 완료![/bold green] "
            f"{len(dialogues)}개 대화 생성됨 → [cyan]{dialogue_path}[/cyan]\n"
        )

    except FileNotFoundError as e:
        _print_error("설정 파일 오류", e, hints=_get_error_hints(e))
        raise typer.Exit(code=1)
    except Exception as e:
        _print_error("대화 생성 실패", e, hints=_get_error_hints(e))
        raise typer.Exit(code=1)


@eval_app.command(name="compare")
def compare_models(
    config: str = typer.Option("project.yaml", "--config", help=_CONFIG_HELP),
    base_model: str = typer.Option(..., "--base-model", help="비교 기준 모델 이름 (Ollama)"),
    finetuned_model: str = typer.Option(..., "--ft", help="파인튜닝된 모델 이름 (Ollama)"),
    data: Optional[str] = typer.Option(None, "--data", help="QA 데이터 파일 경로"),
) -> None:
    """Base 모델과 Fine-tuned 모델의 답변을 비교합니다."""
    try:
        from .comparator import ModelComparator

        pipeline = _load_pipeline(config)
        pipeline.config.paths.ensure_dirs()

        pairs = _load_qa_data(pipeline, data)

        pipeline.config.compare.base_model = base_model
        pipeline.config.compare.finetuned_model = finetuned_model

        comparator = ModelComparator(pipeline.config)
        results = comparator.compare(pairs)

        compare_output = pipeline.output_dir / pipeline.config.compare.output_file
        comparator.save_results(results, compare_output)
        comparator.print_summary(results)

        console.print(
            f"\n[bold green]비교 완료![/bold green] "
            f"결과: [cyan]{compare_output}[/cyan] ({len(results)}건)\n"
        )

    except FileNotFoundError as e:
        _print_error("설정 파일 오류", e, hints=_get_error_hints(e))
        raise typer.Exit(code=1)
    except Exception as e:
        _print_error("비교 실패", e, hints=_get_error_hints(e))
        raise typer.Exit(code=1)


@tool_app.command(name="dashboard")
def dashboard(
    config: str = typer.Option("project.yaml", "--config", help=_CONFIG_HELP),
) -> None:
    """파이프라인 모니터링 TUI 대시보드를 실행합니다."""
    from .config import load_config

    try:
        cfg = load_config(_find_config(config))
    except Exception as e:
        _print_error("설정 로드 실패", e, hints=_get_error_hints(e))
        raise typer.Exit(code=1)

    output_dir = Path(cfg.paths.output)

    try:
        from .tui.dashboard import PipelineDashboard

        dash_app = PipelineDashboard(
            output_dir=output_dir,
            refresh_interval=cfg.dashboard.refresh_interval,
        )
        dash_app.run()
    except Exception as e:
        _print_error("대시보드 실행 실패", e, hints=_get_error_hints(e))
        raise typer.Exit(code=1)


@tool_app.command(name="review")
def review_qa(
    config: str = typer.Option("project.yaml", "--config", help=_CONFIG_HELP),
    data: Optional[str] = typer.Option(None, "--data", help="QA 데이터 파일 경로"),
) -> None:
    """QA 쌍을 수동으로 리뷰하는 TUI를 실행합니다."""
    try:
        pipeline = _load_pipeline(config)
        pipeline.config.paths.ensure_dirs()

        pairs = _load_qa_data(
            pipeline, data, extra_candidates=["qa_reviewed.json"],
        )

        output_path = pipeline.output_dir / pipeline.config.review.output_file

        from .tui.reviewer import QAReviewerApp

        reviewer_app = QAReviewerApp(pairs=pairs, output_path=output_path)
        reviewer_app.run()

        statuses = QAReviewerApp.count_statuses(pairs)
        console.print(
            f"\n[bold green]리뷰 완료![/bold green] "
            f"승인: [green]{statuses['approved']}[/green], "
            f"거부: [red]{statuses['rejected']}[/red], "
            f"대기: [yellow]{statuses['pending']}[/yellow]\n"
        )

    except FileNotFoundError as e:
        _print_error("설정 파일 오류", e, hints=_get_error_hints(e))
        raise typer.Exit(code=1)
    except Exception as e:
        _print_error("리뷰 실행 실패", e, hints=_get_error_hints(e))
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# 진입점
# ---------------------------------------------------------------------------


def main() -> None:
    """pyproject.toml에서 호출되는 진입점입니다."""
    app()
