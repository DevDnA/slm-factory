"""slm-factory용 명령줄 인터페이스입니다."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import typer
from rich.console import Console

from . import __version__

if TYPE_CHECKING:
    from .pipeline import Pipeline

app = typer.Typer(
    name="slm-factory",
    help="Teacher-Student Knowledge Distillation framework for domain-specific SLMs.",
    no_args_is_help=True,
)
console = Console()


# ---------------------------------------------------------------------------
# 헬퍼
# ---------------------------------------------------------------------------


def _load_pipeline(config_path: str) -> Pipeline:
    """설정을 로드하고 Pipeline 인스턴스를 반환합니다."""
    from .config import load_config
    from .pipeline import Pipeline
    from .utils import setup_logging

    setup_logging()

    config = load_config(config_path)
    return Pipeline(config)


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


@app.command()
def init(
    name: str = typer.Option(..., "--name", help="Project name"),
    path: str = typer.Option(".", "--path", help="Parent directory for the project"),
) -> None:
    """새로운 slm-factory 프로젝트를 초기화합니다."""
    from .config import create_default_config

    project_dir = Path(path) / name
    documents_dir = project_dir / "documents"
    output_dir = project_dir / "output"

    project_dir.mkdir(parents=True, exist_ok=True)
    documents_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    template = create_default_config()
    config_content = template.replace('name: "my-project"', f'name: "{name}"')
    config_content = config_content.replace(
        'model_name: "my-project-model"', f'model_name: "{name}-model"'
    )

    config_path = project_dir / "project.yaml"
    config_path.write_text(config_content, encoding="utf-8")

    console.print(f"\n[bold green]Project '{name}' created at {project_dir}[/bold green]\n")
    console.print("프로젝트 구조:")
    console.print(f"  {project_dir}/")
    console.print(f"  {documents_dir}/")
    console.print(f"  {output_dir}/")
    console.print(f"  {config_path}")
    console.print("\n[bold]다음 단계:[/bold]")
    console.print(f"  1. [cyan]{documents_dir}[/cyan]에 문서 추가")
    console.print(f"  2. [cyan]{config_path}[/cyan]를 편집하여 설정 커스터마이징")
    console.print(f"  3. 실행: [cyan]slm-factory run --config {config_path}[/cyan]\n")


@app.command()
def run(
    config: str = typer.Option("project.yaml", "--config", help="Path to project.yaml"),
    resume: bool = typer.Option(
        False, "--resume", "-r", help="중간 저장 파일에서 재개합니다"
    ),
) -> None:
    """전체 파이프라인을 실행합니다 (파싱 → 생성 → 검증 → 변환 → 훈련 → 내보내기)."""
    try:
        console.print(
            "\n[bold blue]slm-factory[/bold blue] — 전체 파이프라인 시작 중...\n"
        )

        pipeline = _load_pipeline(config)

        if not resume:
            model_dir = pipeline.run()
        else:
            pipeline.config.paths.ensure_dirs()
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
                    pairs = pipeline.step_validate(data)
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
        console.print(f"\n[bold red]오류:[/bold red] {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"\n[bold red]파이프라인 실패:[/bold red] {e}")
        raise typer.Exit(code=1)


@app.command()
def parse(
    config: str = typer.Option("project.yaml", "--config", help="Path to project.yaml"),
) -> None:
    """문서 파싱 단계만 실행합니다."""
    try:
        pipeline = _load_pipeline(config)
        pipeline.config.paths.ensure_dirs()
        docs = pipeline.step_parse()

        console.print(
            f"\n[bold green]{len(docs)}개 문서 파싱 완료[/bold green]\n"
        )

    except FileNotFoundError as e:
        console.print(f"\n[bold red]오류:[/bold red] {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"\n[bold red]파싱 실패:[/bold red] {e}")
        raise typer.Exit(code=1)


@app.command()
def generate(
    config: str = typer.Option("project.yaml", "--config", help="Path to project.yaml"),
) -> None:
    """파싱 + QA 생성 단계를 실행합니다."""
    try:
        pipeline = _load_pipeline(config)
        pipeline.config.paths.ensure_dirs()

        docs = pipeline.step_parse()
        pairs = pipeline.step_generate(docs)

        console.print(
            f"\n[bold green]{len(docs)}개 문서에서 {len(pairs)}개 QA 쌍 생성 완료[/bold green]\n"
        )

    except FileNotFoundError as e:
        console.print(f"\n[bold red]오류:[/bold red] {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"\n[bold red]생성 실패:[/bold red] {e}")
        raise typer.Exit(code=1)


@app.command()
def validate(
    config: str = typer.Option("project.yaml", "--config", help="Path to project.yaml"),
) -> None:
    """파싱 + 생성 + 검증 단계를 실행합니다."""
    try:
        pipeline = _load_pipeline(config)
        pipeline.config.paths.ensure_dirs()

        docs = pipeline.step_parse()
        pairs = pipeline.step_generate(docs)
        total_generated = len(pairs)

        accepted = pipeline.step_validate(pairs, docs=docs)
        rejected_count = total_generated - len(accepted)

        console.print(
            f"\n[bold green]검증 완료:[/bold green] "
            f"{len(accepted)}개 수락, {rejected_count}개 거부 "
            f"({total_generated}개 생성 중)\n"
        )

    except FileNotFoundError as e:
        console.print(f"\n[bold red]오류:[/bold red] {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"\n[bold red]검증 실패:[/bold red] {e}")
        raise typer.Exit(code=1)


@app.command()
def score(
    config: str = typer.Option("project.yaml", "--config", help="Path to project.yaml"),
    resume: bool = typer.Option(
        False, "--resume", "-r", help="중간 저장 파일에서 재개합니다"
    ),
) -> None:
    """파싱 + 생성 + 검증 + 품질 점수 평가를 실행합니다."""
    try:
        pipeline = _load_pipeline(config)
        pipeline.config.paths.ensure_dirs()

        if resume:
            step, data = _find_resume_point(pipeline)
            if step == "generate":
                docs = data
                pairs = pipeline.step_generate(docs)
                pairs = pipeline.step_validate(pairs, docs=docs)
            elif step == "validate":
                pairs = pipeline.step_validate(data)
            elif step in ("augment", "analyze"):
                console.print(
                    f"\n[bold green]점수 평가 이미 완료됨 ({len(data)}개 쌍)[/bold green]\n"
                )
                return
            else:
                docs = pipeline.step_parse()
                pairs = pipeline.step_generate(docs)
                pairs = pipeline.step_validate(pairs, docs=docs)
        else:
            docs = pipeline.step_parse()
            pairs = pipeline.step_generate(docs)
            pairs = pipeline.step_validate(pairs, docs=docs)

        before = len(pairs)
        pairs = pipeline.step_score(pairs)

        console.print(
            f"\n[bold green]점수 평가 완료:[/bold green] "
            f"{len(pairs)}개 통과, {before - len(pairs)}개 제거 "
            f"({before}개 대상 중)\n"
        )

    except FileNotFoundError as e:
        console.print(f"\n[bold red]오류:[/bold red] {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"\n[bold red]점수 평가 실패:[/bold red] {e}")
        raise typer.Exit(code=1)


@app.command()
def augment(
    config: str = typer.Option("project.yaml", "--config", help="Path to project.yaml"),
    resume: bool = typer.Option(
        False, "--resume", "-r", help="중간 저장 파일에서 재개합니다"
    ),
) -> None:
    """파싱 + 생성 + 검증 + 점수 평가 + 데이터 증강을 실행합니다."""
    try:
        pipeline = _load_pipeline(config)
        pipeline.config.paths.ensure_dirs()

        if resume:
            step, data = _find_resume_point(pipeline)
            if step == "generate":
                docs = data
                pairs = pipeline.step_generate(docs)
                pairs = pipeline.step_validate(pairs, docs=docs)
                pairs = pipeline.step_score(pairs)
            elif step == "validate":
                pairs = pipeline.step_validate(data)
                pairs = pipeline.step_score(pairs)
            elif step == "augment":
                pairs = data
            elif step == "analyze":
                console.print(
                    f"\n[bold green]데이터 증강 이미 완료됨 ({len(data)}개 쌍)[/bold green]\n"
                )
                return
            else:
                docs = pipeline.step_parse()
                pairs = pipeline.step_generate(docs)
                pairs = pipeline.step_validate(pairs, docs=docs)
                pairs = pipeline.step_score(pairs)
        else:
            docs = pipeline.step_parse()
            pairs = pipeline.step_generate(docs)
            pairs = pipeline.step_validate(pairs, docs=docs)
            pairs = pipeline.step_score(pairs)

        before = len(pairs)
        pairs = pipeline.step_augment(pairs)

        console.print(
            f"\n[bold green]데이터 증강 완료:[/bold green] "
            f"{before}개 → {len(pairs)}개 "
            f"(증강 {len(pairs) - before}개 추가)\n"
        )

    except FileNotFoundError as e:
        console.print(f"\n[bold red]오류:[/bold red] {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"\n[bold red]증강 실패:[/bold red] {e}")
        raise typer.Exit(code=1)


@app.command()
def analyze(
    config: str = typer.Option("project.yaml", "--config", help="Path to project.yaml"),
    resume: bool = typer.Option(
        False, "--resume", "-r", help="중간 저장 파일에서 재개합니다"
    ),
) -> None:
    """파싱 + 생성 + 검증 + 점수 평가 + 증강 후 데이터 분석을 실행합니다."""
    try:
        pipeline = _load_pipeline(config)
        pipeline.config.paths.ensure_dirs()

        if resume:
            step, data = _find_resume_point(pipeline)
            if step == "generate":
                docs = data
                pairs = pipeline.step_generate(docs)
                pairs = pipeline.step_validate(pairs, docs=docs)
                pairs = pipeline.step_score(pairs)
                pairs = pipeline.step_augment(pairs)
            elif step == "validate":
                pairs = pipeline.step_validate(data)
                pairs = pipeline.step_score(pairs)
                pairs = pipeline.step_augment(pairs)
            elif step == "augment":
                pairs = pipeline.step_augment(data)
            elif step == "analyze":
                pairs = data
            else:
                docs = pipeline.step_parse()
                pairs = pipeline.step_generate(docs)
                pairs = pipeline.step_validate(pairs, docs=docs)
                pairs = pipeline.step_score(pairs)
                pairs = pipeline.step_augment(pairs)
        else:
            docs = pipeline.step_parse()
            pairs = pipeline.step_generate(docs)
            pairs = pipeline.step_validate(pairs, docs=docs)
            pairs = pipeline.step_score(pairs)
            pairs = pipeline.step_augment(pairs)

        pipeline.step_analyze(pairs)

        console.print(
            f"\n[bold green]분석 완료:[/bold green] "
            f"{len(pairs)}개 QA 쌍 분석됨\n"
        )

    except FileNotFoundError as e:
        console.print(f"\n[bold red]오류:[/bold red] {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"\n[bold red]분석 실패:[/bold red] {e}")
        raise typer.Exit(code=1)


@app.command()
def train(
    config: str = typer.Option("project.yaml", "--config", help="Path to project.yaml"),
    data: Optional[str] = typer.Option(
        None, "--data", help="Path to pre-generated training_data.jsonl"
    ),
    resume: bool = typer.Option(
        False, "--resume", "-r", help="중간 저장 파일에서 재개합니다"
    ),
) -> None:
    """훈련 단계를 실행합니다 (선택적으로 사전 생성된 데이터 사용)."""
    try:
        pipeline = _load_pipeline(config)
        pipeline.config.paths.ensure_dirs()

        if data is not None:
            training_data_path = Path(data)
            if not training_data_path.is_file():
                console.print(
                    f"\n[bold red]오류:[/bold red] 훈련 데이터를 찾을 수 없음: {training_data_path}"
                )
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
                pairs = pipeline.step_validate(loaded)
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
        console.print(f"\n[bold red]오류:[/bold red] {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"\n[bold red]훈련 실패:[/bold red] {e}")
        raise typer.Exit(code=1)


@app.command()
def check(
    config: str = typer.Option("project.yaml", "--config", help="Path to project.yaml"),
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

        cfg = load_config(config)
        table.add_row("설정 파일", "[green]OK[/green]", str(config))
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

    console.print()
    console.print(table)
    if all_ok:
        console.print("\n[bold green]모든 점검 통과![/bold green]\n")
    else:
        console.print("\n[bold yellow]일부 항목에 주의가 필요합니다.[/bold yellow]\n")
        raise typer.Exit(code=1)


@app.command()
def version() -> None:
    """slm-factory 버전을 표시합니다."""
    console.print(f"slm-factory [bold]{__version__}[/bold]")


# ---------------------------------------------------------------------------
# 진입점
# ---------------------------------------------------------------------------


def main() -> None:
    """pyproject.toml에서 호출되는 진입점입니다."""
    app()
