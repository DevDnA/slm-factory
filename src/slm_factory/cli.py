"""slm-factory용 명령줄 인터페이스입니다."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from . import __version__

app = typer.Typer(
    name="slm-factory",
    help="Teacher-Student Knowledge Distillation framework for domain-specific SLMs.",
    no_args_is_help=True,
)
console = Console()


# ---------------------------------------------------------------------------
# 헬퍼
# ---------------------------------------------------------------------------


def _load_pipeline(config_path: str) -> "Pipeline":
    """설정을 로드하고 Pipeline 인스턴스를 반환합니다."""
    from .config import load_config
    from .pipeline import Pipeline
    from .utils import setup_logging

    setup_logging()

    config = load_config(config_path)
    return Pipeline(config)


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

    # 디렉토리 구조 생성
    project_dir.mkdir(parents=True, exist_ok=True)
    documents_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 프로젝트명이 대체된 project.yaml 작성
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
) -> None:
    """전체 파이프라인을 실행합니다 (파싱 → 생성 → 검증 → 변환 → 훈련 → 내보내기)."""
    try:
        console.print(
            "\n[bold blue]slm-factory[/bold blue] — 전체 파이프라인 시작 중...\n"
        )

        pipeline = _load_pipeline(config)
        model_dir = pipeline.run()

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
) -> None:
    """파싱 + 생성 + 검증 + 품질 점수 평가를 실행합니다."""
    try:
        pipeline = _load_pipeline(config)
        pipeline.config.paths.ensure_dirs()

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
) -> None:
    """파싱 + 생성 + 검증 + 점수 평가 + 데이터 증강을 실행합니다."""
    try:
        pipeline = _load_pipeline(config)
        pipeline.config.paths.ensure_dirs()

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
) -> None:
    """파싱 + 생성 + 검증 + 점수 평가 + 증강 후 데이터 분석을 실행합니다."""
    try:
        pipeline = _load_pipeline(config)
        pipeline.config.paths.ensure_dirs()

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
def version() -> None:
    """slm-factory 버전을 표시합니다."""
    console.print(f"slm-factory [bold]{__version__}[/bold]")


# ---------------------------------------------------------------------------
# 진입점
# ---------------------------------------------------------------------------


def main() -> None:
    """pyproject.toml에서 호출되는 진입점입니다."""
    app()
