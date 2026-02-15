"""SLM Factory — 도메인 특화 소형 언어모델을 위한 Teacher-Student 지식 증류(Knowledge Distillation) 프레임워크."""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = [
    # 핵심
    "Pipeline",
    "SLMConfig",
    # 데이터 모델
    "ParsedDocument",
    "QAPair",
    "EvalResult",
    "CompareResult",
    "MultiTurnDialogue",
    "DialogueTurn",
    # 평가 / 비교
    "ModelEvaluator",
    "ModelComparator",
    # 내보내기
    "GGUFExporter",
    # 증분
    "IncrementalTracker",
    # 대화 생성
    "DialogueGenerator",
]


def __getattr__(name: str):
    """지연 임포트 — 실제로 사용할 때만 무거운 모듈을 로드합니다."""
    _imports: dict[str, tuple[str, str]] = {
        "Pipeline": (".pipeline", "Pipeline"),
        "SLMConfig": (".config", "SLMConfig"),
        "ParsedDocument": (".models", "ParsedDocument"),
        "QAPair": (".models", "QAPair"),
        "EvalResult": (".models", "EvalResult"),
        "CompareResult": (".models", "CompareResult"),
        "MultiTurnDialogue": (".models", "MultiTurnDialogue"),
        "DialogueTurn": (".models", "DialogueTurn"),
        "ModelEvaluator": (".evaluator", "ModelEvaluator"),
        "ModelComparator": (".comparator", "ModelComparator"),
        "GGUFExporter": (".exporter.gguf_export", "GGUFExporter"),
        "IncrementalTracker": (".incremental", "IncrementalTracker"),
        "DialogueGenerator": (".teacher.dialogue_generator", "DialogueGenerator"),
    }

    if name in _imports:
        module_path, attr = _imports[name]
        import importlib

        mod = importlib.import_module(module_path, __package__)
        val = getattr(mod, attr)
        globals()[name] = val  # 캐싱하여 다음 접근 시 __getattr__ 재호출 방지
        return val

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
