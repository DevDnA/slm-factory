# teacher/

LLM abstraction layer. Factory pattern with ABC base class, 2 backends, and 2 generator modules.

## STRUCTURE

```
teacher/
├── __init__.py             # create_teacher() factory + exports
├── base.py                 # BaseTeacher ABC — generate/agenerate/health_check
├── ollama.py               # OllamaTeacher — httpx async, retry with backoff
├── openai_compat.py        # OpenAICompatTeacher — OpenAI-compatible API
├── qa_generator.py         # QAGenerator — document → QAPair via teacher LLM
└── dialogue_generator.py   # DialogueGenerator — QAPair → MultiTurnDialogue
```

## HOW IT WORKS

1. `create_teacher(config.teacher)` dispatches on `config.backend` → `OllamaTeacher` or `OpenAICompatTeacher`.
2. Both backends implement `generate(prompt)` (sync) and `agenerate(prompt)` (async with httpx).
3. `QAGenerator` takes parsed docs + config questions, calls teacher LLM per question per doc, parses JSON responses into `QAPair`.
4. `DialogueGenerator` expands QA pairs into multi-turn dialogues via teacher LLM.

## WHERE TO LOOK

| Task | File | Notes |
|------|------|-------|
| Add new LLM backend | New file + `__init__.py` | Subclass `BaseTeacher`, add to `create_teacher()` |
| Change QA prompt template | `qa_generator.py` | Alpaca-format JSON prompt with "Do NOT" guardrails |
| Change dialogue format | `dialogue_generator.py` | Multi-turn JSON prompt |
| Fix Ollama connectivity | `ollama.py` | `health_check()` + retry with exponential backoff |
| Adjust concurrency | Via `TeacherConfig.max_concurrency` | Controls `asyncio.Semaphore` in generators |

## CONVENTIONS

- Async-first: generators use `asyncio.run()` bridge called from Pipeline's sync `step_*` methods.
- Bounded concurrency: all async batch ops use `run_bounded(semaphore, coro, progress, task_id)` from `utils.py`.
- Empty LLM responses logged as `logger.warning()`, not raised.
- Retry pattern: both backends retry failed requests with exponential backoff.
- JSON response parsing: attempts `json.loads()` first, falls back to regex extraction on parse failure.

## ANTI-PATTERNS

- Do NOT call `agenerate()` without a semaphore — unbounded concurrency will overwhelm the LLM server.
- Do NOT import `torch`/`transformers` here — this package is LLM-client-only, no ML deps.
