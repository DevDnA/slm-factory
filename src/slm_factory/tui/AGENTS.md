# tui/

Textual-based terminal UI components. Dashboard for pipeline monitoring and QA reviewer for manual curation.

## STRUCTURE

```
tui/
├── __init__.py     # Package marker (no exports)
├── dashboard.py    # PipelineDashboard — real-time pipeline progress TUI
├── reviewer.py     # QAReviewApp — approve/reject/edit QA pairs TUI
└── widgets.py      # Shared Textual widgets used by dashboard and reviewer
```

## HOW IT WORKS

1. `PipelineDashboard` displays real-time pipeline step progress, logs, and statistics.
2. `QAReviewApp` presents QA pairs for manual review — approve, reject, or edit each pair.
3. Both apps are launched from CLI commands (`tool dashboard`, `tool review`).
4. `widgets.py` contains reusable Textual widget components shared between apps.

## WHERE TO LOOK

| Task | File | Notes |
|------|------|-------|
| Modify dashboard layout | `dashboard.py` | Textual `compose()` method |
| Modify reviewer UI | `reviewer.py` | Textual `compose()` + keybinding handlers |
| Add shared widget | `widgets.py` | Subclass Textual `Widget` or `Static` |
| Change refresh rate | Via `DashboardConfig.refresh_interval` | Config-driven, default 2s |

## CONVENTIONS

- `textual` is an optional dependency — the `[tui]` extra installs it.
- TUI apps are standalone Textual `App` subclasses, launched via `app.run()`.
- Korean labels and status text throughout.
- Review results saved to `qa_reviewed.json` (path from `ReviewConfig.output_file`).

## ANTI-PATTERNS

- Do NOT import `textual` at package level — it's optional. Import inside functions.
- These modules are NOT used by the Pipeline — only CLI commands launch TUI apps.
