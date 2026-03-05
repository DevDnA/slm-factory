# validator/

QA pair quality validation. Two strategies: rule-based filtering and embedding-based groundedness checking.

## STRUCTURE

```
validator/
├── __init__.py       # Re-exports: RuleValidator, ValidationResult, GroundednessChecker
├── rules.py          # RuleValidator — length, emptiness, dedup, pattern rejection
└── similarity.py     # GroundednessChecker — sentence-transformers cosine similarity
```

## HOW IT WORKS

1. `RuleValidator.validate_batch(pairs)` applies configurable rules: min/max answer length, empty Q/A removal, deduplication, regex pattern rejection. Returns `(accepted, rejected)` lists.
2. `GroundednessChecker.check_batch(pairs, source_texts)` computes cosine similarity between answer embeddings and source document chunks. Filters ungrounded answers below threshold.
3. Pipeline calls both in sequence: `step_validate()` → rules first, then groundedness (if enabled).

## WHERE TO LOOK

| Task | File | Notes |
|------|------|-------|
| Change rejection rules | `rules.py:RuleValidator.validate_one()` | Each rule appends reason to `ValidationResult.reasons` |
| Change similarity model | `similarity.py:GroundednessChecker.__init__()` | Uses `config.groundedness.model` |
| Change chunk strategy | `similarity.py:_chunk_text()` | Sliding window with overlap for long documents |
| Change similarity threshold | Via `GroundednessConfig.threshold` | Default 0.5, configurable in project.yaml |

## CONVENTIONS

- `RuleValidator` tracks seen pairs for deduplication via `_seen_pairs` set — call `reset_dedup()` between batches.
- `GroundednessChecker` lazy-loads sentence-transformers model on first `.score()` call.
- `sentence-transformers` is an optional dependency (`[validation]` extra) — `_check_sentence_transformers()` validates availability before use.
- `ValidationResult` is a simple dataclass: `passed: bool`, `reasons: list[str]`.

## ANTI-PATTERNS

- Do NOT import `sentence_transformers` at module level — it's an optional dependency.
- Do NOT call `check_batch()` without verifying sentence-transformers is installed first.
- Do NOT skip the rules validator — it catches obvious quality issues before the expensive embedding check.
