# ontology/

Document knowledge graph extraction module. Extracts entities and relations from parsed documents using Teacher LLM.

## STRUCTURE

```
ontology/
├── __init__.py       # Public API: Entity, Relation, KnowledgeGraph, OntologyExtractor, GraphStore
├── models.py         # Dataclasses: Entity, Relation, KnowledgeGraph (with context string + triple export)
├── graph_store.py    # GraphStore: JSON serialization + merge with deleted doc handling
└── extractor.py      # OntologyExtractor: LLM extraction + validation + normalization (follows QualityScorer pattern)
```

## HOW IT WORKS

1. `OntologyExtractor(teacher, config, teacher_config)` wraps Teacher LLM for entity/relation extraction.
2. `extract_all(docs)` processes all documents with bounded concurrency via `asyncio.Semaphore`.
3. Per document: `extract_one()` splits long docs into chunks via `chunk_document()`, extracts entities/relations per chunk, merges results. `_normalize_entities()` dedup by `(name.upper(), entity_type)`, `_normalize_relations()` dedup by `(subject.upper(), predicate.upper(), object.upper())`.
4. `GraphStore.save/load` handles JSON persistence. `GraphStore.merge` supports incremental updates with proper deleted document handling.
5. `KnowledgeGraph.to_context_string()` formats entities/relations for QA prompt injection.

## WHERE TO LOOK

| Task | File | Notes |
|------|------|-------|
| Change entity types | `config.py:OntologyConfig.entity_types` | Configurable list |
| Change extraction prompt | `extractor.py:_build_extraction_prompt()` | Korean instructions, JSON format |
| Change validation rules | `extractor.py:_validate_extraction()` | Type filtering + confidence threshold |
| Change entity normalization | `extractor.py:_normalize_entities()` | Upper-case dedup, prefer longer names |
| Change relation normalization | `extractor.py:_normalize_relations()` | Upper-case dedup, prefer higher confidence |
| Change storage format | `graph_store.py` | Currently JSON, could add SQLite |
| Change QA enrichment format | `models.py:to_context_string()` | Controls what goes into QA prompts |

## CONVENTIONS

- Follows QualityScorer pattern: constructor → `_build_*_prompt` → `_parse_*` → async one/all
- Data models are dataclasses (NOT Pydantic) — matches project convention
- Models live in `ontology/models.py`, NOT in shared `models.py` (isolation from 19+ module dependency)
- JSON as IPC: `json.dumps(data, ensure_ascii=False, indent=2)`
- Dedup keys: Entity=`(name.upper(), entity_type, source_doc)`, Relation=`(subject, predicate, object, source_doc)`
- `enrich_qa` defaults to `False` — user validates extraction quality first via `tool ontology`

## ANTI-PATTERNS

- Do NOT add ontology models to shared `models.py` — ripple risk to 19+ modules
- Do NOT add `step_extract_ontology` to `_STEP_ORDER` — it's a side product like `step_analyze`
- Do NOT import ML libs here — this module uses Teacher LLM only (no torch/transformers)
- Ollama callers MUST pass `format="json"` + `think=False` — without `think=False`, thinking models emit `<think>` tags that corrupt JSON output.
