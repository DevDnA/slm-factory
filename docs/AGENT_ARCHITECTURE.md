# Agent RAG 아키텍처

oh-my-openagent (OMO) 디자인 패턴을 RAG Q&A 컨텍스트에 이식한 **똑똑한 agent 파이프라인**의 최종 구조 문서.

## 파이프라인 전경

```
사용자 질의
  │
  ▼
[Hook] pre_query        (query 정규화)
  │
  ▼
QueryRouter.route_async
  ├─ IntentClassifier (Phase 5)
  └─ 키워드 fallback
  │
  ▼ { intent, mode }
  │
  ├─ ambiguous + clarifier_enabled ──► Clarifier (Phase 11) ─► clarification 이벤트 ─► 종료
  ├─ simple mode ─────────────────────► 단순 RAG → 종료
  └─ agent mode
       │
       ▼
    PersonaRouter.select(intent)
      ├─ CustomPersona (Phase 14)     ← 우선
      └─ Built-in Persona (Phase 6)    ← Researcher/Comparator/Analyst/Procedural
       │
       ▼
    Planner (Phase 1c) ─► ExecutionPlan(strategy, steps)
       │
       ├─ plan.is_fallback + legacy_fallback_enabled ──► AgentLoop (ReAct, Phase 2)
       │
       ▼
    Plan step 실행
      ├─ parallel_steps + search×N ──► asyncio.gather (Phase 3-b)
      └─ 직렬 실행
         │
         ▼
       ToolRegistry.execute → ToolResult.sources → all_sources
       │
       ▼
    Verifier (Phase 1c) ─► 부족하면 repair search (max N)
       │
       ▼
    [Hook] post_search     (dedup, boosting)
       │
       ▼
    _stream_synthesis
      ├─ Persona's synthesis_prompt_template 우선
      ├─ Skills addon 주입 (Phase 7)
      ├─ 이전 턴 sources 주입 (Phase 3-a)
      └─ Model slot: synthesis_model (Phase 9)
       │
       ▼  { token stream }
       │
       ▼
    [Hook] post_synthesis  (HTML strip 등)
       │
       ▼
    Reflector (Phase 4) ─► 근거 약하면 보완 검색 + 재합성 (max retries)
       │
       ▼
    Review-Work 병렬 (Phase 8)
      ├─ GroundingChecker  ─┐
      ├─ CompletenessChecker├─► AggregatedVerdict
      └─ HallucinationChecker
       │  review_work_retry=true + 실패 → 보완 검색 + 재합성
       ▼
    Self-Improvement Loop (Phase 13)
      ├─ AnswerScorer (1-10) 
      └─ score < min_quality_score → feedback 주입 후 재합성 (max iterations)
       │
       ▼
    세션 기록 (user + assistant)
       │
       ▼
    Memory Compression (Phase 12)
      └─ 긴 세션은 오래된 턴을 LLM 요약으로 압축
       │
       ▼
    sources 이벤트 + done 이벤트
```

## 구성 요소 카탈로그

| 모듈 | 역할 | Phase |
|---|---|---|
| `agent/intent_classifier.py` | LLM 의도 분류 + TTL 캐시 | 5 |
| `agent/router.py` | intent + 키워드 라우팅 | 1a, 5 |
| `agent/orchestrator.py` | 전체 파이프라인 조율 | 1b~ |
| `agent/planner.py` | 질의→ExecutionPlan | 1c |
| `agent/verifier.py` | 컨텍스트 충분성 판정 | 1c |
| `agent/reflector.py` | 답변 자기 검증 | 4 |
| `agent/scorer.py` | 답변 1~10 정량 점수 | 13 |
| `agent/memory.py` | 대화 이력 요약 | 12 |
| `agent/hooks.py` | 파이프라인 lifecycle hook | 10 |
| `agent/personas/` | Researcher/Comparator/Analyst/Procedural/Clarifier | 6, 11 |
| `agent/persona_router.py` | intent→persona 매핑 | 6 |
| `agent/persona_loader.py` | YAML 기반 custom personas | 14 |
| `agent/reviewers/` | Grounding/Completeness/Hallucination | 8 |
| `agent/skills/` | 도메인 YAML 지식 팩 | 7 |
| `agent/state.py` | 파일 기반 세션 영속화 | 1a |
| `agent/session.py` | 인메모리 세션 관리 | 1a |
| `agent/loop.py` | ReAct legacy AgentLoop | 2 |

## Config 프리셋

### `smart_mode: true` — P0 (추천 기본)
활성화: IntentClassifier + Clarifier + Personas + Review-Work + Planner + Verifier + Reflector + Legacy fallback

특성:
- 답변 품질 + 의도 정확성 + 근거 검증
- LLM 호출 수: 질의당 약 6~10회
- 응답 시간: 기본 대비 2~3배

### `ultra_mode: true` — 모든 P0 + P1/P2
smart_mode + Hooks + Memory Compression + Self-Improvement + Review-Work retry + Session source reuse

특성:
- 최대 품질 · 대화 연속성 · 자동 개선
- LLM 호출 수: 질의당 10~15회 (retry 포함)
- 응답 시간: 기본 대비 3~5배

### 세부 opt-in 플래그 (P3)
- `skills_enabled` + `skills_dir` — 도메인 지식 팩
- `custom_personas_dir` — 사용자 정의 persona
- `parallel_steps` — 병렬 search (decompose 전략 전용)
- `persist_sessions` + `sessions_dir` — 세션 영속화
- `models.*_model` — 컴포넌트별 Ollama 모델 분리 (Phase 9)

## 이벤트 계약 (SSE)

모든 이벤트는 `data: {json}\n\n` 형식. 클라이언트는 다음 타입을 처리:

| 이벤트 | 언제 | 필드 |
|---|---|---|
| `route` | `/auto` 라우팅 직후 | `mode`, `intent?` |
| `clarification` | ambiguous → 역질문 | `questions[]`, `is_fallback` |
| `thought` | reasoning (stream_reasoning=True) | `content`, `iteration` |
| `action` | 도구 호출 직전 | `content`(tool), `input`, `iteration` |
| `observation` | 도구 결과 | `content` (truncated), `iteration` |
| `token` | synthesis 스트림 | `content` |
| `review` | Review-Work verdict | `reviewer`, `passed`, `reason` |
| `sources` | 최종 참조 문서 | `sources[]` |
| `done` | 종료 | `session_id?` |

## Never-Raise 계약

모든 LLM 호출 컴포넌트(Planner, Verifier, Reflector, Scorer, Reviewer, Clarifier, IntentClassifier, Compressor)는 **예외를 전파하지 않습니다**. 실패 시 안전한 기본값:

- Planner: 단일 search fallback plan
- Verifier: sufficient=True (중립)
- Reflector: answer_ok=True (통과)
- Scorer: 7.0/10 (중립)
- Reviewer: passed=True (통과)
- Clarifier: 일반적 fallback 질문
- IntentClassifier: ambiguous (Clarifier 트리거)
- Compressor: None (압축 스킵)

이 계약으로 LLM 장애가 agent 전체를 중단시키지 않습니다.

## 테스트 커버리지 (Phase 15b 완료 기준)

- 전체: **1097 passed**, 25 skipped
- 신규 테스트 (Phase 5~15b): **310개**
- 통합 테스트: `tests/test_rag_server_integration.py` (FastAPI TestClient)

## YAML 설정 예시

### smart_mode (권장)
```yaml
rag:
  agent:
    enabled: true
    smart_mode: true
```

### ultra_mode
```yaml
rag:
  agent:
    enabled: true
    ultra_mode: true
    # 선택적 확장
    skills_enabled: true
    skills_dir: /path/to/skills
    custom_personas_dir: /path/to/personas
    parallel_steps: true
    persist_sessions: true
    models:
      router_model: "qwen2.5:0.5b"
      planner_model: "qwen2.5:7b"
      synthesis_model: "qwen2.5:14b"
      verifier_model: "qwen2.5:1.5b"
```

### 개별 플래그 수동 제어
```yaml
rag:
  agent:
    enabled: true
    # P0 중 원하는 것만
    planner_enabled: true
    verifier_enabled: true
    reflector_enabled: true
    # intent + clarifier만
    intent_classifier_enabled: true
    clarifier_enabled: true
```

## 확장 지점

1. **새 persona 추가**: YAML에 정의하고 `custom_personas_dir`에 배치
2. **도메인 지식 추가**: YAML에 skill 정의, `skills_dir` 추가
3. **커스텀 Hook**: `orchestrator.register_hook(point, fn)` 호출
4. **새 모델 슬롯**: `AgentModelsConfig`에 필드 추가 + `_model_for(slot)` 대응

## 설계 원칙

1. **Opt-in 기본값**: 새 기능은 `*_enabled: false`로 시작, 기존 동작 무변경
2. **Never-raise 계약**: LLM 실패도 agent를 중단시키지 않음
3. **SSE 계약 보존**: 이벤트 타입 추가만, 기존 필드/순서 변경 금지
4. **테스트 우선**: 신규 모듈은 단위 + 통합 테스트 동시 작성
5. **설정 응집**: 모든 기능은 `rag.agent` config 아래 단일 네임스페이스
