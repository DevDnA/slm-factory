# 아키텍처 가이드

## 1. 설계 철학

SLM Factory는 다음 5가지 핵심 원칙을 기반으로 설계되었습니다.

### 1.1 YAML 중심 설정
모든 파이프라인 동작은 `project.yaml` 파일만으로 제어됩니다. 코드 수정 없이 문서 파싱 옵션, Teacher 모델 설정, 질문 카테고리, 검증 규칙, Student 모델, 학습 하이퍼파라미터, 내보내기 형식 등을 변경할 수 있습니다.

### 1.2 단계별 독립 실행
전체 파이프라인은 9개의 독립적인 단계(parse, generate, validate, score, augment, analyze, convert, train, export)로 구성됩니다. 각 단계는 CLI 명령어로 개별 실행 가능하며, 이전 단계의 출력 파일을 입력으로 사용합니다. 이를 통해 특정 단계만 재실행하거나 중간 결과를 검토할 수 있습니다.

### 1.3 지연 임포트(Lazy Import)
`torch`, `transformers`, `peft`, `sentence-transformers` 등 무거운 라이브러리는 실제 사용 시점에만 로드됩니다. CLI 초기 응답 속도를 빠르게 유지하고, 특정 단계만 실행할 때 불필요한 의존성을 로드하지 않습니다. 예를 들어 `parse` 명령어는 딥러닝 라이브러리를 전혀 로드하지 않습니다.

### 1.4 Pydantic v2 타입 안전 설정
모든 설정은 Pydantic v2 모델로 정의되어 YAML 로드 시 자동으로 타입 검증이 수행됩니다. 기본값, 필수 필드, 값 범위 제약이 명시적으로 정의되어 있어 오타나 잘못된 값이 입력되면 즉시 에러 메시지를 출력합니다. IDE 자동완성과 타입 힌트도 완벽하게 지원됩니다.

### 1.5 실패 격리
개별 파일 파싱 실패, 특정 문서의 QA 생성 실패, 단일 QA 쌍의 검증 실패 등은 전체 파이프라인을 중단시키지 않습니다. 실패한 항목은 로그에 기록되고 건너뛰며, 나머지 항목은 정상적으로 처리됩니다. 이를 통해 대량의 문서를 처리할 때 일부 문제가 전체 작업을 방해하지 않습니다.

## 2. 전체 아키텍처 다이어그램

### 2.1 컴포넌트 구조

```
                     ┌─────────────────────────────┐
                       CLI (cli.py)
                       Entry Point (Typer)
                     └──────────────┬──────────────┘
                                    ▼
                     ┌─────────────────────────────┐
                       Pipeline (pipeline.py)
                       Orchestrator (9 steps)
                     └──────────────┬──────────────┘
                                    ▼

┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
  1. Parse             2. Generate          3. Validate
  parsers/             teacher/             validator/
└────────┬─────────┘ └────────┬─────────┘ └────────┬─────────┘
         ▼                    ▼                    ▼
  parsed_documents.json qa_alpaca.json       (filtered pairs)

┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
  3a. Score            3b. Augment          3c. Analyze
  scorer.py            augmenter.py         analyzer.py
└────────┬─────────┘ └────────┬─────────┘ └────────┬─────────┘
         ▼                    ▼                    ▼
  qa_scored.json       qa_augmented.json   data_analysis.json

┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
  4. Convert           5. Train             6. Export
  converter.py         trainer/             exporter/
└────────┬─────────┘ └────────┬─────────┘ └────────┬─────────┘
         ▼                    ▼                    ▼
  training_data.jsonl  adapter/            merged_model/ Modelfile

                     ┌─────────────────────────────┐
                       Config (config.py)
                       Injected into all components
                     └─────────────────────────────┘
```

### 2.2 모듈 의존성

```
cli.py
  └─→ pipeline.py
        ├─→ parsers/
        │     ├─→ base.py (BaseParser, ParserRegistry)
        │     ├─→ pdf.py (PDFParser)
        │     ├─→ hwpx.py (HWPXParser)
        │     ├─→ html.py (HTMLParser)
        │     ├─→ text.py (TextParser)
        │     └─→ docx.py (DOCXParser)
        │
        ├─→ teacher/
        │     ├─→ base.py (BaseTeacher)
        │     ├─→ ollama.py (OllamaTeacher)
        │     ├─→ openai_compat.py (OpenAICompatTeacher)
        │     ├─→ qa_generator.py (QAGenerator)
        │     └─→ dialogue_generator.py (DialogueGenerator)
        │
        ├─→ validator/
        │     ├─→ rules.py (RuleValidator)
        │     └─→ similarity.py (GroundednessChecker)
        │
        ├─→ scorer.py (QualityScorer)
        │
        ├─→ augmenter.py (DataAugmenter)
        │
        ├─→ analyzer.py (DataAnalyzer)
        │
        ├─→ converter.py (ChatFormatter)
        │
        ├─→ trainer/
        │     └─→ lora_trainer.py (DataLoader, LoRATrainer)
        │
        ├─→ exporter/
        │     ├─→ hf_export.py (HFExporter)
        │     ├─→ ollama_export.py (OllamaExporter)
        │     └─→ gguf_export.py (GGUFExporter)
        │
        ├─→ evaluator.py (ModelEvaluator)
        ├─→ comparator.py (ModelComparator)
        ├─→ incremental.py (IncrementalManager)
        └─→ tui/ (ReviewApp, DashboardApp)

All modules
  ├─→ config.py (SLMConfig + 26 하위 모델)
  ├─→ models.py (ParsedDocument, QAPair, EvalResult, DialogueTurn, MultiTurnDialogue, CompareResult)
  └─→ utils.py (setup_logging, get_logger)
```

## 3. 핵심 설계 패턴

### 3.1 Registry 패턴 (parsers/)

파서 시스템은 Registry 패턴을 사용하여 확장 가능한 파일 형식 지원을 제공합니다.

**핵심 구조:**
- `ParserRegistry` 클래스: `BaseParser` 인스턴스 목록을 저장
- `register()` 메서드: 파서 클래스를 인스턴스화하여 목록에 추가 (데코레이터로 사용 가능)
- `get_parser(path)`: 등록된 파서를 순회하며 `can_parse(path)`를 만족하는 첫 번째 파서 반환
- `parse_directory()`: 디렉토리를 스캔하고 파일별로 파서를 선택하여 파싱, Rich 진행 표시줄 표시, 실패 격리

**등록 메커니즘:**
`parsers/__init__.py`에서 전역 `registry` 인스턴스를 생성하고 `PDFParser`, `HWPXParser`, `HTMLParser`, `TextParser`, `DOCXParser`를 자동 등록합니다.

**BaseParser 추상 클래스:**
- `extensions` ClassVar: 지원하는 파일 확장자 목록
- `parse()` 추상 메서드: 파일 경로를 받아 `ParsedDocument` 반환
- `can_parse()`: 파일 확장자가 `extensions`에 포함되는지 확인

**커스텀 파서 추가 예제:**

다음은 커스텀 파서 추가 예제입니다 (참고: DOCX 파서는 이미 내장되어 있습니다):

```python
from slm_factory.parsers.base import BaseParser
from slm_factory.models import ParsedDocument
from slm_factory.parsers import registry

@registry.register
class CSVParser(BaseParser):
    """CSV 파일 파서"""
    extensions = [".csv"]
    
    def parse(self, path: Path) -> ParsedDocument:
        import csv
        
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        # CSV를 텍스트로 변환
        content = "\n".join([str(row) for row in rows])
        
        return ParsedDocument(
            doc_id=path.name,
            title=path.stem,
            content=content,
            metadata={"rows": len(rows)}
        )
```

등록 후 자동으로 `.csv` 파일이 파이프라인에서 처리됩니다.

### 3.2 Factory 패턴 (teacher/)

Teacher 백엔드 생성은 Factory 패턴을 사용하여 설정 기반으로 적절한 구현체를 반환합니다.

**핵심 구조:**
- `create_teacher(config: TeacherConfig) → BaseTeacher`: 팩토리 함수
- `config.backend` 값에 따라 분기:
  - `"ollama"` → `OllamaTeacher` 인스턴스 반환
  - `"openai"` → `OpenAICompatTeacher` 인스턴스 반환
  - 기타 → `ValueError` 발생

**BaseTeacher 추상 클래스:**
- `generate(prompt: str, **kwargs) → str`: 프롬프트를 받아 응답 생성
- `health_check() → bool`: 백엔드 연결 상태 확인

**커스텀 백엔드 추가 예제:**

```python
# teacher/anthropic.py
from slm_factory.teacher.base import BaseTeacher

class AnthropicTeacher(BaseTeacher):
    def __init__(self, config):
        import anthropic
        self.client = anthropic.Anthropic(api_key=config.api_key)
        self.model = config.model
    
    def generate(self, prompt: str, **kwargs) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=kwargs.get("max_tokens", 2048),
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    
    def health_check(self) -> bool:
        try:
            self.generate("test", max_tokens=1)
            return True
        except Exception:
            return False

# teacher/__init__.py에 추가
def create_teacher(config: TeacherConfig) -> BaseTeacher:
    if config.backend == "ollama":
        from .ollama import OllamaTeacher
        return OllamaTeacher(config)
    elif config.backend == "openai":
        from .openai_compat import OpenAICompatTeacher
        return OpenAICompatTeacher(config)
    elif config.backend == "anthropic":
        from .anthropic import AnthropicTeacher
        return AnthropicTeacher(config)
    else:
        raise ValueError(f"Unknown backend: {config.backend}")
```

### 3.3 Strategy 패턴 (validator/)

검증 시스템은 Strategy 패턴을 사용하여 여러 검증 전략을 조합합니다.

**핵심 구조:**
- `RuleValidator`: 4개의 규칙을 순서대로 적용하는 체인 (empty → length → pattern → dedup)
- `GroundednessChecker`: 임베딩 기반 코사인 유사도 검증 (sentence-transformers)
- `pipeline.py`의 `step_validate()`: 규칙 검증 먼저 수행, 이후 선택적으로 근거성 검증 수행
- 두 검증기 모두 `(accepted, rejected)` 튜플 반환

**RuleValidator 체인:**
1. **Empty 체크**: 질문 또는 답변이 비어있으면 거부
2. **Length 체크**: 답변 길이가 `min_answer_length`(기본 20) 미만 또는 `max_answer_length`(기본 2000) 초과 시 거부
3. **Pattern 체크**: 3개의 정규식 패턴에 매칭되면 거부 (기본: "i don't know", "not (available|provided|mentioned|found)", "the document does not contain")
4. **Deduplication**: 이미 처리된 질문-답변 쌍과 중복되면 거부

**GroundednessChecker 전략:**
- 문서를 512자 청크로 분할 (64자 오버랩)
- 각 청크와 답변을 임베딩으로 변환
- 코사인 유사도 계산 후 임계값(기본 0.3) 이상인 청크가 있으면 통과

**조합 예제:**

```python
# pipeline.py step_validate()
validator = RuleValidator(self.config.validation)
accepted, rejected = validator.validate_batch(pairs)

if self.config.validation.groundedness.enabled and docs is not None:
    checker = GroundednessChecker(self.config.validation.groundedness)
    source_texts = {doc.doc_id: doc.content for doc in docs}
    accepted, ungrounded = checker.check_batch(accepted, source_texts)
```

### 3.4 Adapter 패턴 (converter.py)

채팅 형식 변환은 Adapter 패턴을 사용하여 다양한 모델의 템플릿을 통일된 인터페이스로 처리합니다.

**핵심 구조:**
- `ChatFormatter`: HuggingFace `tokenizer.apply_chat_template()`을 래핑
- 통일된 `QAPair` 형식을 모델별 채팅 형식으로 변환 (Gemma, Llama, Mistral, Phi 등 각기 다른 템플릿)
- 자동 폴백: 시스템 역할 실패 시 (Gemma 등) 시스템 메시지 없이 재시도
- 토큰 길이 필터링 (`max_seq_length` 초과 시 제외)

**변환 프로세스:**

```python
# 1. QAPair → 메시지 리스트 구성 (시스템 메시지는 조건부)
messages = []
if self.system_prompt:
    messages.append({"role": "system", "content": self.system_prompt})
messages.append({"role": "user", "content": pair.question})
messages.append({"role": "assistant", "content": pair.answer})

# 2. 토크나이저 템플릿 적용 (모델별 자동 처리)
try:
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
except Exception:
    # 3. 시스템 역할 실패 시 폴백
    messages = messages[1:]  # 시스템 메시지 제거
    text = tokenizer.apply_chat_template(messages, ...)

# 4. 토큰 길이 체크
tokens = tokenizer.encode(text)
if len(tokens) <= max_seq_length:
    return {"text": text}
```

**모델별 출력 예제:**

Llama 형식:
```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
당신은 도움이 되는 AI 어시스턴트입니다.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
질문 내용<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
답변 내용<|eot_id|>
```

Gemma 형식 (시스템 역할 미지원):
```
<start_of_turn>user
질문 내용<end_of_turn>
<start_of_turn>model
답변 내용<end_of_turn>
```

### 3.5 비동기 동시성 패턴 (scorer.py, augmenter.py)

Scorer와 Augmenter는 asyncio.Semaphore를 사용한 비동기 동시성 패턴을 공유합니다.

- asyncio.Semaphore로 max_concurrency 제한
- asyncio.gather()로 병렬 처리
- teacher.agenerate() 비동기 호출
- 예외 격리: return_exceptions=True로 개별 실패 허용

코드 패턴:
```python
semaphore = asyncio.Semaphore(config.max_concurrency)

async def _bounded_task(item):
    async with semaphore:
        return await process(item)

tasks = [_bounded_task(item) for item in items]
results = await asyncio.gather(*tasks, return_exceptions=True)
```

또한 scorer, augmenter, qa_generator에 Rich Progress 바가 통합되어 대량 처리 시 실시간 진행 상황을 표시합니다.

### 3.6 통계 분석 패턴 (analyzer.py)

Analyzer는 LLM 의존성 없이 순수 통계 분석을 수행합니다.

- Counter로 분포 계산
- statistics 모듈로 기초 통계
- 자동 경고 생성 (불균형, 이상치 등)
- Rich 콘솔 시각적 보고서

## 4. 데이터 흐름 상세

### 4.1 Step 1: parse

**입력:**
- `documents/` 디렉토리 내 파일들 (PDF, HWPX, HTML, TXT, DOCX 등)

**처리 로직:**
1. `ParserRegistry.parse_directory()` 호출
2. 디렉토리 내 최상위 파일을 스캔 (재귀 없음)
3. 각 파일에 대해 `get_parser(path)`로 적절한 파서 선택
4. 파서의 `parse()` 메서드 호출하여 `ParsedDocument` 생성
5. Rich Progress 진행 표시줄로 진행 상황 표시
6. 개별 파일 실패 시 로그 기록 후 건너뜀 (전체 프로세스 계속)

**출력:**
- 타입: `list[ParsedDocument]`
- 저장 파일: `parsed_documents.json`
- 구조:
  ```json
  [
    {
      "doc_id": "document1.pdf",
      "title": "문서 제목",
      "content": "추출된 텍스트 내용...",
      "tables": ["| 헤더1 | 헤더2 |\n| --- | --- |\n| 값1 | 값2 |"],
      "metadata": {"pages": 10, "author": "작성자"}
    }
  ]
  ```

**에러 처리:**
- 개별 파일 파싱 실패: `logger.exception()` 호출 후 해당 파일 건너뜀
- 전체 디렉토리 접근 실패: 예외 발생 및 파이프라인 중단
- 지원되지 않는 파일 형식: 경고 로그 출력 후 건너뜀

### 4.2 Step 2: generate

**입력:**
- `list[ParsedDocument]` (parsed_documents.json에서 로드)

**처리 로직:**
1. `QAGenerator.generate_all_async()` 호출
2. 각 문서 × 각 질문 카테고리에 대해 반복:
   - `build_prompt()`: 프롬프트 구성
     - 시스템 지시사항 (JSON 형식 요구)
     - 문서 제목 및 내용 (max_context_chars 12000자로 제한)
     - 테이블 데이터 (있는 경우)
     - 질문 카테고리 설명
     - JSON 형식 예제
   - `teacher.generate()`: Teacher 모델에 프롬프트 전송
   - `parse_response()`: JSON 응답 파싱
3. 응답 파싱 시 4가지 JSON 형태 처리:
   - 직접 객체: `{"question": "...", "answer": "..."}`
   - 배열: `[{"question": "...", "answer": "..."}]`
   - "data" 래핑: `{"data": [...]}`
   - "items" 래핑: `{"items": [...]}`
4. 키 정규화: `question`/`instruction` → `question`, `answer`/`output` → `answer`

**프롬프트 구조 예제:**

```
당신은 교육 자료를 기반으로 고품질 질문-답변 쌍을 생성하는 AI입니다.

문서 제목: 파이썬 기초
문서 내용: 파이썬은 간결하고 읽기 쉬운 문법을 가진 프로그래밍 언어입니다...

질문 유형: 개념 설명
다음 형식의 JSON으로 3개의 질문-답변 쌍을 생성하세요:
[
  {"question": "...", "answer": "..."},
  ...
]
```

**출력:**
- 타입: `list[QAPair]`
- 저장 파일: `qa_alpaca.json`
- 구조:
  ```json
  [
    {
      "question": "파이썬의 주요 특징은 무엇인가요?",
      "answer": "파이썬은 간결하고 읽기 쉬운 문법을 가진 프로그래밍 언어입니다...",
      "source_doc": "document1.pdf",
      "category": "개념 설명"
    }
  ]
  ```

**에러 처리:**
- 개별 문서/질문 생성 실패: 로그 기록 후 건너뜀
- JSON 파싱 실패: 원본 응답 로그 출력 후 건너뜀
- Teacher 백엔드 연결 실패: `RuntimeError` 발생 및 파이프라인 중단
- 타임아웃: httpx TimeoutException 처리 후 재시도 또는 건너뜀

### 4.3 Step 3: validate

**입력:**
- `list[QAPair]` (qa_alpaca.json에서 로드)
- `list[ParsedDocument]` (근거성 검증 시 필요, 선택적)

**처리 로직:**

**1단계: RuleValidator.validate_batch()**
- 4개 규칙을 순서대로 적용:

  **규칙 1 - Empty 체크:**
  ```python
  if not qa.question.strip() or not qa.answer.strip():
      reject("empty_field")
  ```

  **규칙 2 - Length 체크:**
  ```python
  if len(qa.answer) < min_answer_length:  # 기본 20
      reject("answer_too_short")
  if len(qa.answer) > max_answer_length:  # 기본 2000
      reject("answer_too_long")
  ```

  **규칙 3 - Pattern 체크:**
  ```python
   reject_patterns = [
       r"(?i)i don't know",
       r"(?i)not (available|provided|mentioned|found)",
       r"(?i)the document does not contain",
   ]
   for pattern in self._compiled_patterns:
       if pattern.search(pair.answer):
           reasons.append(f"matched_reject_pattern: {pattern.pattern}")
  ```

  **규칙 4 - Deduplication:**
  ```python
   pair_key = f"{pair.question.strip().lower()}|{pair.answer.strip().lower()}"
   if pair_key in self._seen_pairs:
       reasons.append("duplicate")
   else:
       self._seen_pairs.add(pair_key)
  ```

**2단계: GroundednessChecker.check_batch() (선택적)**
- 문서 청킹:
  ```python
  def _chunk_text(text, chunk_size=512, overlap=64):
      chunks = []
      for i in range(0, len(text), chunk_size - overlap):
          chunks.append(text[i:i + chunk_size])
      return chunks
  ```
- 임베딩 생성 및 유사도 계산:
  ```python
  doc_chunks = _chunk_text(document.content)
  doc_embeddings = model.encode(doc_chunks)
  answer_embedding = model.encode([qa.answer])
  
  similarities = cosine_similarity(answer_embedding, doc_embeddings)
  max_similarity = similarities.max()
  
  if max_similarity < threshold:  # 기본 0.3
      reject("low_groundedness")
  ```

**출력:**
- 타입: `list[QAPair]` (필터링됨)
- 파일 저장 없음 (다음 단계로 직접 전달)

**에러 처리:**
- `ValidationResult.reasons`: 각 QA 쌍의 거부 사유를 리스트로 추적
- 모델 로드 실패 (GroundednessChecker): 예외 발생 및 파이프라인 중단
- 개별 QA 검증 실패: 로그 기록 후 거부 목록에 추가

### 4.4 Step 3a: score

**입력:** list[QAPair] (검증 통과한 데이터)

**처리 로직:**
1. scoring.enabled 체크 (false면 건너뜀)
2. create_teacher()로 Teacher 인스턴스 생성
3. QualityScorer 초기화
4. asyncio.run(scorer.score_all(pairs)) 호출
5. 각 QA 쌍에 1~5점 부여
6. threshold(기본 3.0) 미만 제거

**출력:**
- 타입: `list[QAPair]` (필터링됨)
- 저장 파일: `qa_scored.json` (중간 저장, --resume으로 재개 가능)

**에러 처리:** 점수 파싱 실패 시 기본값 3점 적용, 개별 평가 실패는 로그 후 건너뜀

### 4.5 Step 3b: augment

**입력:** list[QAPair] (점수 통과한 데이터)

**처리 로직:**
1. augment.enabled 체크
2. create_teacher()로 Teacher 인스턴스 생성
3. DataAugmenter 초기화
4. asyncio.run(augmenter.augment_all(pairs)) 호출
5. 원본 질문을 num_variants(기본 2)개 패러프레이즈
6. 증강된 QAPair에 is_augmented=True 설정

**출력:**
- 타입: `list[QAPair]` (원본 + 증강)
- 저장 파일: `qa_augmented.json` (중간 저장, --resume으로 재개 가능)

**에러 처리:** 개별 패러프레이즈 실패는 로그 후 건너뜀

### 4.6 Step 3c: analyze

**입력:** list[QAPair] (최종 데이터)

**처리 로직:**
1. analyzer.enabled 체크
2. DataAnalyzer 초기화 (LLM 불필요)
3. 통계 분석: 전체/원본/증강 카운트, 카테고리/문서 분포, 길이 통계
4. 경고 생성: 데이터 불균형, 카테고리 부족, 길이 이상치, 데이터 부족
5. Rich 콘솔 보고서 출력
6. JSON 보고서 저장

**출력:** data_analysis.json (보고서 파일)

**에러 처리:** 분석 실패 시 경고 로그 출력 후 파이프라인 계속

### 4.7 Step 4: convert

**입력:**
- `list[QAPair]` (검증 통과한 데이터)

**처리 로직:**

**1단계: converter.py의 ChatFormatter.format_batch()**
- 각 QA 쌍에 대해 `build_messages()` 호출:
   ```python
   def build_messages(self, pair: QAPair) -> list[dict[str, str]]:
      messages = []
      if self.system_prompt:
          messages.append({"role": "system", "content": self.system_prompt})
      messages.append({"role": "user", "content": pair.question})
      messages.append({"role": "assistant", "content": pair.answer})
      return messages
  ```

**2단계: apply_chat_template()**
- 토크나이저의 채팅 템플릿 적용 (모델별 자동 처리):
  ```python
  try:
      text = tokenizer.apply_chat_template(
          messages,
          tokenize=False,
          add_generation_prompt=False
      )
  except Exception as e:
      # 시스템 역할 미지원 모델 (Gemma 등) 폴백
      logger.warning(f"System role failed, retrying without: {e}")
      messages = messages[1:]  # 시스템 메시지 제거
      text = tokenizer.apply_chat_template(messages, ...)
  ```

**3단계: 토큰 길이 체크**
- 토큰화 후 길이 확인:
  ```python
  tokens = tokenizer.encode(text)
  if len(tokens) > max_seq_length:
      logger.debug(f"Skipping QA (tokens={len(tokens)} > {max_seq_length})")
      continue
  ```

**출력:**
- 타입: JSONL 파일 (각 줄이 하나의 JSON 객체)
- 저장 파일: `training_data.jsonl`
- 구조:
  ```jsonl
  {"text": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n당신은...<|eot_id|>..."}
  {"text": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n당신은...<|eot_id|>..."}
  ```

**에러 처리:**
- 템플릿 적용 실패: 시스템 메시지 제거 후 재시도
- 재시도 실패: 로그 기록 후 해당 QA 건너뜀
- 토큰 길이 초과: 디버그 로그 출력 후 건너뜀
- 토크나이저 로드 실패: 예외 발생 및 파이프라인 중단

convert와 export 단계는 CLI에서 단독 실행할 수 있습니다 (`slm-factory tool convert`, `slm-factory export`). 이를 통해 전체 파이프라인을 재실행하지 않고 특정 단계만 반복할 수 있습니다.

### 4.8 Step 5: train

**입력:**
- `training_data.jsonl` 파일 경로

**처리 로직:**

**1단계: trainer/lora_trainer.py의 DataLoader.load_and_split()**
- JSONL 파일 로드 및 train/eval 분할:
  ```python
  with open(path) as f:
      data = [json.loads(line) for line in f]
  
  split_idx = int(len(data) * (1 - eval_ratio))
  train_data = data[:split_idx]
  eval_data = data[split_idx:]
  
  return Dataset.from_list(train_data), Dataset.from_list(eval_data)
  ```

**2단계: LoRATrainer._load_model()**
- 모델 및 토크나이저 로드:
  ```python
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  
  # 양자화 설정 (선택적)
  if quantization_enabled:
      bnb_config = BitsAndBytesConfig(
          load_in_4bit=True,
          bnb_4bit_quant_type="nf4",
          bnb_4bit_compute_dtype=torch.bfloat16
      )
  
  model = AutoModelForCausalLM.from_pretrained(
      model_name,
      quantization_config=bnb_config if quantization_enabled else None,
      device_map="auto"
  )
  ```

**3단계: LoRATrainer._create_lora_config()**
- LoRA 설정 생성:
  ```python
  from peft import LoraConfig, TaskType
  
  lora_config = LoraConfig(
      task_type=TaskType.CAUSAL_LM,
      r=lora_r,              # 기본 16
      lora_alpha=lora_alpha,  # 기본 32
      lora_dropout=lora_dropout,  # 기본 0.05
      target_modules=target_modules,  # ["q_proj", "v_proj"] 등
      bias="none"
  )
  ```

**4단계: get_peft_model()**
- LoRA 어댑터 적용:
  ```python
  from peft import get_peft_model
  
  model = get_peft_model(model, lora_config)
  model.print_trainable_parameters()
  # 출력 예: trainable params: 4,194,304 || all params: 6,738,415,616 || trainable%: 0.06
  ```

**5단계: SFTTrainer 설정 및 학습**
- TrainingArguments 및 콜백 설정:
  ```python
  from trl import SFTTrainer
  from transformers import TrainingArguments, EarlyStoppingCallback
  
  training_args = TrainingArguments(
      output_dir=output_dir,
      num_train_epochs=num_epochs,
      per_device_train_batch_size=batch_size,
      learning_rate=learning_rate,
      logging_steps=logging_steps,
      save_strategy="epoch",
      evaluation_strategy="epoch",
      load_best_model_at_end=True
  )
  
  callbacks = [
      EarlyStoppingCallback(
          early_stopping_patience=patience,
          early_stopping_threshold=threshold
      )
  ]
  
  trainer = SFTTrainer(
      model=model,
      args=training_args,
      train_dataset=train_dataset,
      eval_dataset=eval_dataset,
      callbacks=callbacks,
      dataset_text_field="text"
  )
  
  trainer.train()
  ```

**출력:**
- 디렉토리: `checkpoints/adapter/`
- 내용:
  - `adapter_config.json`: LoRA 설정
  - `adapter_model.safetensors`: 어댑터 가중치
  - `tokenizer_config.json`, `tokenizer.json`: 토크나이저 파일
  - `training_args.bin`: 학습 인자

**에러 처리:**
- CUDA 메모리 부족: 예외 발생 및 배치 크기 감소 권장 메시지 출력
- 체크포인트 저장 실패: 로그 기록 후 학습 계속
- Early stopping 트리거: 정상 종료로 처리
- HuggingFace 라이브러리 에러: 원본 예외 전파

### 4.9 Step 6: export

**입력:**
- `adapter_path`: LoRA 어댑터 디렉토리 경로

**처리 로직:**

**HFExporter (merge_lora=True 시):**
1. 베이스 모델 및 어댑터 로드:
   ```python
   from peft import PeftModel
   
   base_model = AutoModelForCausalLM.from_pretrained(
       base_model_name,
       device_map="auto",
       torch_dtype=torch.bfloat16
   )
   
   model = PeftModel.from_pretrained(base_model, adapter_path)
   ```

2. 어댑터 병합 및 저장:
   ```python
   merged_model = model.merge_and_unload()
   
   merged_model.save_pretrained(
       output_path,
       safe_serialization=True  # safetensors 형식
   )
   
   tokenizer.save_pretrained(output_path)
   ```

**OllamaExporter:**
1. Modelfile 생성:
   ```python
   modelfile_content = f"""FROM {base_model_path}
   ADAPTER {adapter_path}/adapter_model.safetensors
   
   SYSTEM {system_prompt}
   
   PARAMETER temperature {temperature}
   PARAMETER top_p {top_p}
   PARAMETER top_k {top_k}
   """
   
   with open(output_path / "Modelfile", "w") as f:
       f.write(modelfile_content)
   ```

2. Ollama 모델 생성:
   ```python
   import subprocess
   
   subprocess.run(
       ["ollama", "create", model_name, "-f", "Modelfile"],
       cwd=output_path,
       check=True
   )
   ```

**출력:**

**HFExporter:**
- 디렉토리: `merged_model/`
- 내용:
  - `model.safetensors` 또는 `model-00001-of-00002.safetensors` 등
  - `config.json`: 모델 설정
  - `tokenizer_config.json`, `tokenizer.json`: 토크나이저

**OllamaExporter:**
- 파일: `Modelfile`
- Ollama 레지스트리에 모델 등록 (로컬)

**에러 처리:**
- 어댑터 로드 실패: 경로 확인 메시지 출력 후 예외 발생
- 병합 중 메모리 부족: 예외 발생 및 권장 사항 출력
- Ollama CLI 미설치: `FileNotFoundError` 처리 및 설치 안내
- Modelfile 생성 실패: 권한 확인 메시지 출력

## 5. 설정 시스템 아키텍처

### 5.1 설정 모델 계층 구조

```
SLMConfig (root)
├── project: ProjectConfig
│   ├── name: str = "my-project"
│   ├── version: str = "1.0.0"
│   └── language: str = "en"
│
├── paths: PathsConfig
│   ├── documents: Path = "./documents"
│   └── output: Path = "./output"
│
├── parsing: ParsingConfig
│   ├── formats: list[str] = ["pdf", "txt", "html"]
│   ├── pdf: PdfOptions
│   │   └── extract_tables: bool = True
│   └── hwpx: HwpxOptions
│       └── apply_spacing: bool = True
│
├── teacher: TeacherConfig
│   ├── backend: Literal["ollama", "openai"] = "ollama"
│   ├── model: str = "qwen3:8b"
│   ├── api_base: str = "http://localhost:11434"
│   ├── api_key: str | None = None
│   ├── temperature: float = 0.3
│   ├── timeout: int = 180
│   ├── max_context_chars: int = 12000
│   └── max_concurrency: int = 4
│
├── questions: QuestionsConfig
│   ├── categories: dict[str, list[str]] = {}
│   ├── file: Path | None = None
│   ├── system_prompt: str = "You are a helpful..."
│   └── output_format: str = "alpaca"
│
├── validation: ValidationConfig
│   ├── enabled: bool = True
│   ├── min_answer_length: int = 20
│   ├── max_answer_length: int = 2000
│   ├── remove_empty: bool = True
│   ├── deduplicate: bool = True
│   ├── reject_patterns: list[str] = [...]
│   └── groundedness: GroundednessConfig
│       ├── enabled: bool = False
│       ├── model: str = "all-MiniLM-L6-v2"
│       └── threshold: float = 0.3
│
├── scoring: ScoringConfig
│   ├── enabled: bool = False
│   ├── threshold: float = 3.0
│   └── max_concurrency: int = 4
│
├── augment: AugmentConfig
│   ├── enabled: bool = False
│   ├── num_variants: int = 2
│   └── max_concurrency: int = 4
│
├── analyzer: AnalyzerConfig
│   ├── enabled: bool = True
│   └── output_file: str = "data_analysis.json"
│
├── student: StudentConfig
│   ├── model: str = "google/gemma-3-1b-it"
│   └── max_seq_length: int = 4096
│
├── training: TrainingConfig
│   ├── batch_size: int = 4
│   ├── gradient_accumulation_steps: int = 4
│   ├── learning_rate: float = 2e-5
│   ├── lr_scheduler: str = "cosine"
│   ├── warmup_ratio: float = 0.1
│   ├── num_epochs: int = 20
│   ├── optimizer: str = "adamw_torch_fused"
│   ├── bf16: bool = True
│   ├── train_split: float = 0.9
│   ├── save_strategy: str = "epoch"
│   ├── lora: LoraConfig
│   │   ├── r: int = 16
│   │   ├── alpha: int = 32
│   │   ├── dropout: float = 0.05
│   │   ├── target_modules: str | list[str] = "auto"
│   │   └── use_rslora: bool = False
│   ├── early_stopping: EarlyStoppingConfig
│   │   ├── enabled: bool = True
│   │   ├── patience: int = 3
│   │   └── threshold: float = 0.01
│   └── quantization: QuantizationConfig
│       ├── enabled: bool = False
│       └── bits: int = 4
│
├── export: ExportConfig
│   ├── merge_lora: bool = True
│   ├── output_format: str = "safetensors"
│   └── ollama: OllamaExportConfig
│       ├── enabled: bool = True
│       ├── model_name: str = "my-project-model"
│       ├── system_prompt: str = "You are a helpful..."
│       └── parameters: dict = {temperature: 0.7, top_p: 0.9, num_ctx: 4096}
│
├── eval: EvalConfig
│   ├── enabled: bool = False
│   ├── test_split: float = 0.1
│   ├── metrics: list[str] = ["bleu", "rouge"]
│   ├── max_samples: int = 50
│   └── output_file: str = "eval_results.json"
│
├── gguf_export: GGUFExportConfig
│   ├── enabled: bool = False
│   ├── quantization_type: str = "q4_k_m"
│   └── llama_cpp_path: str = ""
│
├── incremental: IncrementalConfig
│   ├── enabled: bool = False
│   ├── hash_file: str = "document_hashes.json"
│   ├── merge_strategy: Literal["append", "replace"] = "append"
│   └── resume_adapter: str = ""
│
├── dialogue: DialogueConfig
│   ├── enabled: bool = False
│   ├── min_turns: int = 2
│   ├── max_turns: int = 5
│   └── include_single_qa: bool = True
│
├── review: ReviewConfig
│   ├── enabled: bool = False
│   ├── auto_open: bool = True
│   └── output_file: str = "qa_reviewed.json"
│
├── compare: CompareConfig
│   ├── enabled: bool = False
│   ├── base_model: str = ""
│   ├── finetuned_model: str = ""
│   ├── metrics: list[str] = ["bleu", "rouge"]
│   ├── max_samples: int = 20
│   └── output_file: str = "compare_results.json"
│
└── dashboard: DashboardConfig
    ├── enabled: bool = False
    ├── refresh_interval: float = 2.0
    └── theme: str = "dark"
```

### 5.2 설정 로드 프로세스

**load_config() 흐름:**

```python
def load_config(path: str | Path) -> SLMConfig:
    # 1. 경로 확인
    filepath = Path(path).resolve()
    if not filepath.is_file():
        raise FileNotFoundError(f"Config file not found: {filepath}")
    
    # 2. YAML 파일 읽기 + Pydantic 검증
    raw = yaml.safe_load(filepath.read_text(encoding="utf-8")) or {}
    config = SLMConfig.model_validate(raw)
    
    # 3. 상대 경로를 설정 파일 기준 절대 경로로 변환
    config_dir = filepath.parent
    if not config.paths.documents.is_absolute():
        config.paths.documents = (config_dir / config.paths.documents).resolve()
    if not config.paths.output.is_absolute():
        config.paths.output = (config_dir / config.paths.output).resolve()
    
    return config
```

**_strip_none_sections 검증기:**

```python
@model_validator(mode="before")
def _strip_none_sections(cls, values: dict) -> dict:
    """None 값을 가진 최상위 키를 제거하여 기본값이 적용되도록 함"""
    if not isinstance(values, dict):
        return values
    
    return {k: v for k, v in values.items() if v is not None}
```

**동작 예제:**

```yaml
# project.yaml
project:
  name: "my-project"

# parsing 섹션이 없음 (또는 parsing: null)
```

위 YAML을 로드하면:
1. `raw_data = {"project": {"name": "my-project"}}`
2. `_strip_none_sections`가 호출되지만 None 키가 없으므로 그대로 통과
3. `ParsingConfig`의 기본값이 자동 적용됨:
   ```python
   parsing: ParsingConfig = ParsingConfig()  # 모든 필드가 기본값
   ```

### 5.3 기본 설정 생성

**create_default_config() 프로세스:**

```python
def create_default_config() -> str:
    """기본 YAML 프로젝트 템플릿을 문자열로 반환합니다."""
    # 1. 형제 경로에서 templates/project.yaml 읽기 시도
    pkg_root = Path(__file__).resolve().parent.parent.parent
    template = pkg_root / "templates/project.yaml"
    if template.is_file():
        return template.read_text(encoding="utf-8")
    
    # 2. 폴백: importlib.resources (설치된 wheel)
    try:
        ref = importlib.resources.files("slm_factory").joinpath(
            "../../templates/project.yaml"
        )
        return ref.read_text(encoding="utf-8")
    except Exception:
        pass
    
    # 3. 최후의 수단: Pydantic 모델에서 JSON 기본값 생성
    return SLMConfig().model_dump_json(indent=2)
```

**생성되는 기본 설정 예제:**

```yaml
project:
  name: "my-project"
  version: "1.0.0"
  language: "en"

paths:
  documents: "./documents"
  output: "./output"

teacher:
  backend: "ollama"
  model: "qwen3:8b"
  api_base: "http://localhost:11434"
  temperature: 0.3
  timeout: 180
  max_context_chars: 12000
  max_concurrency: 4

student:
  model: "google/gemma-3-1b-it"
  max_seq_length: 4096

training:
  num_epochs: 20
  batch_size: 4
  learning_rate: 2.0e-5
  optimizer: "adamw_torch_fused"
  lora:
    r: 16
    alpha: 32
    dropout: 0.05
    target_modules: "auto"
```

## 6. 에러 처리 전략

### 6.1 계층별 에러 처리

**CLI 계층 (cli.py):**
- 모든 명령어 함수를 try/except로 래핑
- 예외 발생 시 Rich Console로 포맷팅된 에러 메시지 출력
- `typer.Exit(1)`로 비정상 종료 코드 반환

cli.py는 `@app.callback(invoke_without_command=True)` 패턴으로 전역 옵션(`--verbose`, `--quiet`)을 처리합니다. 콜백은 모든 명령어 실행 전에 로그 레벨을 설정하며, 하위 명령어가 없으면 도움말을 출력합니다.

```python
@app.command()
def run(config_path: Path = typer.Option("project.yaml")):
    try:
        config = load_config(config_path)
        pipeline = Pipeline(config)
        pipeline.run()
    except FileNotFoundError as e:
        console.print(f"[red]설정 파일을 찾을 수 없습니다: {e}[/red]")
        raise typer.Exit(1)
    except ValidationError as e:
        console.print(f"[red]설정 검증 실패:[/red]\n{e}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]파이프라인 실행 중 오류 발생:[/red]\n{e}")
        logger.exception("Unexpected error")
        raise typer.Exit(1)
```

**Pipeline 계층 (pipeline.py):**
- 각 단계 실행 시간 추적 (`time.time()`)
- 단계 실패 시 `logger.exception()` 호출 후 예외 전파
- 성공 시 소요 시간 및 결과 통계 로그 출력

```python
def step_parse(self) -> list[ParsedDocument]:
    logger.info("Step 1: Parsing documents...")
    start_time = time.time()
    
    try:
        documents = registry.parse_directory(self.config.paths.documents)
        
        elapsed = time.time() - start_time
        logger.info(f"Parsed {len(documents)} documents in {elapsed:.2f}s")
        
        return documents
    except Exception as e:
        logger.exception("Failed to parse documents")
        raise
```

**Parser 계층 (parsers/):**
- `parse_directory()`: 개별 파일 실패를 격리
- 실패한 파일은 `logger.exception()` 호출 후 건너뜀
- 전체 프로세스는 계속 진행

```python
def parse_directory(
    self, dir_path: Path, formats: list[str] | None = None,
    files: list[Path] | None = None,
) -> list[ParsedDocument]:
    documents = []
    if files is not None:
        target_files = [Path(f) for f in files]
    else:
        target_files = sorted(
            f for f in dir_path.iterdir()
            if f.is_file() and self.get_parser(f) is not None
        )
    
    with Progress(SpinnerColumn(), ...) as progress:
        task = progress.add_task("Parsing documents", total=len(target_files))
        
        for file_path in target_files:
            parser = self.get_parser(file_path)
            if parser is None:
                progress.advance(task)
                continue
            try:
                doc = parser.parse(file_path)
                documents.append(doc)
            except Exception:
                logger.exception("Failed to parse %s", file_path.name)
            finally:
                progress.advance(task)
    
    return documents
```

**Teacher 계층 (teacher/):**
- httpx 예외를 분류하여 명확한 에러 메시지 제공
- 연결 실패, 타임아웃, HTTP 상태 에러를 구분

```python
def generate(self, prompt: str, **kwargs) -> str:
    try:
        response = httpx.post(
            f"{self.api_base}/api/generate",
            json={"model": self.model, "prompt": prompt, "stream": False, ...},
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()["response"]
    
    except httpx.TimeoutException:
        raise RuntimeError(
            f"Ollama request timed out after {self.timeout}s "
        )
    except httpx.ConnectError:
        raise RuntimeError(
            f"Cannot connect to Ollama at {self.api_base}. "
            f"Ollama가 실행 중인지 확인하세요."
        ) from e
    except httpx.HTTPStatusError as e:
        raise RuntimeError(
            f"Teacher API 오류 (status={e.response.status_code}): "
            f"{e.response.text}"
        ) from e
```

**Validator 계층 (validator/):**
- `ValidationResult` 클래스로 거부 사유 추적
- 각 규칙 실패 시 `reasons` 리스트에 추가

```python
from dataclasses import dataclass, field

@dataclass
class ValidationResult:
    """QA 쌍 검증 결과."""
    passed: bool
    reasons: list[str] = field(default_factory=list)

def validate_one(self, pair: QAPair) -> ValidationResult:
    reasons = []
    
    if not pair.question.strip() or not pair.answer.strip():
        reasons.append("empty_question_or_answer")
    
    if len(pair.answer.strip()) < self.config.min_answer_length:
        reasons.append(f"answer_too_short (...)")
    
    # ... 추가 규칙
    
    return ValidationResult(passed=len(reasons) == 0, reasons=reasons)
```

**Trainer 계층 (trainer/):**
- HuggingFace Transformers의 기본 에러 처리 상속
- CUDA 메모리 부족 시 명확한 권장 사항 제공

```python
def train(self):
    try:
        self.trainer.train()
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.error(
                "CUDA 메모리 부족. 다음을 시도해보세요:\n"
                "1. training.batch_size 감소\n"
                "2. training.quantization.enabled: true 설정\n"
                "3. student.max_seq_length 감소"
            )
        raise
```

### 6.2 에러 복구 전략

| 에러 유형 | 복구 전략 | 구현 위치 |
|----------|----------|----------|
| 개별 파일 파싱 실패 | 로그 후 건너뜀, 나머지 파일 계속 처리 | `ParserRegistry.parse_directory()` |
| QA 생성 JSON 파싱 실패 | 원본 응답 로그 출력 후 건너뜀 | `QAGenerator.parse_response()` |
| 시스템 역할 미지원 | 시스템 메시지 제거 후 재시도 | `ChatFormatter.format_one()` |
| Teacher 타임아웃 | 설정 값 증가 권장 메시지 출력 후 실패 | `BaseTeacher.generate()` |
| CUDA OOM | 배치 크기/양자화 권장 후 실패 | `LoRATrainer.train()` |
| 설정 파일 없음 | 기본 설정 생성 제안 | `cli.py` 명령어 |

## 7. 확장 포인트 정리

| 확장 대상 | 방법 | 관련 파일 | 난이도 |
|----------|------|----------|--------|
| 새 파서 추가 | `BaseParser` 상속 + `@registry.register` 데코레이터 | `parsers/base.py`<br>`parsers/__init__.py` | 쉬움 |
| 새 Teacher 백엔드 | `BaseTeacher` 상속 + `create_teacher()`에 분기 추가 | `teacher/base.py`<br>`teacher/__init__.py` | 쉬움 |
| 커스텀 질문 카테고리 | `questions.categories` 딕셔너리 수정 또는 `questions.file` 경로 지정 | `project.yaml` | 매우 쉬움 |
| 검증 규칙 추가 | `RuleValidator.validate_one()`에 로직 추가 | `validator/rules.py` | 보통 |
| 품질 점수 활성화 | `scoring.enabled: true` + threshold 조정 | `project.yaml` | 매우 쉬움 |
| 데이터 증강 활성화 | `augment.enabled: true` + num_variants 조정 | `project.yaml` | 매우 쉬움 |
| 분석 보고서 설정 | `analyzer.enabled: true` + output_file 지정 | `project.yaml` | 매우 쉬움 |
| 새 내보내기 형식 | 새 Exporter 클래스 작성 + `pipeline.step_export()`에 추가 | `exporter/`<br>`pipeline.py` | 보통 |
| Student 모델 변경 | `student.model`을 HuggingFace 모델 ID로 변경 | `project.yaml` | 매우 쉬움 |
| LoRA 타겟 모듈 변경 | `training.lora.target_modules` 리스트 수정 | `project.yaml` | 쉬움 |
| 양자화 활성화 | `training.quantization.enabled: true` 설정 | `project.yaml` | 매우 쉬움 |
| 커스텀 프롬프트 | `questions.system_prompt` 또는 `student.system_prompt` 수정 | `project.yaml` | 매우 쉬움 |
| Early Stopping 설정 | `training.early_stopping.enabled: true` + patience/threshold 조정 | `project.yaml` | 쉬움 |
| `python -m` 실행 | `__main__.py` 자동 포함 | `__main__.py` | 매우 쉬움 |
| 파이프라인 재개 | `--resume` 옵션으로 중간 파일에서 재개 | `cli.py` | 매우 쉬움 |
| 설정 검증 | `slm-factory check` 명령 | `cli.py` | 매우 쉬움 |
| CLI 유틸리티 명령어 | `status`, `clean`, `convert`, `export` 내장 | `cli.py` | 매우 쉬움 |
| 전역 로그 레벨 | `--verbose`/`--quiet` 콜백 | `cli.py` | 매우 쉬움 |
| 설정 자동 탐색 | `_find_config()` 디렉토리 탐색 | `cli.py` | 매우 쉬움 |

### 7.1 확장 예제: 새 파서 추가

**요구사항:** Markdown 파일 (`.md`) 지원 추가

**구현 단계:**

1. `parsers/markdown_parser.py` 생성:

```python
from pathlib import Path
from slm_factory.parsers.base import BaseParser
from slm_factory.models import ParsedDocument

class MarkdownParser(BaseParser):
    """Markdown 문서 파서"""
    extensions = [".md", ".markdown"]
    
    def parse(self, path: Path) -> ParsedDocument:
        with open(path, encoding="utf-8") as f:
            content = f.read()
        
        # 첫 번째 # 헤더를 제목으로 추출
        lines = content.split("\n")
        title = path.stem
        for line in lines:
            if line.startswith("# "):
                title = line[2:].strip()
                break
        
        return ParsedDocument(
            doc_id=path.name,
            title=title,
            content=content,
            metadata={"lines": len(lines)}
        )
```

2. `parsers/__init__.py`에 등록:

```python
from .markdown_parser import MarkdownParser

# 기존 등록 후 추가
registry.register(MarkdownParser)
```

3. 즉시 사용 가능:

```bash
# documents/ 디렉토리에 .md 파일 추가 후
slm-factory run --until parse
```

### 7.2 확장 예제: 커스텀 검증 규칙

**요구사항:** 답변에 특정 키워드가 포함되어야 함

**구현 단계:**

`validator/rules.py`의 `RuleValidator.validate_one()` 수정 (QAPair는 `models.py`에서 import):

```python
from slm_factory.models import QAPair

def validate_one(self, qa: QAPair) -> tuple[bool, list[str]]:
    reasons = []
    
    # 기존 규칙들...
    
    # 새 규칙: 필수 키워드 체크
    required_keywords = self.config.get("required_keywords", [])
    if required_keywords:
        answer_lower = qa.answer.lower()
        missing_keywords = [
            kw for kw in required_keywords 
            if kw.lower() not in answer_lower
        ]
        if missing_keywords:
            reasons.append(f"missing_keywords: {missing_keywords}")
    
    return (len(reasons) == 0, reasons)
```

`project.yaml`에 설정 추가:

```yaml
validation:
  required_keywords: ["파이썬", "예제"]
```

### 7.3 확장 예제: 새 내보내기 형식

**요구사항:** GGUF 형식으로 내보내기

**구현 단계:**

1. `exporter/gguf_export.py` 생성:

```python
from pathlib import Path
import subprocess
from slm_factory.config import ExportConfig

class GGUFExporter:
    """GGUF 형식 내보내기"""
    
    def __init__(self, config: ExportConfig):
        self.config = config
    
    def export(self, model_path: Path, output_path: Path) -> None:
        """llama.cpp의 convert.py 사용"""
        subprocess.run(
            [
                "python", "convert.py",
                str(model_path),
                "--outfile", str(output_path / "model.gguf"),
                "--outtype", "f16"
            ],
            check=True
        )
```

2. `pipeline.py`의 `step_export()` 수정:

```python
def step_export(self):
    # 기존 HF/Ollama 내보내기...
    
    # GGUF 내보내기 추가
    if self.config.export.gguf and self.config.export.gguf.enabled:
        from slm_factory.exporter.gguf_export import GGUFExporter
        
        exporter = GGUFExporter(self.config.export)
        exporter.export(merged_path, output_path / "gguf")
```

3. `config.py`에 설정 모델 추가:

```python
class GGUFExportConfig(BaseModel):
    enabled: bool = False
    quantization: str = "f16"  # f16, q4_0, q8_0 등

class ExportConfig(BaseModel):
    merge_lora: bool = True
    ollama: OllamaExportConfig | None = None
    gguf: GGUFExportConfig | None = None
```

4. `project.yaml`에서 활성화:

```yaml
export:
  merge_lora: true
  gguf:
    enabled: true
    quantization: q4_0
```

---

이 아키텍처 가이드는 SLM Factory의 설계 원칙, 컴포넌트 구조, 데이터 흐름, 설정 시스템, 에러 처리 전략, 확장 방법을 상세히 설명합니다. 각 섹션은 실제 코드 예제와 함께 제공되어 개발자가 시스템을 이해하고 확장할 수 있도록 돕습니다.

---

## 8. Wizard 대화형 모드 아키텍처 (권장 사용 방식)

### 8.1 설계 목표

Wizard 모드는 **처음 사용자에게 권장하는 기본 실행 방식**입니다. CLI의 23개 명령어를 개별로 호출하는 대신, 단일 `wizard` 명령으로 전체 파이프라인을 단계별 확인하며 실행할 수 있습니다.

**핵심 원칙:**
- **무지식 실행**: 사용자가 파이프라인 구조를 몰라도 진행 가능
- **탈출 가능**: 어느 단계에서든 건너뛸 수 있고, 나중에 개별 CLI 명령으로 재개
- **설정 반영**: 선택적 단계의 기본값은 `project.yaml` 설정을 따름
- **피드백 즉시 제공**: 각 단계 완료 시 건수/경로 등 결과 즉시 표시

### 8.2 실행 흐름

```
wizard
  ├─ Step 1. 설정 파일 ──────────── _find_config() → _load_pipeline()
  │    └─ 자동 탐색 실패 시 Prompt.ask()
  │
  ├─ Step 2. 문서 선택 ──────────── 디렉토리 스캔 → Rich Table → Confirm/번호 입력
  │    └─ 전체 선택 또는 개별 번호 입력
  │
  ├─ Step 3. 문서 파싱 ──────────── pipeline.step_parse(files=selected)
  │    └─ 자동 진행 (확인 없음)
  │
  ├─ Step 4. QA 생성 ────────────── Confirm → pipeline.step_generate(docs)
  │    └─ 거부 시 → 파싱 결과 경로 안내 후 종료
  │
  ├─ Step 5. 검증 ───────────────── pipeline.step_validate(pairs, docs)
  │    └─ 자동 진행 (확인 없음)
  │
  ├─ Step 6. 품질 평가 (선택적) ──── Confirm(default=scoring.enabled) → step_score()
  │    └─ config.scoring.enabled 기본값 반영
  │
  ├─ Step 7. 데이터 증강 (선택적) ── Confirm(default=augment.enabled) → step_augment()
  │    └─ config.augment.enabled 기본값 반영
  │
  ├─ 분석 ───────────────────────── pipeline.step_analyze(pairs)
  │    └─ 자동 진행 (확인 없음)
  │
  ├─ Step 8. 학습 ───────────────── step_convert() → Confirm → step_train()
  │    └─ 거부 시 → 학습 데이터 경로 + train 명령어 안내 후 종료
  │
  ├─ Step 9. 내보내기 ──────────── Confirm → step_export()
  │    └─ 거부 시 → 어댑터 경로 + export 명령어 안내 후 종료
  │
  ├─ Step 10. 멀티턴 대화 (선택적) ── Confirm(default=dialogue.enabled) → step_dialogue()
  │    └─ config.dialogue.enabled 기본값 반영
  │
  ├─ Step 11. GGUF 변환 (선택적) ──── Confirm(default=gguf_export.enabled) → step_gguf_export()
  │    └─ config.gguf_export.enabled 기본값 반영
  │
  └─ Step 12. 모델 평가 (선택적) ──── Confirm(default=eval.enabled) → step_eval()
       └─ config.eval.enabled 기본값 반영
```

### 8.3 단계 분류

| 분류 | 단계 | 확인 방식 | 기본값 |
|------|------|----------|--------|
| **필수** | 설정, 파싱, 검증, 분석 | 자동 진행 | — |
| **선택+진행** | QA 생성, 학습, 내보내기 | `Confirm.ask(default=True)` | Y |
| **선택+설정** | 품질 평가, 증강, 멀티턴 대화, GGUF 변환, 모델 평가 | `Confirm.ask(default=config값)` | config 의존 |

### 8.4 탈출 시 복구 전략

각 단계를 건너뛸 때 wizard는 나중에 해당 단계를 개별 실행할 수 있는 정확한 CLI 명령어를 안내합니다:

```python
# QA 생성 건너뜀
console.print(f"나중에 실행: slm-factory run --until generate --config {resolved}")

# 학습 건너뜀
console.print(f"나중에 실행: slm-factory train --config {resolved} --data {training_data_path}")

# 내보내기 건너뜀
console.print(f"나중에 실행: slm-factory export --config {resolved} --adapter {adapter_path}")
```

이 설계로 wizard를 중간에 중단하더라도 중간 결과 파일이 보존되어 `--resume` 옵션 또는 개별 명령어로 이어서 진행할 수 있습니다.

### 8.5 Rich UI 컴포넌트

wizard는 다음 Rich 컴포넌트를 사용합니다:

| 컴포넌트 | 용도 |
|----------|------|
| `Panel` | 시작 배너, 완료 요약 |
| `Table` | 문서 목록 (번호/파일명/크기) |
| `Confirm.ask()` | Y/n 선택 (기본값 지원) |
| `Prompt.ask()` | 텍스트 입력 (설정 파일 경로, 문서 번호) |
| 색상 마커 | `[green]✓` 성공, `[red]✗` 실패, `[yellow]⏭` 건너뜀 |

---

## 관련 문서

- [README](../README.md) — 프로젝트 소개, 설치, 빠른 시작, CLI 레퍼런스
- [설정 레퍼런스](configuration.md) — `project.yaml`의 모든 설정 옵션 상세 설명
- [모듈별 상세 문서](modules.md) — 각 모듈의 클래스, 함수, 확장 방법
