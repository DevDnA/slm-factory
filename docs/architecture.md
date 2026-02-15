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
┌─────────────────────────────────────────────────────────────────────┐
│                            CLI (cli.py)                              │
│                         Entry Point (Typer)                          │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       Pipeline (pipeline.py)                         │
│                         Orchestrator (9 steps)                       │
└─┬───┬───┬───┬───┬───┬───┬───┬───┬───────────────────────────────────┘
  │   │   │   │   │   │   │   │   │
  ▼   ▼   ▼   ▼   ▼   ▼   ▼   ▼   ▼
┌───┬───┬───┬───┬───┬───┬───┬───┬───┐
│ 1 │ 2 │ 3 │3a │3b │3c │ 4 │ 5 │ 6 │
└─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┘
  │   │   │   │   │   │   │   │   │
  ▼   ▼   ▼   ▼   ▼   ▼   ▼   ▼   ▼
┌───┬───┬───┬───┬───┬───┬───┬───┬───┐
│Par│Tea│Val│Sco│Aug│Ana│Con│Tra│Exp│
│ser│cher│ida│rer│men│lyz│ver│iner│ort│
│s/ │/  │tor│/  │ter│er/│ter│/  │er/│
└─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┘
  │   │   │   │   │   │   │   │   │
  ▼   ▼   ▼   ▼   ▼   ▼   ▼   ▼   ▼
┌────┬────┐   │   │ ┌────┬────┬────┬────┐
│.json│.json│  │   │ │.json│.jsonl│ckpt│model│
└────┴────┘   │   │ └────┴────┴────┴────┘
              │   │
parsed_   qa_ │   │ data_  training_  adapter/  merged_
documents alpaca│  │ analysis data              model/
.json     .json │  │ .json  .jsonl            Modelfile
              (filter) (augment)

┌─────────────────────────────────────────────────────────────────────┐
│                    Config (config.py)                                │
│              Injected into all components                            │
└─────────────────────────────────────────────────────────────────────┘
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
        │     └─→ text.py (TextParser)
        │
        ├─→ teacher/
        │     ├─→ base.py (BaseTeacher)
        │     ├─→ ollama.py (OllamaTeacher)
        │     ├─→ openai_compat.py (OpenAICompatTeacher)
        │     └─→ qa_generator.py (QAGenerator)
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
        └─→ exporter/
              ├─→ hf_export.py (HFExporter)
              └─→ ollama_export.py (OllamaExporter)

All modules
  ├─→ config.py (SLMConfig + 14 sub-models)
  ├─→ models.py (QAPair, ParsedDocument)
  └─→ utils.py (setup_logging)
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
`parsers/__init__.py`에서 전역 `registry` 인스턴스를 생성하고 `PDFParser`, `HWPXParser`, `HTMLParser`, `TextParser`를 자동 등록합니다.

**BaseParser 추상 클래스:**
- `extensions` ClassVar: 지원하는 파일 확장자 목록
- `parse()` 추상 메서드: 파일 경로를 받아 `ParsedDocument` 반환
- `can_parse()`: 파일 확장자가 `extensions`에 포함되는지 확인

**커스텀 파서 추가 예제:**

```python
from slm_factory.parsers.base import BaseParser
from slm_factory.models import ParsedDocument
from slm_factory.parsers import registry

@registry.register
class DOCXParser(BaseParser):
    """Microsoft Word 문서 파서"""
    extensions = [".docx"]
    
    def parse(self, path: Path) -> ParsedDocument:
        from docx import Document
        
        doc = Document(path)
        text = "\n".join([para.text for para in doc.paragraphs])
        
        return ParsedDocument(
            doc_id=path.name,
            title=path.stem,
            content=text,
            metadata={"paragraphs": len(doc.paragraphs)}
        )
```

등록 후 자동으로 `.docx` 파일이 파이프라인에서 처리됩니다.

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
3. **Pattern 체크**: 3개의 정규식 패턴에 매칭되면 거부 (예: "I don't have", "정보가 없습니다")
4. **Deduplication**: 이미 처리된 질문과 중복되면 거부

**GroundednessChecker 전략:**
- 문서를 512자 청크로 분할 (64자 오버랩)
- 각 청크와 답변을 임베딩으로 변환
- 코사인 유사도 계산 후 임계값(기본 0.3) 이상인 청크가 있으면 통과

**조합 예제:**

```python
# pipeline.py step_validate()
rule_validator = RuleValidator(config.validation)
accepted, rejected = rule_validator.validate_batch(qa_pairs)

if config.validation.groundedness and config.validation.groundedness.enabled:
    checker = GroundednessChecker(config.validation.groundedness)
    accepted, more_rejected = checker.check_batch(accepted, documents)
    rejected.extend(more_rejected)
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
# 1. QAPair → 메시지 리스트 구성
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": qa.question},
    {"role": "assistant", "content": qa.answer}
]

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

### 3.6 통계 분석 패턴 (analyzer.py)

Analyzer는 LLM 의존성 없이 순수 통계 분석을 수행합니다.

- Counter로 분포 계산
- statistics 모듈로 기초 통계
- 자동 경고 생성 (불균형, 이상치 등)
- Rich 콘솔 시각적 보고서

## 4. 데이터 흐름 상세

### 4.1 Step 1: parse

**입력:**
- `documents/` 디렉토리 내 파일들 (PDF, HWPX, HTML, TXT 등)

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
      "tables": [["헤더1", "헤더2"], ["값1", "값2"]],
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
1. `QAGenerator.generate_all()` 호출
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
      "source": "document1.pdf",
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
      r"(?i)I don't have.*information",
      r"(?i)정보가 없습니다",
      r"(?i)답변할 수 없습니다"
  ]
  for pattern in reject_patterns:
      if re.search(pattern, qa.answer):
          reject("reject_pattern_match")
  ```

  **규칙 4 - Deduplication:**
  ```python
  question_hash = qa.question.lower().strip()
  if question_hash in seen_questions:
      reject("duplicate_question")
  seen_questions.add(question_hash)
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

**출력:** list[QAPair] (필터링됨)

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

**출력:** list[QAPair] (원본 + 증강)

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
   from slm_factory.models import QAPair
   
   def build_messages(qa: QAPair, system_prompt: str):
      return [
          {"role": "system", "content": system_prompt},
          {"role": "user", "content": qa.question},
          {"role": "assistant", "content": qa.answer}
      ]
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
  tokens = tokenizer.encode(text, add_special_tokens=False)
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
│   ├── formats: list[str] = ["pdf"]
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
└── export: ExportConfig
    ├── merge_lora: bool = True
    ├── output_format: str = "safetensors"
    └── ollama: OllamaExportConfig
        ├── enabled: bool = True
        ├── model_name: str = "my-project-model"
        ├── system_prompt: str = "You are a helpful..."
        └── parameters: dict = {temperature: 0.7, top_p: 0.9, num_ctx: 4096}
```

### 5.2 설정 로드 프로세스

**load_config() 흐름:**

```python
def load_config(config_path: Path) -> SLMConfig:
    # 1. YAML 파일 읽기
    with open(config_path) as f:
        raw_data = yaml.safe_load(f)
    
    # 2. Pydantic 검증 (자동으로 _strip_none_sections 호출됨)
    config = SLMConfig.model_validate(raw_data)
    
    # 3. 검증된 설정 객체 반환
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
def create_default_config(output_path: Path) -> None:
    try:
        # 1. 패키지 내 기본 설정 파일 읽기 시도
        if importlib.resources.is_resource("slm_factory", "project.yaml"):
            content = importlib.resources.read_text("slm_factory", "project.yaml")
            with open(output_path, "w") as f:
                f.write(content)
        else:
            raise FileNotFoundError
    except (FileNotFoundError, ModuleNotFoundError):
        # 2. 폴백: Pydantic 모델에서 기본값 생성
        default_config = SLMConfig()
        yaml_content = yaml.dump(
            default_config.model_dump(mode="json", exclude_none=True),
            allow_unicode=True,
            sort_keys=False
        )
        with open(output_path, "w") as f:
            f.write(yaml_content)
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
def parse_directory(self, directory: Path) -> list[ParsedDocument]:
    documents = []
    files = sorted(f for f in directory.iterdir() if f.is_file())
    
    with Progress() as progress:
        task = progress.add_task("Parsing...", total=len(files))
        
        for file_path in files:
            try:
                parser = self.get_parser(file_path)
                if parser:
                    doc = parser.parse(file_path)
                    documents.append(doc)
            except Exception as e:
                logger.exception(f"Failed to parse {file_path}")
                # 계속 진행
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
from dataclasses import dataclass
from slm_factory.models import QAPair

@dataclass
class ValidationResult:
     accepted: list[QAPair]
     rejected: list[QAPair]
     reasons: dict[str, list[str]]  # qa_id → [reason1, reason2, ...]

def validate_one(self, qa: QAPair) -> tuple[bool, list[str]]:
    reasons = []
    
    if not qa.question.strip() or not qa.answer.strip():
        reasons.append("empty_field")
    
    if len(qa.answer) < self.config.min_answer_length:
        reasons.append("answer_too_short")
    
    # ... 추가 규칙
    
    return (len(reasons) == 0, reasons)
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
| 커스텀 질문 카테고리 | `questions.categories` 리스트 수정 또는 `questions.file` 경로 지정 | `project.yaml` | 매우 쉬움 |
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
slm-factory parse
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

## 관련 문서

- [README](../README.md) — 프로젝트 소개, 설치, 빠른 시작, CLI 레퍼런스
- [설정 레퍼런스](configuration.md) — `project.yaml`의 모든 설정 옵션 상세 설명
- [모듈별 상세 문서](modules.md) — 각 모듈의 클래스, 함수, 확장 방법
