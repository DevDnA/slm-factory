"""교사 LLM을 사용한 QA 쌍 생성.

파싱된 문서에서 질문-답변 쌍의 생성을 조율합니다.
각 질문은 최대 답변 품질을 위해 문서별로 개별적으로 처리됩니다.
출력은 Alpaca 형식입니다: {instruction, input, output}.
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn

if TYPE_CHECKING:
    from ..config import ChunkingConfig, QuestionsConfig, SLMConfig, TeacherConfig
    from ..models import ParsedDocument

from ..models import QAPair
from ..utils import get_logger

logger = get_logger("teacher.qa_generator")


def chunk_document(content: str, chunk_size: int, overlap: int) -> list[str]:
    """문서 내용을 중첩 청크로 분할합니다.

    문단 경계(\n\n)를 우선 존중하여 자연스러운 분할을 시도합니다.
    문단 경계를 찾을 수 없으면 ``chunk_size``에서 강제 분할합니다.
    """
    if len(content) <= chunk_size:
        return [content]

    chunks: list[str] = []
    start = 0
    while start < len(content):
        end = start + chunk_size

        if end >= len(content):
            chunks.append(content[start:])
            break

        # 문단 경계(\n\n)를 찾아 자연스럽게 분할
        boundary = content.rfind("\n\n", start + chunk_size // 2, end)
        if boundary > start:
            end = boundary

        chunks.append(content[start:end])
        start = end - overlap

    return chunks


class QAGenerator:
    """교사 LLM을 사용하여 문서에서 QA 쌍을 생성합니다.
    
    Args:
        config: 전체 SLMConfig(교사, 질문 섹션 필요)
    """
    
    def __init__(self, config: SLMConfig) -> None:
        from ..teacher import create_teacher
        self.config = config
        self.teacher = create_teacher(config.teacher)
        self.questions_config = config.questions
        self.teacher_config = config.teacher
        self.chunking_config = config.chunking
        self.max_context = config.teacher.max_context_chars
    
    def build_prompt(
        self,
        doc_title: str,
        content: str,
        question: str,
        tables: list[str] | None = None,
        system_prompt: str | None = None,
        ontology_context: str | None = None,
        chunk_info: str | None = None,
    ) -> str:
        """QA 생성을 위한 전체 프롬프트를 구성합니다.
        
        Args:
            doc_title: 문서 제목
            content: 문서 내용(max_context_chars로 잘림)
            question: 답변할 질문
            tables: 선택적 테이블 마크다운 문자열 목록
            system_prompt: 선택적 시스템 프롬프트(기본값: config.questions.system_prompt)
            ontology_context: 온톨로지 지식 그래프 컨텍스트 문자열
            chunk_info: 청크 위치 정보 (예: "Part 2/5")
        
        Returns:
            교사 LLM을 위해 준비된 완전한 프롬프트 문자열
        """
        if system_prompt is None:
            system_prompt = self.questions_config.system_prompt
        
        truncated_content = content[:self.max_context]
        if len(content) > self.max_context:
            truncated_content += "\n\n[이하 생략...]"
        
        title_with_chunk = doc_title
        if chunk_info:
            title_with_chunk = f"{doc_title} ({chunk_info})"
        
        prompt_parts = [
            f"# 시스템 지시사항\n{system_prompt}",
            f"\n# 문서: {title_with_chunk}\n{truncated_content}",
        ]
        
        if tables:
            tables_section = "\n## 관련 표\n" + "\n\n".join(tables)
            prompt_parts.append(tables_section)

        if ontology_context:
            prompt_parts.append(f"\n## 관련 지식\n{ontology_context}")
        
        prompt_parts.extend([
            f"\n# 질문\n{question}",
            '\n# 지시사항',
            '위 문서 내용만을 근거로 질문에 답변하세요.',
            '반드시 아래 JSON 형식으로만 응답하세요:',
            '{"instruction": "질문 내용", "output": "상세한 답변"}',
            '',
            '코드 블록, 마크다운 서식, JSON 외의 텍스트를 포함하지 마세요.',
            '추론 과정이나 부가 설명을 추가하지 마세요.',
        ])
        
        return "\n".join(prompt_parts)
    
    def parse_response(self, text: str) -> dict[str, str] | None:
        """LLM 응답을 {instruction, output} 딕셔너리로 파싱합니다.
        
        여러 응답 형태를 처리합니다:
        - instruction/output이 있는 직접 JSON 객체
        - JSON 배열(첫 번째 항목 선택)
        - {"data": [...]} 또는 {"items": [...]}로 래핑됨
        - "instruction"/"output" 대신 "question"/"answer" 키 사용(정규화)
        
        Args:
            text: 원본 LLM 응답 텍스트
        
        Returns:
            'instruction'과 'output' 키가 있는 딕셔너리, 파싱 실패 시 None
        """
        # 코드 펜스 제거
        text = text.strip()
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
        text = text.strip()
        
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse JSON response: %s", e)
            return None
        
        # 다양한 응답 형태 처리
        result = None
        
        # 경우 1: instruction/output이 있는 직접 객체
        if isinstance(data, dict) and ("instruction" in data or "question" in data):
            result = data
        
        # 경우 2: 배열 - 첫 번째 항목 선택
        elif isinstance(data, list) and len(data) > 0:
            result = data[0]
        
        # 경우 3: "data" 또는 "items" 키로 래핑됨
        elif isinstance(data, dict):
            if "data" in data:
                wrapped = data["data"]
                if isinstance(wrapped, list) and len(wrapped) > 0:
                    result = wrapped[0]
                elif isinstance(wrapped, dict):
                    result = wrapped
            elif "items" in data:
                wrapped = data["items"]
                if isinstance(wrapped, list) and len(wrapped) > 0:
                    result = wrapped[0]
                elif isinstance(wrapped, dict):
                    result = wrapped
        
        if result is None:
            logger.warning("Could not extract QA pair from response structure")
            return None
        
        # 키 정규화: question -> instruction, answer -> output
        normalized = {}
        
        if "instruction" in result:
            normalized["instruction"] = result["instruction"]
        elif "question" in result:
            normalized["instruction"] = result["question"]
        else:
            logger.warning("Response missing 'instruction' or 'question' field")
            return None
        
        if "output" in result:
            normalized["output"] = result["output"]
        elif "answer" in result:
            normalized["output"] = result["answer"]
        else:
            logger.warning("Response missing 'output' or 'answer' field")
            return None
        
        return normalized
    
    def generate_for_document(
        self,
        doc: ParsedDocument,
        questions: list[str] | None = None,
        category: str = "",
    ) -> list[QAPair]:
        """단일 문서에 대한 QA 쌍을 생성합니다.
        
        최대 답변 품질을 위해 질문을 한 번에 하나씩 처리합니다.
        
        Args:
            doc: QA 쌍을 생성할 파싱된 문서
            questions: 선택적 질문 목록(기본값: config.questions.get_all_questions())
            category: QA 쌍에 대한 선택적 카테고리 레이블
        
        Returns:
            성공적으로 생성된 QAPair 객체 목록
        """
        if questions is None:
            questions = self.questions_config.get_all_questions()
        
        if not questions:
            logger.warning("No questions configured for document %s", doc.doc_id)
            return []
        
        pairs: list[QAPair] = []
        total = len(questions)
        
        for i, question in enumerate(questions, 1):
            logger.info("Generating QA %d/%d for %s", i, total, doc.title)
            
            try:
                # 프롬프트 구성
                prompt = self.build_prompt(
                    doc_title=doc.title,
                    content=doc.content,
                    question=question,
                    tables=doc.tables if doc.tables else None,
                )
                
                # 응답 생성
                # Ollama 백엔드의 경우 format="json" 전달
                kwargs = {}
                if self.teacher_config.backend == "ollama":
                    kwargs["format"] = "json"
                
                response = self.teacher.generate(prompt, **kwargs)
                
                # 응답 파싱
                parsed = self.parse_response(response)
                if parsed is None:
                    logger.warning("Failed to parse response for question: %s", question[:50])
                    continue
                
                # QAPair 생성
                pair = QAPair(
                    question=question,
                    answer=parsed["output"],
                    instruction=parsed["instruction"],
                    source_doc=doc.doc_id,
                    category=category,
                )
                pairs.append(pair)
                
            except Exception as e:
                logger.error("Error generating QA for question '%s': %s", question[:50], e)
                continue
        
        logger.info("Generated %d/%d QA pairs for %s", len(pairs), total, doc.title)
        return pairs
    
    def generate_all(
        self,
        docs: list[ParsedDocument],
        questions: list[str] | None = None,
    ) -> list[QAPair]:
        """여러 문서에 대한 QA 쌍을 생성합니다.
        
        내부적으로 generate_all_async()를 사용하여 동시 요청으로 성능을 향상합니다.
        
        Args:
            docs: 파싱된 문서 목록
            questions: 선택적 질문 목록(기본값: config.questions.get_all_questions())
        
        Returns:
            생성된 모든 QAPair 객체의 평탄화된 목록
        """
        from ..utils import run_async
        return run_async(self.generate_all_async(docs, questions=questions))
    
    # ------------------------------------------------------------------
    # 비동기 생성(세마포어를 사용한 동시 요청)
    # ------------------------------------------------------------------

    async def _generate_one_async(
        self,
        semaphore: asyncio.Semaphore,
        doc: ParsedDocument,
        question: str,
        category: str = "",
        ontology_context: str | None = None,
        chunk_content: str | None = None,
        chunk_info: str | None = None,
    ) -> QAPair | None:
        async with semaphore:
            try:
                content = chunk_content if chunk_content is not None else doc.content
                prompt = self.build_prompt(
                    doc_title=doc.title,
                    content=content,
                    question=question,
                    tables=doc.tables if doc.tables else None,
                    ontology_context=ontology_context,
                    chunk_info=chunk_info,
                )

                kwargs: dict[str, Any] = {}
                if self.teacher_config.backend == "ollama":
                    kwargs["format"] = "json"

                response = await self.teacher.agenerate(prompt, **kwargs)

                parsed = self.parse_response(response)
                if parsed is None:
                    logger.warning("Failed to parse response for question: %s", question[:50])
                    return None

                return QAPair(
                    question=question,
                    answer=parsed["output"],
                    instruction=parsed["instruction"],
                    source_doc=doc.doc_id,
                    category=category,
                )
            except Exception as e:
                logger.error("Error generating QA for question '%s': %s", question[:50], e)
                return None

    def _get_doc_chunks(self, doc: ParsedDocument) -> list[tuple[str, str | None]]:
        """문서를 청크로 분할하여 (content, chunk_info) 튜플 리스트를 반환합니다.

        청킹이 비활성화되었거나 문서가 chunk_size보다 짧으면
        원본 문서를 단일 청크로 반환합니다.
        """
        if not self.chunking_config.enabled:
            return [(doc.content, None)]

        chunks = chunk_document(
            doc.content,
            self.chunking_config.chunk_size,
            self.chunking_config.overlap_chars,
        )
        if len(chunks) == 1:
            return [(chunks[0], None)]

        return [
            (chunk, f"Part {i + 1}/{len(chunks)}")
            for i, chunk in enumerate(chunks)
        ]

    async def generate_all_async(
        self,
        docs: list[ParsedDocument],
        questions: list[str] | None = None,
        ontology_context: dict[str, str] | None = None,
    ) -> list[QAPair]:
        """여러 문서에 대한 QA 쌍을 비동기적으로 생성합니다.

        세마포어를 사용하여 동시 요청 수를 제한하며,
        ``generate_all``의 비동기 버전입니다.
        청킹이 활성화되면 문서를 청크로 분할하여 각 청크별로 QA를 생성합니다.

        Args:
            docs: 파싱된 문서 목록
            questions: 선택적 질문 목록(기본값: config.questions.get_all_questions())
            ontology_context: 문서 제목 → 온톨로지 컨텍스트 문자열 매핑

        Returns:
            생성된 모든 QAPair 객체의 평탄화된 목록
        """
        max_concurrency = self.teacher_config.max_concurrency
        semaphore = asyncio.Semaphore(max_concurrency)

        all_questions = questions or self.questions_config.get_all_questions()
        if not all_questions:
            logger.warning("No questions configured")
            return []

        doc_chunks: list[tuple[ParsedDocument, str, str | None]] = []
        for doc in docs:
            for chunk_content, chunk_info in self._get_doc_chunks(doc):
                doc_chunks.append((doc, chunk_content, chunk_info))

        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeRemainingColumn(),
        )

        total = len(doc_chunks) * len(all_questions)

        if self.chunking_config.enabled and len(doc_chunks) > len(docs):
            logger.info(
                "Chunking enabled: %d documents → %d chunks, "
                "generating %d QA pairs (concurrency=%d)...",
                len(docs), len(doc_chunks), total, max_concurrency,
            )
        else:
            logger.info(
                "Generating %d QA pairs (concurrency=%d)...",
                total, max_concurrency,
            )

        pairs: list[QAPair] = []

        with progress:
            task_id = progress.add_task("QA 쌍 생성 중...", total=total)

            async def _generate_with_progress(
                semaphore: asyncio.Semaphore,
                doc: ParsedDocument,
                question: str,
                onto_ctx: str | None,
                chunk_content: str | None,
                chunk_info: str | None,
            ) -> QAPair | None:
                result = await self._generate_one_async(
                    semaphore, doc, question,
                    ontology_context=onto_ctx,
                    chunk_content=chunk_content,
                    chunk_info=chunk_info,
                )
                progress.advance(task_id)
                return result

            tasks: list[asyncio.Task[QAPair | None]] = []
            for doc, chunk_content, chunk_info in doc_chunks:
                doc_onto_ctx = (
                    ontology_context.get(doc.title)
                    if ontology_context
                    else None
                )
                for question in all_questions:
                    task = asyncio.create_task(
                        _generate_with_progress(
                            semaphore, doc, question, doc_onto_ctx,
                            chunk_content, chunk_info,
                        )
                    )
                    tasks.append(task)

            results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                logger.error("Task failed: %s", result)
            elif result is not None:
                pairs.append(result)

        logger.info(
            "Generated %d QA pairs from %d documents (%d failed)",
            len(pairs),
            len(docs),
            total - len(pairs),
        )
        return pairs

    def save_alpaca(self, pairs: list[QAPair], output_path: str | Path) -> Path:
        """QA 쌍을 Alpaca 형식 JSON 파일로 저장합니다.
        
        형식: {"instruction": ..., "input": "", "output": ...}의 목록
        
        Args:
            pairs: 저장할 QAPair 객체 목록
            output_path: JSON 파일을 쓸 경로
        
        Returns:
            작성된 파일의 Path 객체
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Alpaca 형식으로 변환
        alpaca_data = [
            {
                "instruction": pair.instruction,
                "input": "",
                "output": pair.answer,
            }
            for pair in pairs
        ]
        
        # JSON 작성
        output_path.write_text(
            json.dumps(alpaca_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        
        logger.info("Saved %d QA pairs to %s", len(pairs), output_path)
        return output_path
