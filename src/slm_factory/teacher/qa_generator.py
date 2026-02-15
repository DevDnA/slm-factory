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

if TYPE_CHECKING:
    from ..config import QuestionsConfig, SLMConfig, TeacherConfig
    from ..models import ParsedDocument

from ..models import QAPair
from ..utils import get_logger

logger = get_logger("teacher.qa_generator")


class QAGenerator:
    """교사 LLM을 사용하여 문서에서 QA 쌍을 생성합니다.
    
    Args:
        config: 전체 SLMConfig(교사, 질문 섹션 필요)
    """
    
    def __init__(self, config: SLMConfig):
        from ..teacher import create_teacher
        self.config = config
        self.teacher = create_teacher(config.teacher)
        self.questions_config = config.questions
        self.teacher_config = config.teacher
        self.max_context = config.teacher.max_context_chars
    
    def build_prompt(
        self,
        doc_title: str,
        content: str,
        question: str,
        tables: list[str] | None = None,
        system_prompt: str | None = None,
    ) -> str:
        """QA 생성을 위한 전체 프롬프트를 구성합니다.
        
        Args:
            doc_title: 문서 제목
            content: 문서 내용(max_context_chars로 잘림)
            question: 답변할 질문
            tables: 선택적 테이블 마크다운 문자열 목록
            system_prompt: 선택적 시스템 프롬프트(기본값: config.questions.system_prompt)
        
        Returns:
            교사 LLM을 위해 준비된 완전한 프롬프트 문자열
        """
        if system_prompt is None:
            system_prompt = self.questions_config.system_prompt
        
        # 최대 컨텍스트로 내용 잘라내기
        truncated_content = content[:self.max_context]
        if len(content) > self.max_context:
            truncated_content += "\n\n[Content truncated...]"
        
        # 프롬프트 섹션 구성
        prompt_parts = [
            f"# System Instructions\n{system_prompt}",
            f"\n# Document: {doc_title}\n{truncated_content}",
        ]
        
        # 테이블이 있으면 추가
        if tables:
            tables_section = "\n## Related Tables\n" + "\n\n".join(tables)
            prompt_parts.append(tables_section)
        
        # 질문 및 형식 지침 추가
        prompt_parts.extend([
            f"\n# Question\n{question}",
            '\n# Instructions',
            'Answer the question based strictly on the document content above.',
            'Return ONLY valid JSON in this exact format:',
            '{"instruction": "the question", "output": "your detailed answer"}',
            '',
            'Do NOT include code fences, markdown formatting, or any text outside the JSON object.',
            'Do NOT add reasoning or explanations.',
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
        
        Args:
            docs: 파싱된 문서 목록
            questions: 선택적 질문 목록(기본값: config.questions.get_all_questions())
        
        Returns:
            생성된 모든 QAPair 객체의 평탄화된 목록
        """
        all_pairs: list[QAPair] = []
        
        for doc in docs:
            pairs = self.generate_for_document(doc, questions=questions)
            all_pairs.extend(pairs)
        
        logger.info("Generated %d QA pairs from %d documents", len(all_pairs), len(docs))
        return all_pairs
    
    # ------------------------------------------------------------------
    # 비동기 생성(세마포어를 사용한 동시 요청)
    # ------------------------------------------------------------------

    async def _generate_one_async(
        self,
        semaphore: asyncio.Semaphore,
        doc: ParsedDocument,
        question: str,
        category: str = "",
    ) -> QAPair | None:
        async with semaphore:
            try:
                prompt = self.build_prompt(
                    doc_title=doc.title,
                    content=doc.content,
                    question=question,
                    tables=doc.tables if doc.tables else None,
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

    async def generate_all_async(
        self,
        docs: list[ParsedDocument],
        questions: list[str] | None = None,
    ) -> list[QAPair]:
        max_concurrency = self.teacher_config.max_concurrency
        semaphore = asyncio.Semaphore(max_concurrency)

        all_questions = questions or self.questions_config.get_all_questions()
        if not all_questions:
            logger.warning("No questions configured")
            return []

        tasks: list[asyncio.Task[QAPair | None]] = []
        for doc in docs:
            for question in all_questions:
                task = asyncio.create_task(
                    self._generate_one_async(semaphore, doc, question)
                )
                tasks.append(task)

        total = len(tasks)
        logger.info(
            "Generating %d QA pairs (concurrency=%d)...",
            total,
            max_concurrency,
        )

        results = await asyncio.gather(*tasks, return_exceptions=True)

        pairs: list[QAPair] = []
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
