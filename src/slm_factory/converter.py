"""형식 변환기 — QA 쌍을 채팅 템플릿 학습 데이터로 변환합니다.

HuggingFace tokenizer.apply_chat_template()을 사용하여 모든 학생 모델
(Gemma, Llama, Mistral, Phi, Qwen 등)에 대한 올바른 형식을 보장합니다.

이것은 QA 생성(Alpaca 형식)과 LoRA(Low-Rank Adaptation) 미세 조정 간의 다리입니다.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .config import SLMConfig
    from .models import QAPair

from .utils import get_logger

logger = get_logger("converter")


class ChatFormatter:
    """QA 쌍을 채팅 템플릿 형식의 학습 데이터로 변환합니다.
    
    학생 모델의 토크나이저를 사용하여 올바른 채팅 템플릿을 적용하고,
    모든 HuggingFace 모델에 대해 올바른 역할 태그(<start_of_turn>, [INST], <|im_start|> 등)를 보장합니다.
    """
    
    def __init__(self, config: SLMConfig):
        self.config = config
        self.model_name = config.student.model
        self.max_seq_length = config.student.max_seq_length
        self.system_prompt = config.questions.system_prompt
        self._tokenizer = None  # 지연 로딩
    
    @property
    def tokenizer(self):
        """토크나이저를 지연 로딩합니다 (필요하지 않으면 import 비용 회피)."""
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
            )
            logger.info("토크나이저 로드됨: %s", self.model_name)
        return self._tokenizer
    
    def build_messages(self, pair: QAPair) -> list[dict[str, str]]:
        """QA 쌍에서 OpenAI 메시지 형식을 구성합니다.
        
        매개변수
        ----------
        pair : QAPair
            형식화할 질문-답변 쌍.
        
        반환값
        -------
        list[dict[str, str]]
            'role'과 'content' 키를 가진 메시지 딕셔너리 리스트.
        """
        messages = []
        
        # 시스템 메시지가 있으면만 추가
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        
        messages.append({"role": "user", "content": pair.question})
        messages.append({"role": "assistant", "content": pair.answer})
        
        return messages
    
    def format_one(self, pair: QAPair) -> str | None:
        """모델의 채팅 템플릿을 사용하여 단일 QA 쌍을 형식화합니다.
        
        tokenizer.apply_chat_template()을 적용하여 메시지를 학생 모델의 올바른 형식으로 변환합니다.
        모델이 시스템 역할을 지원하지 않으면(예: Gemma) 시스템 역할 없이 재시도합니다.
        
        매개변수
        ----------
        pair : QAPair
            형식화할 질문-답변 쌍.
        
        반환값
        -------
        str or None
            학습 준비가 된 형식화된 문자열, 또는 형식화 실패 시 None.
        """
        messages = self.build_messages(pair)
        
        try:
            # 먼저 시스템 메시지로 시도
            formatted = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            return formatted
        except Exception as e:
            # 일부 모델(예: Gemma)은 시스템 역할을 지원하지 않음
            # 시스템 메시지 없이 재시도
            logger.debug(
                "%s에서 시스템 역할 실패 (오류: %s), 시스템 없이 재시도 중",
                self.model_name,
                type(e).__name__,
            )
            
            # 시스템 없이 메시지 재구성
            messages_no_system = [m for m in messages if m["role"] != "system"]
            
            try:
                formatted = self.tokenizer.apply_chat_template(
                    messages_no_system,
                    tokenize=False,
                    add_generation_prompt=False,
                )
                return formatted
            except Exception as e2:
                logger.error(
                    "시스템 역할 없이도 쌍 형식화 실패: %s",
                    type(e2).__name__,
                )
                return None
    
    def format_batch(self, pairs: list[QAPair]) -> list[dict[str, str]]:
        """QA 쌍 배치를 형식화하고 max_seq_length로 필터링합니다.
        
        매개변수
        ----------
        pairs : list[QAPair]
            형식화할 질문-답변 쌍 리스트.
        
        반환값
        -------
        list[dict[str, str]]
            HF 데이터셋 준비가 된 {"text": formatted_string} 딕셔너리 리스트.
        """
        formatted_data = []
        skipped = 0
        
        for pair in pairs:
            formatted_text = self.format_one(pair)
            
            if formatted_text is None:
                skipped += 1
                continue
            
            # 토큰 개수를 세어 max_seq_length 확인
            tokens = self.tokenizer.encode(formatted_text)
            token_count = len(tokens)
            
            if token_count > self.max_seq_length:
                skipped += 1
                logger.debug(
                    "쌍 건너뜀: %d 토큰 > max_seq_length (%d)",
                    token_count,
                    self.max_seq_length,
                )
                continue
            
            formatted_data.append({"text": formatted_text})
        
        total = len(pairs)
        accepted = len(formatted_data)
        logger.info(
            "%d/%d 쌍 형식화됨 (%d개 max_seq_length 초과 또는 실패)",
            accepted,
            total,
            skipped,
        )
        
        return formatted_data
    
    def save_training_data(
        self,
        pairs: list[QAPair],
        output_path: str | Path,
    ) -> Path:
        """쌍을 형식화하고 JSONL 학습 데이터로 저장합니다.
        
        매개변수
        ----------
        pairs : list[QAPair]
            형식화하고 저장할 질문-답변 쌍 리스트.
        output_path : str or Path
            JSONL 파일을 저장할 경로.
        
        반환값
        -------
        Path
            출력 경로 (Path 객체).
        """
        output_path = Path(output_path)
        
        # 모든 쌍 형식화
        formatted_data = self.format_batch(pairs)
        
        # JSONL로 저장 (한 줄에 하나의 JSON 객체)
        with open(output_path, "w", encoding="utf-8") as f:
            for item in formatted_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        
        logger.info(
            "%d개 학습 예제를 %s에 저장함",
            len(formatted_data),
            output_path,
        )
        
        return output_path
    
    def format_from_alpaca_file(
        self,
        input_path: str | Path,
        output_path: str | Path,
    ) -> Path:
        """Alpaca JSON 파일을 로드하고, QAPair로 변환하고, 형식화된 학습 데이터를 저장합니다.
        
        독립 실행형 사용을 위한 편의 메서드입니다.
        
        매개변수
        ----------
        input_path : str or Path
            Alpaca JSON 파일 경로 ('instruction', 'input', 'output' 키를 가진 딕셔너리 리스트).
        output_path : str or Path
            형식화된 JSONL 파일을 저장할 경로.
        
        반환값
        -------
        Path
            출력 경로 (Path 객체).
        """
        from .models import QAPair
        
        input_path = Path(input_path)
        
        # Alpaca JSON 로드
        with open(input_path, "r", encoding="utf-8") as f:
            alpaca_data = json.load(f)
        
        # QAPair 객체로 변환
        pairs = []
        for item in alpaca_data:
            # Alpaca 형식: instruction, input (선택사항), output
            instruction = item.get("instruction", "")
            input_text = item.get("input", "")
            output_text = item.get("output", "")
            
            # instruction과 input을 질문으로 결합
            question = instruction
            if input_text:
                question = f"{instruction}\n{input_text}".strip()
            
            pair = QAPair(
                question=question,
                answer=output_text,
                instruction=instruction,
            )
            pairs.append(pair)
        
        logger.info("%d개 쌍을 %s에서 로드함", len(pairs), input_path)
        
        # 형식화 및 저장
        return self.save_training_data(pairs, output_path)
