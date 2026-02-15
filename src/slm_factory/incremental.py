"""증분 학습을 위한 문서 변경 추적기입니다."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import SLMConfig
    from .models import QAPair

from .utils import get_logger

logger = get_logger("incremental")


class IncrementalTracker:
    """문서 해시 기반 증분 변경 추적기입니다."""

    def __init__(self, config: SLMConfig) -> None:
        self.config = config

    def compute_document_hashes(
        self, doc_dir: Path, formats: list[str],
    ) -> dict[str, str]:
        """디렉토리 내 문서들의 SHA-256 해시를 계산합니다."""
        from .utils import compute_file_hash

        extensions = [
            ext if ext.startswith(".") else f".{ext}" for ext in formats
        ]
        hashes: dict[str, str] = {}
        doc_path = Path(doc_dir)
        if not doc_path.is_dir():
            return hashes
        for f in sorted(doc_path.iterdir()):
            if f.is_file() and f.suffix.lower() in extensions:
                hashes[f.name] = compute_file_hash(f)
        return hashes

    def load_saved_hashes(self, hash_file: Path) -> dict[str, str]:
        """저장된 해시 파일을 로드합니다."""
        import json

        path = Path(hash_file)
        if not path.is_file():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def save_hashes(self, hashes: dict[str, str], hash_file: Path) -> None:
        """해시를 JSON 파일로 저장합니다."""
        import json

        path = Path(hash_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(hashes, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def detect_changes(
        self, current: dict[str, str], saved: dict[str, str],
    ) -> tuple[list[str], list[str], list[str]]:
        """현재와 저장된 해시를 비교하여 변경사항을 감지합니다."""
        current_keys = set(current)
        saved_keys = set(saved)

        new_files = sorted(current_keys - saved_keys)
        deleted_files = sorted(saved_keys - current_keys)
        modified_files = sorted(
            k for k in current_keys & saved_keys if current[k] != saved[k]
        )

        return new_files, modified_files, deleted_files

    def merge_qa_pairs(
        self,
        existing: list[QAPair],
        new_pairs: list[QAPair],
        strategy: str,
    ) -> list[QAPair]:
        """기존 QA 쌍과 새 QA 쌍을 병합합니다."""
        if strategy == "replace":
            merged = list(new_pairs)
        else:
            merged = list(existing) + list(new_pairs)

        seen: set[str] = set()
        deduped: list[QAPair] = []
        for pair in merged:
            if pair.question not in seen:
                seen.add(pair.question)
                deduped.append(pair)

        return deduped

    def get_changed_files(self, doc_dir: Path) -> list[Path]:
        """변경된 파일 목록을 반환하고 해시를 업데이트합니다."""
        doc_path = Path(doc_dir)
        formats = self.config.parsing.formats
        hash_file = Path(self.config.paths.output) / self.config.incremental.hash_file

        current = self.compute_document_hashes(doc_path, formats)
        saved = self.load_saved_hashes(hash_file)
        new_files, modified_files, _deleted = self.detect_changes(current, saved)

        self.save_hashes(current, hash_file)

        changed = [doc_path / name for name in new_files + modified_files]
        logger.info(
            "Incremental: %d new, %d modified, %d deleted",
            len(new_files), len(modified_files), len(_deleted),
        )
        return changed
