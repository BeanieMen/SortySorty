from dataclasses import dataclass
from typing import List


@dataclass
class OcrResult:
    """Result of OCR text extraction."""
    text: str
    confidence: float
    is_screenshot: bool
    is_chat: bool
    
    @property
    def has_text(self) -> bool:
        """Check if meaningful text was extracted."""
        return len(self.text.strip()) > 0
