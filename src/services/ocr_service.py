from pathlib import Path
from typing import List
import pytesseract
from PIL import Image

from ..types.ocr import OcrResult
from ..types.config import OcrConfig


class OcrService:
    
    def __init__(self, config: OcrConfig):
        self.config = config
    
    def extract_text(self, image_path: Path) -> OcrResult:
        try:
            image = Image.open(image_path)
            text = pytesseract.image_to_string(image)
            confidence = 0.8 if text.strip() else 0.0
            is_screenshot = len(text.strip()) >= self.config.text_length_threshold
            
            is_chat = False
            if is_screenshot:
                text_lower = text.lower()
                is_chat = any(keyword in text_lower for keyword in self.config.chat_keywords)
            
            return OcrResult(
                text=text,
                confidence=confidence,
                is_screenshot=is_screenshot,
                is_chat=is_chat
            )
            
        except Exception as e:
            print(f"Error performing OCR on {image_path}: {e}")
            return OcrResult(
                text="",
                confidence=0.0,
                is_screenshot=False,
                is_chat=False
            )
    
    def is_screenshot(self, image_path: Path) -> bool:
        result = self.extract_text(image_path)
        return result.is_screenshot
    
    def is_chat_screenshot(self, image_path: Path) -> bool:
        result = self.extract_text(image_path)
        return result.is_chat
