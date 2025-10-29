import hashlib
from pathlib import Path
from typing import Tuple, Optional
from PIL import Image
import numpy as np
import numpy.typing as npt


def compute_hash(image_path: Path) -> str:
    sha1 = hashlib.sha1()
    
    with open(image_path, 'rb') as f:
        while chunk := f.read(8192):
            sha1.update(chunk)
    
    return sha1.hexdigest()


def get_image_metadata(image_path: Path) -> dict:
    try:
        with Image.open(image_path) as img:
            return {
                "width": img.width,
                "height": img.height,
                "format": img.format,
                "mode": img.mode,
            }
    except Exception as e:
        return {"error": str(e)}


def resize_image(image_path: Path, max_dimension: int = 1600) -> Image.Image:
    img = Image.open(image_path)
    
    if img.width <= max_dimension and img.height <= max_dimension:
        return img
    
    if img.width > img.height:
        new_width = max_dimension
        new_height = int(img.height * (max_dimension / img.width))
    else:
        new_height = max_dimension
        new_width = int(img.width * (max_dimension / img.height))
    
    return img.resize((new_width, new_height), Image.Resampling.LANCZOS)


def create_fallback_embedding(image_path: Path, size: int = 32) -> npt.NDArray[np.float64]:
    try:
        img = Image.open(image_path)
        img = img.convert('L')
        img = img.resize((size, size), Image.Resampling.LANCZOS)
        
        pixels = np.array(img, dtype=np.float64).flatten()
        mean = pixels.mean()
        normalized = (pixels - mean) / 255.0
        
        return normalized
    except Exception:
        return np.zeros(size * size, dtype=np.float64)


def is_valid_image(file_path: Path) -> bool:
    valid_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
    
    if file_path.suffix.lower() not in valid_extensions:
        return False
    
    try:
        with Image.open(file_path) as img:
            img.verify()
        return True
    except Exception:
        return False

