import hashlib
from pathlib import Path
from PIL import Image


def compute_hash(image_path: Path) -> str:
    """Compute SHA-1 hash of an image file for duplicate detection."""
    sha1 = hashlib.sha1()
    with open(image_path, 'rb') as f:
        while chunk := f.read(8192):
            sha1.update(chunk)
    return sha1.hexdigest()


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

