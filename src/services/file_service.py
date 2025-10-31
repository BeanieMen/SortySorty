from pathlib import Path
from typing import Dict, Set, Optional
from datetime import datetime

from ..helpers.fs import ensure_directory, copy_file
from ..helpers.image import compute_hash


class FileService:
    
    def __init__(self):
        self.hash_to_destination: Dict[str, Path] = {}
        self.processed_hashes: Set[str] = set()
    
    def reset(self) -> None:
        self.hash_to_destination.clear()
        self.processed_hashes.clear()
    
    def is_duplicate(self, source: Path) -> bool:
        file_hash = compute_hash(source)
        return file_hash in self.processed_hashes
    
    def copy_photo(
        self,
        source: Path,
        destination_dir: Path,
        filename: Optional[str] = None,
        rename_with_timestamp: bool = False
    ) -> Optional[Path]:
        file_hash = compute_hash(source)
        
        if file_hash in self.processed_hashes:
            return self.hash_to_destination.get(file_hash)
        
        if filename is None:
            filename = source.name
        
        if rename_with_timestamp:
            stem = Path(filename).stem
            suffix = Path(filename).suffix
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{stem}_{timestamp}{suffix}"
        
        ensure_directory(destination_dir)
        destination = destination_dir / filename
        
        counter = 1
        while destination.exists():
            stem = Path(filename).stem
            suffix = Path(filename).suffix
            destination = destination_dir / f"{stem}_{counter}{suffix}"
            counter += 1
        
        if copy_file(source, destination):
            self.processed_hashes.add(file_hash)
            self.hash_to_destination[file_hash] = destination
            return destination
        
        return None
    
    def get_image_files(self, directory: Path, recursive: bool = True) -> list[Path]:
        if not directory.exists():
            return []
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
        files = []
        
        pattern = "**/*" if recursive else "*"
        for file_path in directory.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                files.append(file_path)
        
        return sorted(files)
