import os
import shutil
from pathlib import Path
from typing import List, Optional


def ensure_directory(directory: Path) -> None:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        directory: Path to directory
    """
    directory.mkdir(parents=True, exist_ok=True)


def file_exists(file_path: Path) -> bool:
    """
    Check if a file exists.
    
    Args:
        file_path: Path to file
    
    Returns:
        True if file exists, False otherwise
    """
    return file_path.exists() and file_path.is_file()


def copy_file(source: Path, destination: Path, overwrite: bool = False) -> bool:
    """
    Copy a file from source to destination.
    
    Args:
        source: Source file path
        destination: Destination file path
        overwrite: Whether to overwrite existing files
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Ensure destination directory exists
        ensure_directory(destination.parent)
        
        # Check if destination exists
        if destination.exists() and not overwrite:
            return False
        
        shutil.copy2(source, destination)
        return True
    except Exception:
        return False


def move_file(source: Path, destination: Path, overwrite: bool = False) -> bool:
    """
    Move a file from source to destination.
    
    Args:
        source: Source file path
        destination: Destination file path
        overwrite: Whether to overwrite existing files
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Ensure destination directory exists
        ensure_directory(destination.parent)
        
        # Check if destination exists
        if destination.exists() and not overwrite:
            return False
        
        shutil.move(str(source), str(destination))
        return True
    except Exception:
        return False


def list_files(directory: Path, extensions: Optional[List[str]] = None, recursive: bool = True) -> List[Path]:
    """
    List files in a directory.
    
    Args:
        directory: Directory to search
        extensions: List of file extensions to filter (e.g., ['.jpg', '.png'])
        recursive: Whether to search recursively
    
    Returns:
        List of file paths
    """
    if not directory.exists() or not directory.is_dir():
        return []
    
    pattern = "**/*" if recursive else "*"
    files = []
    
    for file_path in directory.glob(pattern):
        if not file_path.is_file():
            continue
        
        if extensions is None or file_path.suffix.lower() in extensions:
            files.append(file_path)
    
    return sorted(files)


def get_unique_filename(file_path: Path) -> Path:
    """
    Get a unique filename by appending a number if file already exists.
    
    Args:
        file_path: Desired file path
    
    Returns:
        Unique file path
    """
    if not file_path.exists():
        return file_path
    
    stem = file_path.stem
    suffix = file_path.suffix
    parent = file_path.parent
    counter = 1
    
    while True:
        new_path = parent / f"{stem}_{counter}{suffix}"
        if not new_path.exists():
            return new_path
        counter += 1
