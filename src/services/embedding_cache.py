import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import numpy.typing as npt


class EmbeddingCache:
    
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_file = cache_dir / "embeddings.json"
        self.metadata_file = cache_dir / "metadata.json"
        
    def _compute_file_hash(self, file_path: Path) -> str:
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def _get_profile_metadata(self, profiles_dir: Path) -> Dict[str, Dict[str, str]]:
        metadata: Dict[str, Dict[str, str]] = {}
        
        if not profiles_dir.exists():
            return metadata
        
        for profile_path in profiles_dir.iterdir():
            if not profile_path.is_dir():
                continue
            
            profile_name = profile_path.name
            metadata[profile_name] = {}
            
            for img_path in profile_path.glob("*"):
                if not img_path.is_file():
                    continue
                
                if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']:
                    continue
                
                file_hash = self._compute_file_hash(img_path)
                metadata[profile_name][img_path.name] = file_hash
        
        return metadata
    
    def needs_update(self, profiles_dir: Path) -> bool:
        if not self.cache_file.exists() or not self.metadata_file.exists():
            return True
        
        # Load old metadata
        try:
            with open(self.metadata_file, 'r') as f:
                old_metadata = json.load(f)
        except Exception:
            return True
        
        # Get current metadata
        current_metadata = self._get_profile_metadata(profiles_dir)
        
        # Compare metadata
        return old_metadata != current_metadata
    
    def save(self, embeddings: Dict[str, List[npt.NDArray[np.float64]]], profiles_dir: Path) -> None:
        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_embeddings: Dict[str, List[List[float]]] = {}
        for person_name, embedding_list in embeddings.items():
            serializable_embeddings[person_name] = [
                emb.tolist() for emb in embedding_list
            ]
        
        # Save embeddings
        with open(self.cache_file, 'w') as f:
            json.dump(serializable_embeddings, f)
        
        # Save metadata
        metadata = self._get_profile_metadata(profiles_dir)
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def load(self) -> Optional[Dict[str, List[npt.NDArray[np.float64]]]]:
        if not self.cache_file.exists():
            return None
        
        try:
            with open(self.cache_file, 'r') as f:
                serialized = json.load(f)
            
            # Convert lists back to numpy arrays
            embeddings: Dict[str, List[npt.NDArray[np.float64]]] = {}
            for person_name, embedding_list in serialized.items():
                embeddings[person_name] = [
                    np.array(emb, dtype=np.float64) for emb in embedding_list
                ]
            
            return embeddings
        except Exception:
            return None
