from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import numpy.typing as npt
import cv2
from insightface.app import FaceAnalysis

from ..types.face import FaceMatchResult, FaceMatch, FaceCandidate
from ..helpers.math import cosine_similarity
from .embedding_cache import EmbeddingCache


class FaceService:
    
    def __init__(self, threshold: float = 0.52, low_confidence_threshold: float = 0.45):
        self.threshold = threshold
        self.low_confidence_threshold = low_confidence_threshold
        self.profile_embeddings: Dict[str, List[npt.NDArray[np.float64]]] = {}
        self.cache: Optional[EmbeddingCache] = None
        self._face_app: Optional[FaceAnalysis] = None
    
    @property
    def face_app(self) -> FaceAnalysis:
        if self._face_app is None:
            self._face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
            self._face_app.prepare(ctx_id=0, det_size=(640, 640))
        return self._face_app
    
    def load_profiles(self, profiles_dir: Path, verbose: bool = False) -> None:
        from rich import print as rprint
        
        self.profile_embeddings = {}
        if not profiles_dir.exists():
            return
        
        cache_dir = profiles_dir / ".cache"
        self.cache = EmbeddingCache(cache_dir)
        
        if not self.cache.needs_update(profiles_dir):
            if verbose:
                rprint("  [dim cyan]Loading embeddings from cache...[/dim cyan]")
            cached_embeddings = self.cache.load()
            if cached_embeddings is not None:
                self.profile_embeddings = cached_embeddings
                if verbose:
                    total_embeddings = sum(len(embs) for embs in cached_embeddings.values())
                    rprint(f"  [green]✓[/green] Loaded [bold]{total_embeddings}[/bold] embeddings from cache for [bold]{len(cached_embeddings)}[/bold] profiles")
                return
        
        if verbose:
            rprint("  [yellow]Computing fresh embeddings (cache outdated or missing)...[/yellow]")
        
        self._compute_embeddings(profiles_dir, verbose)
        
        if self.profile_embeddings:
            self.cache.save(self.profile_embeddings, profiles_dir)
            if verbose:
                total_embeddings = sum(len(embs) for embs in self.profile_embeddings.values())
                rprint(f"  [green]✓[/green] Computed and cached [bold]{total_embeddings}[/bold] embeddings for [bold]{len(self.profile_embeddings)}[/bold] profiles")
    
    def _compute_embeddings(self, profiles_dir: Path, verbose: bool = False) -> None:
        for profile_path in profiles_dir.iterdir():
            if not profile_path.is_dir() or profile_path.name.startswith('.'):
                continue
            
            profile_name = profile_path.name
            embeddings = []
            
            for img_path in profile_path.glob("*"):
                if not img_path.is_file():
                    continue
                if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']:
                    continue
                
                embedding = self.extract_embedding(img_path, verbose=False)
                if embedding is not None:
                    embeddings.append(embedding)
            
            if embeddings:
                self.profile_embeddings[profile_name] = embeddings
    
    def load_from_cache_only(self, profiles_dir: Path) -> bool:
        """Load embeddings from cache only (for worker processes)."""
        self.profile_embeddings = {}
        if not profiles_dir.exists():
            return False
        
        cache_dir = profiles_dir / ".cache"
        self.cache = EmbeddingCache(cache_dir)
        cached_embeddings = self.cache.load()
        
        if cached_embeddings is not None:
            self.profile_embeddings = cached_embeddings
            return True
        return False
    
    def extract_embedding(self, image_path: Path, verbose: bool = False) -> Optional[npt.NDArray[np.float64]]:
        """Extract normalized 512-D face embedding with preprocessing."""
        import time
        t_start = time.perf_counter()
        
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                return None
            
            # Preprocessing: histogram equalization in LAB space for better illumination
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            l = cv2.equalizeHist(l)
            lab = cv2.merge([l, a, b])
            image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            faces = self.face_app.get(image)
            t_total = time.perf_counter() - t_start
            
            if not faces:
                if verbose:
                    from rich import print as rprint
                    rprint(f"  [dim]⏱  {image_path.name}: NO FACE - {t_total:.3f}s[/dim]")
                return None
            
            embedding = faces[0].embedding.astype(np.float64)
            
            # Normalize to unit vector for cosine similarity
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            if verbose:
                from rich import print as rprint
                rprint(f"  [cyan]⏱[/cyan]  [bold]{image_path.name}[/bold]: {t_total:.3f}s")
            
            return embedding
            
        except Exception as e:
            if verbose:
                print(f"Error extracting embedding from {image_path}: {e}")
            return None
    
    def match_face(self, embedding: npt.NDArray[np.float64]) -> FaceMatchResult:
        """Match embedding against profiles using cosine similarity."""
        if not self.profile_embeddings:
            return FaceMatchResult(best_match=None, alternatives=[], is_ambiguous=False)
        
        candidates: List[FaceCandidate] = []
        for profile_name, profile_embeddings in self.profile_embeddings.items():
            for ref_embedding in profile_embeddings:
                similarity = cosine_similarity(embedding, ref_embedding)
                candidates.append(FaceCandidate(name=profile_name, similarity=similarity))
        
        candidates.sort(key=lambda x: x.similarity, reverse=True)
        
        if candidates and candidates[0].similarity >= self.threshold:
            best = candidates[0]
            best_match = FaceMatch(name=best.name, similarity=best.similarity, embedding=embedding)
            is_ambiguous = best.similarity < self.threshold and best.similarity >= self.low_confidence_threshold
            
            return FaceMatchResult(
                best_match=best_match,
                alternatives=candidates[1:4],
                is_ambiguous=is_ambiguous
            )
        
        return FaceMatchResult(
            best_match=None,
            alternatives=candidates[:3],
            is_ambiguous=False
        )
    
    def detect_faces_count(self, image_path: Path) -> int:
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                return 0
            return len(self.face_app.get(image))
        except Exception:
            return 0
    
    def extract_all_faces(self, image_path: Path) -> List[npt.NDArray[np.float64]]:
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                return []
            faces = self.face_app.get(image)
            return [face.embedding.astype(np.float64) for face in faces]
        except Exception:
            return []
