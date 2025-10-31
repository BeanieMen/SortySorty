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
        
        # Initialize InsightFace once (lazy loading)
        self._face_app: Optional[FaceAnalysis] = None
    
    @property
    def face_app(self) -> FaceAnalysis:
        """Lazy-load InsightFace model."""
        if self._face_app is None:
            self._face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
            self._face_app.prepare(ctx_id=0, det_size=(640, 640))
        return self._face_app
    
    def load_profiles(self, profiles_dir: Path, verbose: bool = False) -> None:
        from rich import print as rprint
        
        self.profile_embeddings = {}
        
        if not profiles_dir.exists():
            return
        
        # Initialize cache
        cache_dir = profiles_dir / ".cache"
        self.cache = EmbeddingCache(cache_dir)
        
        # Check if we can use cached embeddings
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
        
        # Cache miss or outdated - compute embeddings
        if verbose:
            rprint("  [yellow]Computing fresh embeddings (cache outdated or missing)...[/yellow]")
        
        self._compute_embeddings(profiles_dir, verbose)
        
        # Save to cache
        if self.profile_embeddings:
            self.cache.save(self.profile_embeddings, profiles_dir)
            if verbose:
                total_embeddings = sum(len(embs) for embs in self.profile_embeddings.values())
                rprint(f"  [green]✓[/green] Computed and cached [bold]{total_embeddings}[/bold] embeddings for [bold]{len(self.profile_embeddings)}[/bold] profiles")
    
    def _compute_embeddings(self, profiles_dir: Path, verbose: bool = False) -> None:
        """
        Compute embeddings for all profile images.
        
        Args:
            profiles_dir: Directory containing profile subdirectories
            verbose: Whether to print detailed information
        """
        # Iterate through each profile directory
        for profile_path in profiles_dir.iterdir():
            if not profile_path.is_dir() or profile_path.name.startswith('.'):
                continue
            
            profile_name = profile_path.name
            embeddings = []
            
            # Load all images in this profile
            for img_path in profile_path.glob("*"):
                if not img_path.is_file():
                    continue
                
                # Skip non-image files and metadata
                if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']:
                    continue
                
                # Extract embedding
                embedding = self.extract_embedding(img_path, verbose=False)
                if embedding is not None:
                    embeddings.append(embedding)
            
            if embeddings:
                self.profile_embeddings[profile_name] = embeddings
    
    def load_from_cache_only(self, profiles_dir: Path) -> bool:
        """
        Load embeddings from cache without computing if missing.
        Used by worker processes in parallel mode.
        
        Args:
            profiles_dir: Directory containing profile subdirectories
            
        Returns:
            True if loaded successfully, False otherwise
        """
        self.profile_embeddings = {}
        
        if not profiles_dir.exists():
            return False
        
        # Initialize cache
        cache_dir = profiles_dir / ".cache"
        self.cache = EmbeddingCache(cache_dir)
        
        # Try to load from cache
        cached_embeddings = self.cache.load()
        if cached_embeddings is not None:
            self.profile_embeddings = cached_embeddings
            return True
        
        return False
    
    def extract_embedding(self, image_path: Path, verbose: bool = False) -> Optional[npt.NDArray[np.float64]]:
        """
        Extract face embedding from an image using InsightFace with preprocessing.
        
        Args:
            image_path: Path to image file
            verbose: Whether to print detailed timing information
        
        Returns:
            Normalized 512-D face embedding or None if extraction fails
        """
        import time
        t_start = time.perf_counter()
        
        try:
            # Load image with OpenCV (BGR format)
            image = cv2.imread(str(image_path))
            if image is None:
                return None
            
            # Apply image preprocessing for better face detection
            # 1. Convert to RGB for better color representation
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 2. Apply histogram equalization to improve contrast
            # Convert to LAB color space for better illumination normalization
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            l = cv2.equalizeHist(l)
            lab = cv2.merge([l, a, b])
            image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            
            # Convert back to BGR for InsightFace
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Use InsightFace for face detection and embedding extraction
            faces = self.face_app.get(image)
            t_total = time.perf_counter() - t_start
            
            if not faces:
                if verbose:
                    from rich import print as rprint
                    rprint(f"  [dim]⏱  {image_path.name}: NO FACE - {t_total:.3f}s[/dim]")
                return None
            
            # Get the first face (highest confidence)
            embedding = faces[0].embedding.astype(np.float64)
            
            # Normalize embedding to unit vector for better comparison
            # This makes distance calculations more consistent
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
        """
        Match a face embedding against all loaded profiles using cosine similarity.
        
        Args:
            embedding: Normalized face embedding to match
        
        Returns:
            FaceMatchResult with best match and alternatives
        """
        if not self.profile_embeddings:
            return FaceMatchResult(best_match=None, alternatives=[], is_ambiguous=False)
        
        candidates: List[FaceCandidate] = []
        
        # Compare against all profile embeddings using cosine similarity
        # Cosine similarity ranges from -1 to 1, where 1 is identical
        # This is more accurate for normalized embeddings
        for profile_name, profile_embeddings in self.profile_embeddings.items():
            for ref_embedding in profile_embeddings:
                similarity = cosine_similarity(embedding, ref_embedding)
                candidates.append(FaceCandidate(name=profile_name, similarity=similarity))
        
        # Sort by similarity (highest first)
        candidates.sort(key=lambda x: x.similarity, reverse=True)
        
        # Check if we have a match above threshold
        if candidates and candidates[0].similarity >= self.threshold:
            best = candidates[0]
            best_match = FaceMatch(
                name=best.name,
                similarity=best.similarity,
                embedding=embedding
            )
            
            # Check if match is ambiguous
            # Ambiguous = similarity between low_confidence_threshold (0.45) and threshold (0.52)
            # This means we're uncertain about the match
            is_ambiguous = best.similarity < self.threshold and best.similarity >= self.low_confidence_threshold
            
            return FaceMatchResult(
                best_match=best_match,
                alternatives=candidates[1:4],  # Top 3 alternatives
                is_ambiguous=is_ambiguous
            )
        
        # No match found
        return FaceMatchResult(
            best_match=None,
            alternatives=candidates[:3],  # Top 3 candidates for review
            is_ambiguous=False
        )
    
    def detect_faces_count(self, image_path: Path) -> int:
        """
        Count the number of faces in an image.
        
        Args:
            image_path: Path to image file
        
        Returns:
            Number of faces detected
        """
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                return 0
            faces = self.face_app.get(image)
            return len(faces)
        except Exception:
            return 0
    
    def extract_all_faces(self, image_path: Path) -> List[npt.NDArray[np.float64]]:
        """
        Extract embeddings for all faces in an image.
        
        Args:
            image_path: Path to image file
        
        Returns:
            List of face embeddings (512-D each)
        """
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                return []
            faces = self.face_app.get(image)
            return [face.embedding.astype(np.float64) for face in faces]
        except Exception:
            return []
