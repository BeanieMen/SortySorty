from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import numpy.typing as npt
import face_recognition

from ..types.face import FaceMatchResult, FaceMatch, FaceCandidate
from ..helpers.math import euclidean_distance
from ..helpers.image import create_fallback_embedding
from .embedding_cache import EmbeddingCache


class FaceService:
    
    def __init__(self, threshold: float = 0.6, low_confidence_threshold: float = 0.5):
        self.threshold = threshold
        self.low_confidence_threshold = low_confidence_threshold
        self.profile_embeddings: Dict[str, List[npt.NDArray[np.float64]]] = {}
        self.cache: Optional[EmbeddingCache] = None
    
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
    
    def extract_embedding(self, image_path: Path, use_fallback: bool = True, verbose: bool = False) -> Optional[npt.NDArray[np.float64]]:
        """
        Extract face embedding from an image.
        
        Args:
            image_path: Path to image file
            use_fallback: Whether to use pixel-based fallback if face detection fails
            verbose: Whether to print detailed timing information
        
        Returns:
            128-D face embedding or None if extraction fails
        """
        import time
        t_total_start = time.perf_counter()
        
        try:
            # Load image
            t_start = time.perf_counter()
            image = face_recognition.load_image_file(str(image_path))
            t_load = time.perf_counter() - t_start
            
            # PERFORMANCE: Downscale huge images (>1600px) for faster processing
            # Face detection on 5760x3840 images is extremely slow
            height, width = image.shape[:2]
            max_dimension = max(height, width)
            
            t_resize = 0.0
            if max_dimension > 1600:
                # Downscale to max 1600px on longest side
                t_start = time.perf_counter()
                scale_factor = 1600 / max_dimension
                from PIL import Image as PILImage
                import numpy as np
                pil_img = PILImage.fromarray(image)
                new_size = (int(width * scale_factor), int(height * scale_factor))
                pil_img = pil_img.resize(new_size, PILImage.Resampling.LANCZOS)
                image = np.array(pil_img)
                t_resize = time.perf_counter() - t_start
            
            # Use RetinaFace for accurate face detection
            t_start = time.perf_counter()
            
            face_locations_list = []
            try:
                from retinaface import RetinaFace
                # RetinaFace requires file path
                faces_dict = RetinaFace.detect_faces(str(image_path))
                
                if faces_dict and isinstance(faces_dict, dict):
                    # Get first face (highest confidence)
                    face_data = list(faces_dict.values())[0]
                    x1, y1, x2, y2 = face_data['facial_area']
                    # Convert to face_recognition format (top, right, bottom, left)
                    face_locations_list = [(int(y1), int(x2), int(y2), int(x1))]
            except Exception:
                # RetinaFace not available or failed
                pass
            
            t_detect = time.perf_counter() - t_start
            
            if not face_locations_list:
                if verbose:
                    from rich import print as rprint
                    t_total = time.perf_counter() - t_total_start
                    rprint(f"  [dim]⏱  {image_path.name}: NO FACE - load={t_load:.3f}s, resize={t_resize:.3f}s, detect={t_detect:.3f}s, total={t_total:.3f}s[/dim]")
                if use_fallback:
                    return create_fallback_embedding(image_path)
                return None
            
            t_start = time.perf_counter()
            encodings = face_recognition.face_encodings(
                image,
                known_face_locations=face_locations_list,
                num_jitters=10
            )
            t_encode = time.perf_counter() - t_start
            
            t_total = time.perf_counter() - t_total_start
            
            if verbose:
                from rich import print as rprint
                rprint(f"  [cyan]⏱[/cyan]  [bold]{image_path.name}[/bold]: load=[green]{t_load:.3f}s[/green], resize=[yellow]{t_resize:.3f}s[/yellow], detect=[blue]{t_detect:.3f}s[/blue], encode=[magenta]{t_encode:.3f}s[/magenta], total=[bold green]{t_total:.3f}s[/bold green]")
            
            if len(encodings) > 0:
                # Return the first face found
                return encodings[0]
            
            # No encoding extracted - use fallback if enabled
            if use_fallback:
                return create_fallback_embedding(image_path)
            
            return None
            
        except Exception as e:
            print(f"Error extracting embedding from {image_path}: {e}")
            
            # Use fallback on error
            if use_fallback:
                return create_fallback_embedding(image_path)
            
            return None
    
    def match_face(self, embedding: npt.NDArray[np.float64]) -> FaceMatchResult:
        """
        Match a face embedding against all loaded profiles.
        
        Args:
            embedding: Face embedding to match
        
        Returns:
            FaceMatchResult with best match and alternatives
        """
        if not self.profile_embeddings:
            return FaceMatchResult(best_match=None, alternatives=[], is_ambiguous=False)
        
        candidates: List[FaceCandidate] = []
        
        # Compare against all profile embeddings using euclidean distance
        # face_recognition uses euclidean distance, convert to similarity (1 - distance)
        for profile_name, profile_embeddings in self.profile_embeddings.items():
            for ref_embedding in profile_embeddings:
                distance = euclidean_distance(embedding, ref_embedding)
                similarity = 1.0 - distance  # Convert distance to similarity
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
            # Ambiguous = similarity between low_confidence_threshold (0.45-0.5) and threshold (0.6)
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
            image = face_recognition.load_image_file(str(image_path))
            face_locations = face_recognition.face_locations(image)
            return len(face_locations)
        except Exception:
            return 0
    
    def extract_all_faces(self, image_path: Path) -> List[npt.NDArray[np.float64]]:
        """
        Extract embeddings for all faces in an image.
        
        Args:
            image_path: Path to image file
        
        Returns:
            List of face embeddings
        """
        try:
            image = face_recognition.load_image_file(str(image_path))
            encodings = face_recognition.face_encodings(image)
            return list(encodings)
        except Exception:
            return []
