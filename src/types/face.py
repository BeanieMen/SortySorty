from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import numpy.typing as npt


@dataclass
class FaceMatch:
    name: str
    similarity: float
    embedding: npt.NDArray[np.float64]


@dataclass
class FaceCandidate:
    """Represents a potential face match."""
    name: str
    similarity: float


@dataclass
class FaceMatchResult:
    """Complete face matching result with best match and alternatives."""
    best_match: Optional[FaceMatch]
    alternatives: List[FaceCandidate]
    is_ambiguous: bool = False
    
    @property
    def has_match(self) -> bool:
        """Check if a match was found."""
        return self.best_match is not None
