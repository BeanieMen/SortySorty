from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
from datetime import datetime
from pathlib import Path
import numpy.typing as npt
import numpy as np


@dataclass
class ProcessedFile:
    source: Path
    destination: Optional[Path]
    action: str  # "copied", "skipped-duplicate", "error"
    matched_person: Optional[str] = None
    similarity: Optional[float] = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": str(self.source),
            "destination": str(self.destination) if self.destination else None,
            "action": self.action,
            "matchedPerson": self.matched_person,
            "similarity": self.similarity,
            "error": self.error,
        }


@dataclass
class ReviewEntry:
    image_path: Path
    reason: str  # "ambiguous-face", "low-confidence", "multiple-faces"
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "imagePath": str(self.image_path),
            "reason": self.reason,
            "candidates": self.candidates,
        }


@dataclass
class FaceCluster:
    members: List[npt.NDArray[np.float64]] = field(default_factory=list)
    representative: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "members": [member.tolist() for member in self.members],
            "representative": self.representative,
            "size": len(self.members),
        }


@dataclass
class ScanReport:
    processed: int = 0
    copied: int = 0
    duplicates: int = 0
    ambiguous: int = 0
    unknown_faces: int = 0
    errors: int = 0
    processed_files: List[ProcessedFile] = field(default_factory=list)
    clusters: List[FaceCluster] = field(default_factory=list)
    review_entries: List[ReviewEntry] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "processed": self.processed,
            "copied": self.copied,
            "duplicates": self.duplicates,
            "ambiguous": self.ambiguous,
            "unknownFaces": self.unknown_faces,
            "errors": self.errors,
            "processedFiles": [f.to_dict() for f in self.processed_files],
            "clusters": [c.to_dict() for c in self.clusters],
            "reviewEntries": [r.to_dict() for r in self.review_entries],
            "timestamp": self.timestamp.isoformat(),
        }
