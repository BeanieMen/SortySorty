from dataclasses import dataclass, field
from typing import Set, Dict, Any, List
from datetime import datetime
from pathlib import Path


@dataclass
class LearnEntry:
    source_photo: Path
    profile_name: str
    similarity: float
    learned_at: datetime
    profile_photo_path: Path
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "sourcePhoto": str(self.source_photo),
            "profileName": self.profile_name,
            "similarity": self.similarity,
            "learnedAt": self.learned_at.isoformat(),
            "profilePhotoPath": str(self.profile_photo_path),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LearnEntry":
        """Create entry from dictionary."""
        return cls(
            source_photo=Path(data["sourcePhoto"]),
            profile_name=data["profileName"],
            similarity=data["similarity"],
            learned_at=datetime.fromisoformat(data["learnedAt"]),
            profile_photo_path=Path(data["profilePhotoPath"]),
        )


@dataclass
class LearnDatabase:
    """Database tracking learned photos."""
    entries: List[LearnEntry] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "entries": [e.to_dict() for e in self.entries],
            "lastUpdated": self.last_updated.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LearnDatabase":
        """Create database from dictionary."""
        return cls(
            entries=[LearnEntry.from_dict(e) for e in data.get("entries", [])],
            last_updated=datetime.fromisoformat(data.get("lastUpdated", datetime.now().isoformat())),
        )
    
    def add_entry(self, entry: LearnEntry) -> None:
        """Add a new entry to the database."""
        self.entries.append(entry)
        self.last_updated = datetime.now()
