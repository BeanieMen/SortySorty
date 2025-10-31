from dataclasses import dataclass, field
from typing import Dict, List
from pathlib import Path


@dataclass
class OutputStructure:
    people: str = "people"
    unknown_people: str = "people/unknown"
    screenshots: str = "screenshots"
    chats: str = "screenshots/chats"
    others: str = "others"


@dataclass
class OcrConfig:
    """OCR-specific configuration."""
    text_length_threshold: int = 80
    chat_keywords: List[str] = field(default_factory=lambda: ["whatsapp", "telegram", "messenger"])
    enable_on_faces: bool = False


@dataclass
class Config:
    version: int = 1
    profiles_dir: Path = field(default_factory=lambda: Path("profiles"))
    input_dir: Path = field(default_factory=lambda: Path("photos"))
    output_dir: Path = field(default_factory=lambda: Path("output"))
    threshold: float = 0.52
    low_confidence_threshold: float = 0.45
    max_dimension: int = 1600
    concurrency: int = 3
    output_structure: OutputStructure = field(default_factory=OutputStructure)
    ocr: OcrConfig = field(default_factory=OcrConfig)
    rename_with_timestamp: bool = False
    delete_duplicates: bool = False
    fast_mode: bool = False
    store_unknown_clusters: bool = True
    verbose: bool = False

    def to_dict(self) -> Dict:
        return {
            "version": self.version,
            "profilesDir": str(self.profiles_dir),
            "inputDir": str(self.input_dir),
            "outputDir": str(self.output_dir),
            "threshold": self.threshold,
            "lowConfidenceThreshold": self.low_confidence_threshold,
            "maxDimension": self.max_dimension,
            "concurrency": self.concurrency,
            "outputStructure": {
                "people": self.output_structure.people,
                "unknownPeople": self.output_structure.unknown_people,
                "screenshots": self.output_structure.screenshots,
                "chats": self.output_structure.chats,
                "others": self.output_structure.others,
            },
            "ocr": {
                "textLengthThreshold": self.ocr.text_length_threshold,
                "chatKeywords": self.ocr.chat_keywords,
                "enableOnFaces": self.ocr.enable_on_faces,
            },
            "renameWithTimestamp": self.rename_with_timestamp,
            "deleteDuplicates": self.delete_duplicates,
            "fastMode": self.fast_mode,
            "storeUnknownClusters": self.store_unknown_clusters,
            "verbose": self.verbose,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Config":
        """Create config from dictionary."""
        output_struct = OutputStructure(
            people=data.get("outputStructure", {}).get("people", "people"),
            unknown_people=data.get("outputStructure", {}).get("unknownPeople", "people/unknown"),
            screenshots=data.get("outputStructure", {}).get("screenshots", "screenshots"),
            chats=data.get("outputStructure", {}).get("chats", "screenshots/chats"),
            others=data.get("outputStructure", {}).get("others", "others"),
        )
        
        ocr_config = OcrConfig(
            text_length_threshold=data.get("ocr", {}).get("textLengthThreshold", 80),
            chat_keywords=data.get("ocr", {}).get("chatKeywords", ["whatsapp", "telegram", "messenger"]),
            enable_on_faces=data.get("ocr", {}).get("enableOnFaces", False),
        )
        
        return cls(
            version=data.get("version", 1),
            profiles_dir=Path(data.get("profilesDir", "profiles")),
            input_dir=Path(data.get("inputDir", "photos")),
            output_dir=Path(data.get("outputDir", "output")),
            threshold=data.get("threshold", 0.52),
            low_confidence_threshold=data.get("lowConfidenceThreshold", 0.45),
            max_dimension=data.get("maxDimension", 1600),
            concurrency=data.get("concurrency", 3),
            output_structure=output_struct,
            ocr=ocr_config,
            rename_with_timestamp=data.get("renameWithTimestamp", False),
            delete_duplicates=data.get("deleteDuplicates", False),
            fast_mode=data.get("fastMode", False),
            store_unknown_clusters=data.get("storeUnknownClusters", True),
            verbose=data.get("verbose", False),
        )
