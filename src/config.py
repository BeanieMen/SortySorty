import json
from pathlib import Path
from typing import Optional

from .types.config import Config


class ConfigManager:
    """Manager for loading and saving configuration."""
    
    @staticmethod
    def load(config_path: Path) -> Config:
        """
        Load configuration from JSON file.
        
        Args:
            config_path: Path to config file
        
        Returns:
            Config object
        """
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            data = json.load(f)
        
        return Config.from_dict(data)
    
    @staticmethod
    def save(config: Config, config_path: Path) -> None:
        """
        Save configuration to JSON file.
        
        Args:
            config: Config object
            config_path: Path to save config file
        """
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            json.dump(config.to_dict(), f, indent=2)
    
    @staticmethod
    def create_default(base_dir: Path) -> Config:
        """
        Create default configuration for a directory.
        
        Args:
            base_dir: Base directory for the project
        
        Returns:
            Default Config object
        """
        return Config(
            profiles_dir=base_dir / "profiles"
        )
