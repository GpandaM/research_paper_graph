import yaml
from typing import Dict, Any
from pathlib import Path

class ConfigManager:
    """Manages application configuration."""
    
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            return self._default_config()
    
    
    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            'llm': {
                'model': 'gpt-3.5-turbo',
                'embedding_model': 'text-embedding-ada-002',
                'temperature': 0.1
            },
            'graph': {
                'similarity_threshold': 0.3,
                'max_keywords_per_paper': 10
            },
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            }
        }
    
    def get(self, key: str, default=None):
        """Get configuration value by key."""
        keys = key.split('.')
        value = self.config
        for k in keys:
            value = value.get(k, {})
        return value if value != {} else default
