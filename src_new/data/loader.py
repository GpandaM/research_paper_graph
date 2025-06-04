import pandas as pd
from typing import Dict, List, Optional, Union
from pathlib import Path
import logging

class DataLoader:
    """Handles loading and initial processing of research paper data."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        
    def load_excel(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """Load Excel file and return DataFrame."""
        try:
            df = pd.read_excel(file_path)
            self.logger.info(f"Loaded {len(df)} records from {file_path}")
            return df
        except Exception as e:
            self.logger.error(f"Error loading Excel file: {e}")
            raise
    
    def load_csv(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """Load CSV file and return DataFrame."""
        try:
            df = pd.read_csv(file_path)
            self.logger.info(f"Loaded {len(df)} records from {file_path}")
            return df
        except Exception as e:
            self.logger.error(f"Error loading CSV file: {e}")
            raise