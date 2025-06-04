import logger
from .utils.config import ConfigManager
from .data.loader import DataLoader
from .data.preprocessor import DataPreprocessor
from .utils.logger import setup_logger

class ResearchGraphApplication:
    """Main application class for the research paper knowledge graph."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = ConfigManager(config_path)
        self.logger = setup_logger()
        
        # Initialize components
        self.data_loader = DataLoader(self.logger)
        self.preprocessor = DataPreprocessor()
        
    def load_and_preprocess_data(self, file_path: str):
        """Load and preprocess research paper data."""
        self.logger.info(f"Loading data from {file_path}")
        
        # Load data
        if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            df = self.data_loader.load_excel(file_path)
        else:
            df = self.data_loader.load_csv(file_path)
        
        # Preprocess
        df_processed = self.preprocessor.preprocess_dataframe(df)
        self.logger.info("Data preprocessing completed")
        
        return df_processed

    # --------------------------------------------------------------------------------------------- ##
    def main(self, file_path:str): 
        ## Load and preprocess data
        df = self.load_and_preprocess_data(file_path)

        # --------------------------------------------------------------------------------------------- ##
        neo4j_config = {
            "uri": "neo4j://localhost:7687",
            "user": "neo4j",
            "password": "Sundeep@123",
            "database": "r_paper_db"  
            }