from src.neo_store import Neo4jGraphStore
from src.graph_builder import GraphBuilder
from src.data.preprocessor import DataPreprocessor
from src.data.loader import DataLoader
from src.logger import setup_logger
from src.semantic import SemanticRelationshipGenerator
from src.literature_review import LiteratureReviewGenerator

import pandas as pd
import logging
import argparse

import warnings
warnings.filterwarnings("ignore")



class ResearchGraphPipeline:
    """End-to-end research knowledge graph pipeline"""
    
    def __init__(self, config: dict):
        self.config = config
        self.graph_store = Neo4jGraphStore(config["neo4j"])
        self.preprocessor = DataPreprocessor()
        self.logger = setup_logger()
        
        # Initialize components
        self.data_loader = DataLoader(self.logger)
    
    
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

        # print("\n\n\n")
        # print(df_processed.head())
        # print(df_processed.columns)
        # print("\n\n\n")
        
        return df_processed


    def run(self, data_path: str): #, output_path: str = None):
        """Run the complete pipeline"""
        ## Load and preprocess data
        df_processed = self.load_and_preprocess_data(data_path)
        
        ## if you really want to drop the exitsing graph first 
        # self.graph_store.execute("MATCH (n) DETACH DELETE n")
        # self.logger.info("="*50)
        # self.logger.warning(f" {'!'*30} Deleted existing graph!!")
        # self.logger.info("="*50)

        # ## Build knowledge graph
        # graph_builder = GraphBuilder(self.graph_store)
        # graph_builder.build_graph(df_processed)
        
        # # # Add semantic relationships
        # semantic_generator = SemanticRelationshipGenerator(self.graph_store)
        # semantic_generator.generate_embeddings()
        # semantic_generator.create_semantic_relationships(similarity_threshold=0.78, top_k=40)
        # semantic_generator.create_limitation_relationships(importance_percent=0.10, top_k=2, similarity_threshold=0.80) ## increase importance_percent for better future direction in summary

        
        # # Final database stats
        # self.logger.info("PHASE 2: Second layer database statistics...")
        # self.graph_store.log_database_stats()
        
        # self.logger.info("="*50)
        # self.logger.info("Second Phase GRAPH CONSTRUCTION COMPLETED SUCCESSFULLY")
        # self.logger.info("="*50)

        
        # # Generate literature review
        review_generator = LiteratureReviewGenerator(self.graph_store)
        literature_review = review_generator.generate_review()
        # print(literature_review)



def main():
    parser = argparse.ArgumentParser(description="Research Paper Knowledge Graph")
    parser.add_argument("--file", required=True, help="Path to Excel/CSV file")

    args = parser.parse_args()
    # print(args)

    config = {"neo4j" : {
            "uri": "neo4j://localhost:7687",
            "user": "neo4j",
            "password": "Sundeep@123",
            "database": "rpaperdb"
            }
    }
    
    app = ResearchGraphPipeline(config)
    app.run(args.file)


if __name__ == "__main__":
    main()