from src_new.neo_store import Neo4jGraphStore
from src_new.graph_builder import GraphBuilder
from src_new.data.preprocessor import DataPreprocessor
from src_new.data.loader import DataLoader
from src_new.logger import setup_logger
from src_new.semantic_relation import SemanticRelationshipGenerator
from src_new.literature_review import LiteratureReviewGenerator

import pandas as pd
import logging
import argparse


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

    def test_single_node_insertion(self):
        """Test inserting a single node to isolate the issue"""
        test_node = {
            'id': 'test_paper_123',
            'type': 'PAPER',
            'title': 'Test Paper',
            'year': 2024
        }
        
        print(f"Testing node insertion: {test_node}")
        result = self.graph_store.insert_node(test_node)
        print(f"Insert result: {result} (type: {type(result)})")
        
        # Check if it's actually in the database
        verify_query = "MATCH (n:Paper {id: 'test_paper_123'}) RETURN n"
        db_result = self.graph_store.execute(verify_query)
        print(f"Database verification: {db_result}")
        
        return result, db_result

    def run(self, data_path: str): #, output_path: str = None):
        """Run the complete pipeline"""
        # Load and preprocess data
        df_processed = self.load_and_preprocess_data(data_path)
        
        ## insert single node
        # self.test_single_node_insertion()

        # Build knowledge graph
        graph_builder = GraphBuilder(self.graph_store)
        # graph_builder.build_graph(df_processed)
        
        # # Add semantic relationships
        # semantic_generator = SemanticRelationshipGenerator(self.graph_store)
        # semantic_generator.generate_embeddings()
        # semantic_generator.create_semantic_relationships()
        # ## semantic_generator.create_limitation_relationships() ## . this step takes time, to do : make it optimized 
        
        # # Generate literature review
        review_generator = LiteratureReviewGenerator(self.graph_store)
        literature_review = review_generator.generate_review()
        print(literature_review)
        
        # # Save output
        # if output_path:
        #     with open(output_path, "w") as f:
        #         json.dump(literature_review, f, indent=2)
        
        # return literature_review



def main():
    parser = argparse.ArgumentParser(description="Research Paper Knowledge Graph")
    parser.add_argument("--file", required=True, help="Path to Excel/CSV file")
    parser.add_argument("--config", default="config/config.yaml", help="Config file path")

    args = parser.parse_args()
    print(args)

    config = {"neo4j" : {
            "uri": "neo4j://localhost:7687",
            "user": "neo4j",
            "password": "Sundeep@123",
            "database": "rpaperdb"  # Add this line
            }
    }
    
    app = ResearchGraphPipeline(config)
    app.run(args.file)


if __name__ == "__main__":
    main()