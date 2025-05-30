import weaviate
from weaviate.classes.init import AdditionalConfig, Timeout
from typing import List, Dict
from weaviate.classes.config import Property, DataType, Configure
import weaviate.classes as wvc

class WeaviateSetup:
    def __init__(self, host: str = "localhost", port: int = 8080, scheme: str = "http"):
        pass
    
    def create_research_paper_schema(self):
        """Create the ResearchPaper class schema"""
        with weaviate.connect_to_local(
            port=8080,
            grpc_port=50051,  # Disable gRPC
            skip_init_checks=True
        ) as client:
            # Delete existing collection if it exists
            if client.collections.exists("ResearchPaper"):
                client.collections.delete("ResearchPaper")

            # Create new collection with all properties
            research_paper = client.collections.create(
                name="ResearchPaper",
                description="Research papers with metadata",
                vectorizer_config=[
                                        Configure.NamedVectors.text2vec_ollama(
                                            name="research_paper_coll",
                                            api_endpoint="http://host.docker.internal:11434",  # If using Docker, use this to contact your local Ollama instance
                                            model="nomic-embed-text",  # The model to use, e.g. "nomic-embed-text"
                                        )
                                    ],
                properties=[
                                Property(name="title", data_type=DataType.TEXT),
                                Property(name="authors", data_type=DataType.TEXT_ARRAY),
                                Property(name="year", data_type=DataType.INT),
                                Property(name="keywords", data_type=DataType.TEXT_ARRAY),
                                Property(name="methodology", data_type=DataType.TEXT),
                                Property(name="main_findings", data_type=DataType.TEXT),
                                Property(name="equations_models", data_type=DataType.TEXT),
                                
                                # Add missing properties from your node data
                                Property(name="institutions", data_type=DataType.TEXT_ARRAY),
                                Property(name="journal", data_type=DataType.TEXT),
                                Property(name="doi", data_type=DataType.TEXT),
                                Property(name="citations_count", data_type=DataType.INT),
                            ]
            )
            print("ResearchPaper collection created successfully")

