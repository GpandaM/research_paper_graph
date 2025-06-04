from .weaviate_schema import WeaviateSetup
from ..graph.nodes import RichPaperNode


import weaviate
from typing import Dict, List
import pandas as pd

import logging
from time import time
from tqdm import tqdm  # For progress bar (optional)



from weaviate.classes.config import Property, DataType, Configure
import weaviate.classes as wvc

from uuid import UUID



class WeaviateVectorStore:
    def __init__(self):
        # Initialize schema manager but don't connect yet
        pass
        
    def _get_client(self):
        """Helper method to get a client connection"""
        return weaviate.connect_to_local(
            port=8080,
            grpc_port= 50051,  # Use 0 instead of None to properly disable gRPC
            skip_init_checks=True
        )


    def insert_paper_to_weaviate(self, paper_node: RichPaperNode) -> bool:
        """Insert a single paper into Weaviate with robust error handling"""
        try:
            with self._get_client() as client:
                # Safely get attributes with fallbacks
                data_object = {
                    "title": getattr(paper_node, 'title', "No title provided"),
                    "authors": getattr(paper_node, 'authors', []),
                    "year": getattr(paper_node, 'year', None),
                    "keywords": getattr(paper_node, 'keywords', []),
                    "equations_models": getattr(paper_node, 'equations_models', ""),
                    "methodology": getattr(paper_node, 'methodology', ""),
                    "main_findings": getattr(paper_node, 'main_findings', ""),
                    "institutions": getattr(paper_node, 'institutions', []),
                    "journal": getattr(paper_node, 'journal', ""),
                    "doi": getattr(paper_node, 'doi', ""),
                    "citations_count": getattr(paper_node, 'citations_count', 0)
                }
                
                # Remove None values to avoid schema violations
                data_object = {k: v for k, v in data_object.items() if v is not None}

                print(data_object)
                print('\n')
                
                research_papers = client.collections.get("ResearchPaper")
                results = research_papers.data.insert(properties=data_object)

                # print("="*100)
                # print(results)

                # from uuid import UUID

                # if not isinstance(results, UUID):
                #     raise ValueError("Expected UUID object, got something else")
                
                return True
                
        except Exception as e:
            print(f"Error inserting paper {getattr(paper_node, 'id', 'unknown')}: {e}")
            return False



    def batch_insert_papers(self, paper_nodes: list, batch_size: int = 20):
        """Batch-insert all papers into the ResearchPaper collection."""
        with self._get_client() as client:
            coll = client.collections.get("ResearchPaper")

            # 1️⃣ Open a fixed-size batch on the collection
            with coll.batch.fixed_size(batch_size=batch_size) as batch:
                for paper in paper_nodes:
                    props = {
                        "title": paper.title or "No title",
                        "authors": paper.authors or [],
                        "year": paper.year,
                        "keywords": paper.keywords or [],
                        "methodology": paper.methodology or "",
                        "main_findings": paper.main_findings or "",
                        "equations_models": paper.equations_models or "",
                        "institutions": paper.institutions or [],
                        "journal": paper.journal or "",
                        "doi": paper.doi or "",
                        "citations_count": paper.citations_count or 0
                    }
                    batch.add_object(properties=props)

                    # optional: bail out if too many errors
                    if batch.number_errors > 5:
                        print("Too many errors—stopping batch early")
                        break

            # 2️⃣ After the context exits, you can inspect failures
            if coll.batch.failed_objects:
                print("Failed inserts:", len(coll.batch.failed_objects))
                print("First failure:", coll.batch.failed_objects[0])
            else:
                print("✅ All batch objects inserted cleanly.")

            ## list all classes/collections
            # for collection in client.collections.list_all():
            #     print(collection.name)




    def test_single_insert(self):
        # with weaviate.connect_to_local(port=8080, grpc_port=50051) as client:
        with self._get_client() as client:
            research_papers = client.collections.get("ResearchPaper")

            data = {
                "title": "test 2 : Topology-Aware Embeddings for Geometric Data",
                "authors": ["Alice Smith", "Bob Zhang", "Others"],
                "year": 2023,
                "keywords": ["topology", "embedding"],
                "methodology": "We propose a novel persistent homology-based approach...",
                "main_findings": "Significant gains in geometric representation tasks.",
                "equations_models": "Laplace-Beltrami eigenfunctions, Betti numbers...",
                "institutions": ["MIT", "Stanford"],
                "journal": "NeurIPS",
                "doi": "10.5555/1234567",
                "citations_count": 15
            }

            uid = research_papers.data.insert(properties=data)
            print("Inserted with UUID:", uid)
        
    def peek_a_boo(self):
        # with weaviate.connect_to_local(port=8080, grpc_port=50051) as client:
        with self._get_client() as client:
            research_papers = client.collections.get("ResearchPaper")

            result = research_papers.query.fetch_objects(limit=5)

            if not result.objects:
                print("No data found")
            else:
                for i, paper in enumerate(result.objects, 1):
                    print(f"\nPaper #{i}")
                    print(f"Title: {paper.properties['title']}")
                    print(f"Authors: {paper.properties.get('authors')}")



    def verify_weaviate_data(self, sample_size=5):
        """Check inserted data by fetching sample records"""
        with self._get_client() as client:
            research_papers = client.collections.get("ResearchPaper")

            # Fetch a page of results
            sample_page = research_papers.query.fetch_objects(limit=sample_size)
            
            if not sample_page.objects:
                print("No data found in Weaviate.")
                return
            
            for i, paper in enumerate(sample_page.objects, 1):
                print(f"\nPaper #{i}:")
                print(f"ID: {paper.uuid}")
                print(f"Title: {paper.properties.get('title', 'N/A')}")
                print(f"Authors: {', '.join(paper.properties.get('authors', []))}")
                print(f"Year: {paper.properties.get('year', 'N/A')}")
                print(f"DOI: {paper.properties.get('doi', 'N/A')}")
                print(f"Methodology: {paper.properties.get('methodology', 'N/A')[:100]}...")



    def count_inserted_papers(self):
        """Verify total count of inserted papers"""
        '''
        resp = (
            self._get_client()
                .query
                .aggregate("ResearchPaper")
                .with_meta_count()   # tells GraphQL to include the total count
                .do()
        )
        count = resp["data"]["Aggregate"]["ResearchPaper"][0]["meta"]["count"]
        print("GraphQL count:", count)
        '''
        with self._get_client() as client:
            research_papers = client.collections.get("ResearchPaper")
            count = research_papers.aggregate.over_all(total_count=True).total_count
            print(f"\nTotal papers in Weaviate: {count}")
            return count
        
        
    def search_paper_by_title(self, title_fragment):
        """Search for papers containing text in title"""
        with self._get_client() as client:
            research_papers = client.collections.get("ResearchPaper")
            results = research_papers.query.bm25(
                query=title_fragment,
                query_properties=["title"],
                limit=3
            )
            
            print(f"\nSearch results for '{title_fragment}':")
            for i, paper in enumerate(results.objects, 1):
                print(f"{i}. {paper.properties['title']} (Year: {paper.properties.get('year', 'N/A')})")


    def check_vector_embeddings(self, paper_id):
        """Verify if vectors were properly generated"""
        with self._get_client() as client:
            research_papers = client.collections.get("ResearchPaper")
            paper = research_papers.query.fetch_object_by_id(paper_id)
            
            if paper.vector:
                print(f"\nVector embedding exists (dimension: {len(paper.vector)})")
                print(f"First 5 vector values: {paper.vector[:5]}")
            else:
                print("No vector embedding found")


    def generate_data_quality_report(self):
        """Comprehensive check of inserted data"""
        with self._get_client() as client:
            research_papers = client.collections.get("ResearchPaper")
            
            # Get all properties from schema
            schema = client.collections.get("ResearchPaper").config.get()
            properties = [prop.name for prop in schema.properties]
            
            print("\n=== Data Quality Report ===")
            print(f"Available properties: {properties}")
            
            # Check property population rates
            total = research_papers.aggregate.over_all(total_count=True).total_count
            print(f"\nProperty Population Rates (out of {total} papers):")
            
            for prop in properties:
                populated = research_papers.aggregate.over_all(
                    total_count=True,
                    where_filter={
                        "path": [prop],
                        "operator": "IsNull",
                        "valueBoolean": False
                    }
                ).total_count
                print(f"{prop}: {populated}/{total} ({populated/total:.1%})")