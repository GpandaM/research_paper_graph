from llama_index.core import Document, VectorStoreIndex, Settings

import weaviate
from weaviate.connect import ConnectionParams
import asyncio
import networkx as nx
from typing import List, Dict, Any

from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.query_engine import KnowledgeGraphQueryEngine

# from llama_index.graph_stores.nebula import NebulaGraphStore
from llama_index.graph_stores.neo4j import Neo4jGraphStore
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.weaviate import WeaviateVectorStore

from llama_index.core import PropertyGraphIndex
from llama_index.core.query_engine import RetrieverQueryEngine


class Neo4jGraphStoreWrapper(Neo4jGraphStore):
    """Wrapper to add missing attributes to Neo4jGraphStore"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Add missing attributes
        self.supports_vector_queries = False
        self.supports_structured_output = getattr(self, 'supports_structured_output', False)


class GraphQueryEngine:
    def __init__(self, neo_config: dict, weav_config: dict):
        # Neo4j setup with wrapper
        neo4j_config = {
            "url": neo_config.get("uri"),
            "username": neo_config.get("user"),
            "password": neo_config.get("password"),
            "database": neo_config.get("database"),
        }
        
        self.graph_store = Neo4jGraphStoreWrapper(**neo4j_config)
        
        # Create async client without connecting immediately
        connection_params = ConnectionParams.from_url(
            weav_config.get("url", "http://localhost:8080"),
            grpc_port = 50051,
            grpc_secure = False
        )
        
        self.weav_client = weaviate.WeaviateAsyncClient(
            connection_params=connection_params,
            additional_headers=weav_config.get("headers", {}),
            skip_init_checks=weav_config.get("skip_init_checks", False)
        )
        
        self.vector_store = WeaviateVectorStore(
            weaviate_client=self.weav_client,
            index_name="ResearchPaper"
        )
        
        Settings.llm = Ollama(model="mistral:7b-instruct-v0.2-q4_0")
        Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")
    
    
    
    async def initialize(self):
        """Initialize async components"""
        await self.weav_client.connect()
        
        # Create index asynchronously
        self.pg_index = PropertyGraphIndex.from_existing(
            property_graph_store=self.graph_store,
            vector_store=self.vector_store,
            embed_kg_nodes=True,
        )
        
        retriever = self.pg_index.as_retriever(
            similarity_top_k=5,
            traversal_depth=2,
            search_mode="hybrid",
        )
        
        self.query_engine = RetrieverQueryEngine.from_args(
            retriever=retriever,
            llm=Settings.llm
        )
    
    
    async def summarize(self, keyword: str) -> str:
        """Async method to summarize research"""
        query = f"Summarize key findings and methodologies related to {keyword}. " \
                "Include papers from the last 5 years with high citation counts."
        
        response = self.query_engine.aquery(query)
        return str(response)
    
    
    async def close(self):
        """Close async client"""
        await self.weav_client.close()

    
    
    def summarize_entire_graph(self) -> str:
        """Summarize complete graph with all references"""
        # Step 1: Generate narrative summary
        narrative_query = """
        Compose a comprehensive literature review summarizing the entire research graph.
        Structure your response with these sections:
        
        Literature Review
        
        Introduction
        [Provide an overview of the entire research domain and key themes]
        
        [Organize content by major research themes/topics - create subheadings for each]
        [For each theme: summarize key papers, methodologies, and findings]
        
        Conclusion
        [Synthesize main insights and suggest future research directions]
        
        Note: Do not include references in the narrative. We will add them separately.
        """
        narrative = str(self.query_engine.query(narrative_query))
        
        # Step 2: Get ALL papers from the graph
        papers = self.get_all_papers()
        
        # Step 3: Format references
        references = self.format_references(papers)
        
        return f"{narrative}\n\nReferences\n\n{references}"
    
    
    def get_all_papers(self) -> List[Dict]:
        """Retrieve all papers from Neo4j graph"""
        cypher_query = """
        MATCH (p:Paper)
        RETURN p.title AS title, 
               p.authors AS authors, 
               p.year AS year, 
               p.doi AS doi,
               p.journal AS journal,
               p.citations_count AS citations
        ORDER BY citations DESC
        """
        return self.graph_store.query(cypher_query)
    
    def format_references(self, papers: List[Dict]) -> str:
        """Format paper references in academic style"""
        formatted = []
        for paper in papers:
            authors = ", ".join(paper['authors']) if paper.get('authors') else "Unknown"
            year = paper.get('year', 'n.d.')
            title = paper.get('title', 'Untitled')
            journal = paper.get('journal', '')
            doi = paper.get('doi', '')
            
            # Create reference entry
            ref = f"{authors} ({year}). {title}. {journal}"
            if doi:
                ref += f". {doi}"
            formatted.append(ref)
        
        return "\n".join(formatted)



'''
class EnhancedGraphQueryEngine(GraphQueryEngine):
    def __init__(self, neo_config: dict, weaviate_config: dict):
        super().__init__(neo_config, weaviate_config)
        
        # Add hybrid query capability
        self.hybrid_engine = self._create_hybrid_engine()
    
    def _create_hybrid_engine(self):
        """Create an engine that combines graph and vector search"""
        from llama_index.core.query_engine import RetrieverQueryEngine
        from llama_index.core.retrievers import VectorIndexRetriever, KGTableRetriever
        
        # Vector retriever (Weaviate)
        vector_retriever = VectorIndexRetriever(
            index=self.vector_store,
            similarity_top_k=3
        )
        
        # Knowledge Graph retriever (Neo4j)
        kg_retriever = KGTableRetriever(
            storage_context=self.storage_context,
            query_templates={
                "keyword_search": 
                    "MATCH (p:Paper)-[:HAS_KEYWORD]->(k:Keyword) "
                    "WHERE toLower(k.name) CONTAINS toLower('{query}') "
                    "RETURN p"
            }
        )
        
        # Hybrid query engine
        return RetrieverQueryEngine.from_args(
            retriever=[vector_retriever, kg_retriever],
            response_mode="tree_summarize"
        )
    
    def query(self, question: str) -> str:
        """Handle natural language queries with hybrid retrieval"""
        response = self.hybrid_engine.query(question)
        return self._format_response(response)
    
    def _format_response(self, response) -> str:
        """Format response with references"""
        if not hasattr(response, 'source_nodes'):
            return str(response)
            
        # Extract references from source nodes
        refs = []
        for node in response.source_nodes:
            if 'title' in node.metadata:
                ref = (
                    f"{node.metadata.get('authors', ['Unknown'])[0]} et al. "
                    f"({node.metadata.get('year', 'n.d.')}). "
                    f"\"{node.metadata['title']}\""
                )
                if 'doi' in node.metadata:
                    ref += f" [DOI: {node.metadata['doi']}]"
                refs.append(ref)
        
        return f"{response}\n\nReferences:\n" + "\n".join(refs)
'''
