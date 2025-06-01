from llama_index.graph_stores.neo4j import Neo4jGraphStore
from llama_index.core import PropertyGraphIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import StorageContext
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.query_engine import KnowledgeGraphQueryEngine
from llama_index.core import KnowledgeGraphIndex
import logging
import asyncio
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


# Custom implementation to fix missing async methods
class CustomNeo4jGraphStore(Neo4jGraphStore):
    async def aget(self, *args, **kwargs) -> List[Dict[str, Any]]:
        """Handle both properties and ids queries asynchronously"""
        # Check if 'ids' parameter is provided
        if 'ids' in kwargs:
            return await self.aget_by_ids(kwargs['ids'])
        # Otherwise use properties-based query
        properties = kwargs.get('properties', {})
        return await asyncio.to_thread(self.get, properties)

    async def aget_by_ids(self, ids: List[str]) -> List[Dict[str, Any]]:
        """Get nodes by their IDs"""
        if not ids:
            return []
        
        # Format ID list for Cypher query
        id_list = ', '.join([f'"{id_}"' for id_ in ids])
        query = f"MATCH (n) WHERE elementId(n) IN [{id_list}] RETURN n"
        
        logger.debug(f"Fetching nodes by IDs: {ids}")
        result = await asyncio.to_thread(self.query, query)
        return self._format_node_results(result)

    def _format_node_results(self, result) -> List[Dict[str, Any]]:
        """Format Neo4j node results into standard dictionary format"""
        nodes = []
        for record in result:
            node = record["n"]
            nodes.append({
                "id": node.element_id,
                "labels": list(node.labels),
                "properties": dict(node)
            })
        return nodes


class Neo4jRetrievalEngine:
    def __init__(self, neo_config: dict):
        neo4j_params = {
            "url": neo_config.get("uri"),
            "username": neo_config.get("user"),
            "password": neo_config.get("password"),
            "database": neo_config.get("database", "neo4j"),
        }

        Settings.llm = Ollama(model="mistral:7b-instruct-v0.2-q4_0")
        Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

        try:
            # Use our custom graph store implementation
            self.graph_store = CustomNeo4jGraphStore(**neo4j_params)
            self.graph_store.supports_vector_queries = False
        except Exception as e:
            raise RuntimeError(f"Failed to connect to Neo4j: {e}")

        self.storage_context = StorageContext.from_defaults(graph_store=self.graph_store)

        self.index = KnowledgeGraphIndex.load_from_storage_context(
            storage_context=self.storage_context,
            graph_store=self.graph_store,
)

        # Create query engine that uses Cypher generation
        self.query_engine = self.index.as_query_engine(
            include_text=False,
            response_mode="compact", #"tree_summarize",
            embedding_mode="none",
            llm=Settings.llm
        )

    async def query(self, prompt: str) -> str:
        try:
            logger.info(f"Processing query: {prompt}")
            # Use the async query engine directly
            response = await self.query_engine.aquery(prompt)
            return response.response
        except Exception as e:
            logger.error(f"Error during Neo4j query: {e}")
            return f"Error during Neo4j query: {e}"

    def test_connection(self) -> bool:
        try:
            result = self.graph_store.query("RETURN 1 as test")
            return bool(result)
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False

    def get_graph_stats(self) -> dict:
        try:
            stats = {}
            node_result = self.graph_store.query("MATCH (n) RETURN count(n) as node_count")
            stats['node_count'] = node_result[0]['node_count'] if node_result else 0
            rel_result = self.graph_store.query("MATCH ()-[r]->() RETURN count(r) as rel_count")
            stats['relationship_count'] = rel_result[0]['rel_count'] if rel_result else 0
            label_result = self.graph_store.query("CALL db.labels()")
            stats['node_labels'] = [record['label'] for record in label_result]
            type_result = self.graph_store.query("CALL db.relationshipTypes()")
            stats['relationship_types'] = [record['relationshipType'] for record in type_result]
            return stats
        except Exception as e:
            logger.error(f"Error getting graph stats: {e}")
            return {"error": str(e)}

    def close(self):
        try:
            if hasattr(self.graph_store, 'close'):
                self.graph_store.close()
            logger.info("Neo4j connection closed")
        except Exception as e:
            logger.warning(f"Error closing connection: {e}")

