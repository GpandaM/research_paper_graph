from llama_index.core import Document, VectorStoreIndex, Settings

import networkx as nx
from typing import List, Dict, Any

from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

from llama_index.core import StorageContext
from llama_index.core.query_engine import KnowledgeGraphQueryEngine

# from llama_index.graph_stores.nebula import NebulaGraphStore
from llama_index.graph_stores.neo4j import Neo4jGraphStore
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.weaviate import WeaviateVectorStore

class GraphQueryEngine:
    def __init__(self, neo_config: dict, weaviate_config: dict):
        # Neo4j setup
        neo4j_config = {
            "uri": neo_config.get("uri"),
            "user": neo_config.get("user"),
            "password": neo_config.get("password"),
            "database": neo_config.get("database"),
        }
        self.graph_store = Neo4jGraphStore(**neo4j_config)
        
        # Weaviate setup
        self.vector_store = WeaviateVectorStore(
            weaviate_config.get('url'),
            weaviate_config.get('index_name')
        )
        
        storage_context = StorageContext.from_defaults(
            graph_store=self.graph_store,
            vector_store=self.vector_store
        )
        
        # Set global configuration
        Settings.llm = Ollama(model="mistral:7b-instruct")
        Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

        # Create query engine
        self.query_engine = KnowledgeGraphQueryEngine(
            storage_context=storage_context,
            include_text=True
        )
    
    
    def summarize(self, keyword: str) -> str:
        """Summarize research related to a specific keyword"""
        query = f"""
        "Summarize key findings and methodologies related to {keyword}. 
        Include papers from the last 5 years with high citation counts."
        """
        response = self.query_engine.query(query)
        return str(response)
    
    
    def compare_methodologies(self, method1: str, method2: str) -> str:
        """Compare two methodologies"""
        query = f"""
        Compare the {method1} and {method2} methodologies in terms of:
        - Application areas
        - Performance metrics
        - Limitations
        - Recent developments
        """
        return str(self.query_engine.query(query))
    
    
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
class LlamaIndexIntegrator:
    """Integrates NetworkX graph with LlamaIndex for LLM-powered querying using Ollama."""
    
    def __init__(self, 
                 llm_model: str = "mistral:7b-instruct-v0.2-q4_0",
                 embedding_model: str = "nomic-embed-text",
                 ollama_base_url: str = "http://localhost:11434"):
        """
        Initialize with Ollama models.
        
        Args:
            llm_model: Ollama model name for text generation
            embedding_model: Ollama model name for embeddings
            ollama_base_url: Ollama server URL
        """
        # Configure LlamaIndex settings with Ollama
        Settings.llm = Ollama(
            model=llm_model,
            base_url=ollama_base_url,
            request_timeout=300.0,
            temperature=0.1
        )
        
        Settings.embed_model = OllamaEmbedding(
            model_name=embedding_model,
            base_url=ollama_base_url,
            ollama_additional_kwargs={"mirostat": 0}
        )
        
        self.node_parser = SimpleNodeParser.from_defaults()
        self.index = None
    
    def convert_graph_to_documents(self, graph: nx.MultiDiGraph) -> List[Document]:
        """Convert NetworkX graph to LlamaIndex documents."""
        documents = []
        
        # Convert nodes to documents
        for node_id, node_data in graph.nodes(data=True):
            if node_data['type'] == 'paper':
                content = self._create_paper_content(node_id, node_data, graph)
                doc = Document(
                    text=content,
                    metadata={
                        'node_id': node_id,
                        'type': node_data['type'],
                        'title': node_data.get('title', ''),
                        'year': node_data.get('year', 0)
                    }
                )
                documents.append(doc)
        
        return documents
    
    def _create_paper_content(self, paper_id: str, paper_data: Dict,
                             graph: nx.MultiDiGraph) -> str:
        """Create rich text content for a paper including its relationships."""
        
        content_parts = [
            f"Title: {paper_data.get('title', 'Unknown')}",
            f"Year: {paper_data.get('year', 'Unknown')}",
            f"Journal: {paper_data.get('journal', 'Unknown')}"
        ]
        
        # Add methodology and findings if available
        metadata = paper_data.get('metadata', {})
        if 'methodology' in metadata:
            content_parts.append(f"Methodology: {metadata['methodology']}")
        if 'main_findings' in metadata:
            content_parts.append(f"Main Findings: {metadata['main_findings']}")
        if 'strengths' in metadata:
            content_parts.append(f"Strengths: {metadata['strengths']}")
        if 'limitations' in metadata:
            content_parts.append(f"Limitations: {metadata['limitations']}")
        if 'application_area' in metadata:
            content_parts.append(f"Application Area: {metadata['application_area']}")
        
        # Add related entities
        keywords = []
        authors = []
        similar_papers = []
        
        # Check if paper_id exists in graph
        if paper_id not in graph:
            return "\n".join(content_parts)
        
        for neighbor_id in graph.neighbors(paper_id):
            neighbor_data = graph.nodes[neighbor_id]
            
            if neighbor_data['type'] == 'keyword':
                keywords.append(neighbor_data.get('display_name', neighbor_data.get('name', '')))
            elif neighbor_data['type'] == 'author':
                authors.append(neighbor_data.get('name', ''))
        
        # Find similar papers through edges
        for source, target, edge_data in graph.edges(paper_id, data=True):
            if edge_data.get('type') == 'SIMILAR_TO':
                similar_paper_data = graph.nodes[target]
                similarity_score = edge_data.get('weight', 0.0)
                similar_papers.append(
                    f"{similar_paper_data.get('title', 'Unknown')} (similarity: {similarity_score:.2f})"
                )
        
        if keywords:
            content_parts.append(f"Keywords: {', '.join(keywords)}")
        if authors:
            content_parts.append(f"Authors: {', '.join(authors)}")
        if similar_papers:
            content_parts.append(f"Similar Papers: {'; '.join(similar_papers)}")
        
        return "\n".join(content_parts)
    
    def build_index(self, graph: nx.MultiDiGraph) -> VectorStoreIndex:
        """Build LlamaIndex from graph."""
        documents = self.convert_graph_to_documents(graph)
        print(f"Building index from {len(documents)} documents...")
        
        self.index = VectorStoreIndex.from_documents(documents)
        print("Index built successfully!")
        return self.index
    
    def get_query_engine(self):
        """Get configured query engine."""
        if not self.index:
            raise ValueError("Index not built. Call build_index() first.")
        
        return self.index.as_query_engine(
            similarity_top_k=3,
            response_mode="compact", #"tree_summarize"
        )
    
    def query(self, question: str):
        """Query the index with a question."""
        query_engine = self.get_query_engine()
        response = query_engine.query(question)
        return response

# # Usage example
# def setup_ollama_integration(graph: nx.MultiDiGraph):
#     """Setup and use the Ollama integration."""
    
#     # Initialize with your local Mistral model
#     integrator = LlamaIndexIntegrator(
#         llm_model="mistral:7b-instruct-v0.2-q4_0",  # Your GGUF model in Ollama
#         embedding_model="nomic-embed-text"  # Good lightweight embedding model
#     )
    
#     # Build the index
#     index = integrator.build_index(graph)
    
#     # Example queries
#     questions = [
#         "What are the main research methodologies used in these papers?",
#         "Which papers focus on machine learning applications?",
#         "What are the key findings about neural networks?",
#         "Show me papers published after 2020 with their main contributions"
#     ]
    
#     for question in questions:
#         print(f"\nQuestion: {question}")
#         response = integrator.query(question)
#         print(f"Answer: {response}")
#         print("-" * 50)
    
#     return integrator

'''



'''
class LlamaIndexIntegrator:
    """Integrates NetworkX graph with LlamaIndex for LLM-powered querying."""
    
    def __init__(self, llm_model: str = "gpt-3.5-turbo", 
                 embedding_model: str = "text-embedding-ada-002"):
        
        # Configure LlamaIndex settings
        Settings.llm = OpenAI(model=llm_model)
        Settings.embed_model = OpenAIEmbedding(model=embedding_model)
        
        self.node_parser = SimpleNodeParser.from_defaults()
        self.index = None
        
    def convert_graph_to_documents(self, graph: nx.MultiDiGraph) -> List[Document]:
        """Convert NetworkX graph to LlamaIndex documents."""
        documents = []
        
        # Convert nodes to documents
        for node_id, node_data in graph.nodes(data=True):
            if node_data['type'] == 'paper':
                content = self._create_paper_content(node_id, node_data, graph)
                doc = Document(
                    text=content,
                    metadata={
                        'node_id': node_id,
                        'type': node_data['type'],
                        'title': node_data.get('title', ''),
                        'year': node_data.get('year', 0)
                    }
                )
                documents.append(doc)
        
        return documents
    
    def _create_paper_content(self, paper_id: str, paper_data: Dict, 
                            graph: nx.MultiDiGraph) -> str:
        """Create rich text content for a paper including its relationships."""
        
        content_parts = [
            f"Title: {paper_data.get('title', 'Unknown')}",
            f"Year: {paper_data.get('year', 'Unknown')}",
            f"Journal: {paper_data.get('journal', 'Unknown')}"
        ]
        
        # Add methodology and findings if available
        if 'methodology' in paper_data['metadata']:
            content_parts.append(f"Methodology: {paper_data['metadata']['methodology']}")
        
        if 'main_findings' in paper_data['metadata']:
            content_parts.append(f"Main Findings: {paper_data['metadata']['main_findings']}")
        
        # Add keywords
        keywords = []
        authors = []
        similar_papers = []
        
        for neighbor_id in graph.neighbors(paper_id):
            neighbor_data = graph.nodes[neighbor_id]
            edge_data = graph.get_edge_data(paper_id, neighbor_id)
            
            if neighbor_data['type'] == 'keyword':
                keywords.append(neighbor_data['display_name'])
            elif neighbor_data['type'] == 'author':
                authors.append(neighbor_data['name'])
        
        # Find similar papers
        for edge in graph.edges(paper_id, data=True):
            if edge[2]['type'] == 'SIMILAR_TO':
                similar_paper_data = graph.nodes[edge[1]]
                similar_papers.append(f"{similar_paper_data['title']} (similarity: {edge[2]['weight']:.2f})")
        
        if keywords:
            content_parts.append(f"Keywords: {', '.join(keywords)}")
        if authors:
            content_parts.append(f"Authors: {', '.join(authors)}")
        if similar_papers:
            content_parts.append(f"Similar Papers: {'; '.join(similar_papers)}")
        
        return "\n".join(content_parts)
    
    def build_index(self, graph: nx.MultiDiGraph) -> VectorStoreIndex:
        """Build LlamaIndex from graph."""
        documents = self.convert_graph_to_documents(graph)
        self.index = VectorStoreIndex.from_documents(documents)
        return self.index
    
    def get_query_engine(self):
        """Get configured query engine."""
        if not self.index:
            raise ValueError("Index not built. Call build_index() first.")
        
        return self.index.as_query_engine(
            similarity_top_k=5,
            response_mode="tree_summarize"
        )
'''