from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.schema import BaseNode as LlamaNode
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
import networkx as nx
from typing import List, Dict, Any

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