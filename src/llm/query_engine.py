from typing import Dict, List, Optional
import networkx as nx
from .integrator import LlamaIndexIntegrator

class GraphQueryEngine:
    """High-level query engine for the research paper knowledge graph."""
    
    def __init__(self, graph: nx.MultiDiGraph, llm_integrator: LlamaIndexIntegrator):
        self.graph = graph
        self.llm_integrator = llm_integrator
        self.query_engine = llm_integrator.get_query_engine()
    
    def summarize_by_keyword(self, keyword: str) -> str:
        """Summarize research related to a specific keyword."""
        query = f"Summarize all research papers related to '{keyword}'. Include key findings, methodologies, and trends."
        response = self.query_engine.query(query)
        return str(response)
    
    def compare_methodologies(self, method1: str, method2: str) -> str:
        """Compare two research methodologies."""
        query = f"Compare research papers using '{method1}' methodology versus '{method2}' methodology. What are the key differences, advantages, and applications?"
        response = self.query_engine.query(query)
        return str(response)
    
    def find_research_trends(self, year_range: tuple = None) -> str:
        """Identify research trends in the dataset."""
        if year_range:
            query = f"What are the main research trends between {year_range[0]} and {year_range[1]}? Identify emerging topics and methodologies."
        else:
            query = "What are the main research trends in this dataset? Identify key topics, methodologies, and their evolution."
        
        response = self.query_engine.query(query)
        return str(response)
    
    def get_author_expertise(self, author_name: str) -> str:
        """Get summary of an author's research expertise."""
        query = f"Summarize the research expertise and contributions of author '{author_name}'. What are their main research areas and key findings?"
        response = self.query_engine.query(query)
        return str(response)