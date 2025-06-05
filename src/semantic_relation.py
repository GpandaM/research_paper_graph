import requests
import json
import numpy as np
from typing import List
from sklearn.metrics.pairwise import cosine_similarity

from .logger import setup_logger
from .neo_store import Neo4jGraphStore

class SemanticRelationshipGenerator:
    def __init__(self, graph_store: Neo4jGraphStore, ollama_url: str = "http://localhost:11434"):
        self.graph_store = graph_store
        self.ollama_url = ollama_url
        self.logger = setup_logger()
        self.embeddings_cache = {}
        
    def generate_embeddings(self, batch_size: int = 32):
        """Generate embeddings for all papers using Ollama"""
        # Get all papers from database
        query = """
        MATCH (p:PAPER) 
        RETURN p.id AS id, p.title AS title, 
               p.methodology AS methodology, p.main_findings AS findings
        """
        papers = self.graph_store.execute(query)

        # for paper in papers:
        #     text = self._format_embedding_text(paper)
        #     embedding = self._get_ollama_embedding(text)
        #     if embedding:
        #         self.embeddings_cache[paper['id']] = embedding

        # Process in batches for efficiency
        for i in range(0, len(papers), batch_size):
            batch = papers[i:i+batch_size]
            texts = [self._format_embedding_text(paper) for paper in batch]
            embeddings = self._get_batch_ollama_embedding(texts)
            
            for j, paper in enumerate(batch):
                if embeddings and j < len(embeddings):
                    self.embeddings_cache[paper['id']] = embeddings[j]
                    # Store in Neo4j as well
                    self._store_embedding(paper['id'], embeddings[j])
    
    
    def _format_embedding_text(self, paper: dict) -> str:
        """Create text for embedding generation"""
        return f"""
        Title: {paper['title']}
        Methodology: {paper['methodology']}
        Main Findings: {paper['findings']}
        """


    def _store_embedding(self, paper_id: str, embedding: List[float]):
        """Store embedding in Neo4j"""
        query = """
        MATCH (p:PAPER {id: $id})
        SET p.embedding = $embedding
        """
        self.graph_store.execute(query, {'id': paper_id, 'embedding': embedding})


                
    def create_semantic_relationships(self, similarity_threshold: float = 0.80):
        """Create semantic relationships between papers"""
        '''
        ## if we won't have cache
        # Retrieve embeddings
                papers = self.graph_store.execute("""
                MATCH (p:PAPER) 
                WHERE p.embedding IS NOT NULL 
                RETURN p.id AS id, p.embedding AS embedding
                """)
        '''
        paper_ids = list(self.embeddings_cache.keys())
        
        for i, paper_id1 in enumerate(paper_ids):
            for paper_id2 in paper_ids[i+1:]:
                similarity = self._calculate_similarity(paper_id1, paper_id2)
                if similarity > similarity_threshold:
                    self._create_relationship(paper_id1, paper_id2, 'SEMANTIC_SIMILAR', 
                                           {'similarity_score': similarity})
                    
    
    def create_limitation_relationships(self):
        """Create ADDRESSES_LIMITATION relationships using LLM"""
        # Get all papers with their content
        query = """
        MATCH (p:PAPER) 
        RETURN p.id as id, p.title as title, p.keywords as keywords, 
               p.methodology as methodology, p.main_findings AS findings
        ORDER BY p.year
        """
        papers = self.graph_store.execute(query)
        
        for i, base_paper in enumerate(papers):
            for candidate_paper in papers[i+1:]:
                if self._addresses_limitation(base_paper, candidate_paper):
                    self._create_relationship(base_paper['id'], candidate_paper['id'], 
                                           'ADDRESSES_LIMITATION')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    def _get_ollama_embedding(self, text: str):
        """Get embedding from Ollama"""
        try:
            response = requests.post(
                                   f"{self.ollama_url}/api/embed", 
                                   json = {
                                       "model":"nomic-embed-text:latest", 
                                       "input": text
                                   }
                                )
            # print(response.content)
            if response.status_code == 200:
                return response.json()["embeddings"]
        except Exception as e:
            self.logger.warning(f"======================Failed to get embedding: {e}")
        return None


    def _get_batch_ollama_embedding(self, texts: List[str]):
        """Get batch embeddings from Ollama"""
        try:
            response = requests.post(
                                   f"{self.ollama_url}/api/embed", 
                                   json = {
                                       "model":"nomic-embed-text:latest", 
                                       "input": texts
                                   }
                                )
            if response.status_code == 200:
                return response.json()["embeddings"]
        except Exception as e:
            self.logger.error(f"Failed to get batch embeddings: {e}")
        return None

        
    
    
    
    
    
    def _calculate_similarity(self, paper_id1: str, paper_id2: str) -> float:
        """Calculate cosine similarity between two papers"""
        emb1 = np.array(self.embeddings_cache[paper_id1]).reshape(1, -1)
        emb2 = np.array(self.embeddings_cache[paper_id2]).reshape(1, -1)
        return cosine_similarity(emb1, emb2)[0][0]
        
    def _addresses_limitation(self, base_paper: dict, candidate_paper: dict) -> bool:
        """Determine if candidate paper addresses base paper's limitations using LLM"""
        prompt = f"""
        Base paper: "{base_paper['title']}"
        Base Keywords: {base_paper['keywords']}
        Base Methodology: {base_paper['methodology']}
        Base Main Findings: {base_paper['findings']}
        
        Candidate paper: "{candidate_paper['title']}"
        Candidate Keywords: {candidate_paper['keywords']}
        Candidate Methodology: {candidate_paper['methodology']}
        Candidate Main Findings: {candidate_paper['findings']}
        
        Does the candidate paper address potential limitations or gaps from the base paper?
        Answer only: YES or NO
        """
        
        print(f"prompt has {len(prompt.split())} words.")

        request_data = {
            "model": "mistral:7b-instruct-v0.2-q4_0",
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "num_predict": 500,
                "repeat_penalty": 1.1
                    }
                }

        try:
            response = requests.post(
                                    f"{self.ollama_url}/api/generate", 
                                    headers={"Content-Type": "application/json"},
                                    data=json.dumps(request_data),
                                    timeout=180  # 60 seconds timeout
                )
            if response.status_code == 200:
                answer = response.json()["response"].strip().upper()
                print(answer)
                return "YES" in answer
        except Exception as e:
            self.logger.warning(f"=====================================LLM query failed: {e}")
        return False
        
    def _create_relationship(self, source_id: str, target_id: str, rel_type: str, properties: dict = None):
        """Create relationship between two papers"""
        rel_data = {
            'source_id': source_id,
            'target_id': target_id,
            'relationship_type': rel_type,
            'properties': properties or {}
        }
        return self.graph_store.insert_relationship(rel_data)
    
