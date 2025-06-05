import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import requests
import json
from typing import Dict, List

from .logger import setup_logger
from .neo_store import Neo4jGraphStore

class SemanticRelationshipGenerator:
    def __init__(self, graph_store: Neo4jGraphStore, ollama_url: str = "http://localhost:11434"):
        self.graph_store = graph_store
        self.ollama_url = ollama_url
        self.logger = setup_logger()
        self.embeddings_cache = {}
        

    def generate_embeddings(self, batch_size: int = 32):
        """Generate embeddings for all papers using Ollama with batching"""
        query = """
        MATCH (p:PAPER) 
        RETURN p.id AS id, p.title AS title, 
               p.methodology AS methodology, p.main_findings AS findings,
               p.application_area AS application_area, p.strengths AS strengths,
               p.limitations AS limitations
        """
        try:
            papers = self.graph_store.execute(query)

            total_batches = (len(papers) + batch_size - 1) // batch_size

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
                
                batch_num = (i // batch_size) + 1
                self.logger.info(f"Generated and stored embeddings for {batch_num}th batch out of {total_batches}")
        except Exception as e:
            self.logger.exception(f"There is error while generating the embeddings {e}")
            raise
    
    
    def _format_embedding_text(self, paper: dict) -> str:
        """Create text for embedding generation"""
        return f"""
        Title: {paper['title']}
        Methodology: {paper['methodology']}
        Main Findings: {paper['findings']}
        Application Area: {paper['application_area']}
        Strengths: {paper['strengths']}
        Limitations: {paper['limitations']}
        """
    
    
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
    
    def _store_embedding(self, paper_id: str, embedding: List[float]):
        """Store embedding in Neo4j"""
        query = """
        MATCH (p:PAPER {id: $id})
        SET p.embedding = $embedding
        """
        self.graph_store.execute(query, {'id': paper_id, 'embedding': embedding})
    

    def compute_importance_scores(self, current_year=2025, recent_years=7):
        query = """
        MATCH (p:PAPER)
        WHERE p.citations_count IS NOT NULL AND p.year IS NOT NULL
        WITH p, 
            p.citations_count AS citations,
            $current_year - p.year AS age
        SET p.importance = citations + ($recent_years - CASE WHEN age > $recent_years THEN $recent_years ELSE age END)
        RETURN count(p) AS updated
        """
        self.graph_store.execute(query, {
            "current_year": current_year,
            "recent_years": recent_years
        })





    
    def create_semantic_relationships(self, similarity_threshold: float = 0.80, top_k: int = 20):
        """Create semantic relationships between papers using efficient similarity search"""
        try:
            # Get all papers with embeddings
            paper_ids = list(self.embeddings_cache.keys())
            if not paper_ids:
                self.logger.error("No embeddings found in cache during create_semantic_relationships")
                raise ValueError("No embeddings found in cache during create_semantic_relationships")
                
            # Convert to numpy array for vectorized operations
            embeddings = np.array([self.embeddings_cache[pid] for pid in paper_ids])
            
            # Compute similarity matrix
            sim_matrix = cosine_similarity(embeddings)
            
            # Process each paper
            for i, pid1 in enumerate(paper_ids[:-1]):
                # Get top_k most similar papers (excluding self and previous pairs)
                similarities = sim_matrix[i, i+1:]
                
                if top_k > 0 and len(similarities) >= top_k:
                    candidate_indices = np.argpartition(similarities, -top_k)[-top_k:]
                    candidate_indices = candidate_indices[np.argsort(similarities[candidate_indices])]
                else:
                    candidate_indices = np.argsort(similarities)[-top_k:]

                
                for idx in candidate_indices:
                    j = i + 1 + idx  # Actual paper index
                    pid2 = paper_ids[j]
                    similarity = similarities[idx]
                    
                    if similarity > similarity_threshold:
                        self._create_relationship(
                            pid1, pid2, 'SEMANTIC_SIMILAR',
                            {'similarity_score': float(similarity)}
                        )
            
            self.logger.info(f"Semantic Relations are created")
        except Exception as e:
            self.logger.exception(f"There is error while creating semantic relations {e}")
            raise
    
    
    def create_limitation_relationships(self, similarity_threshold: float = 0.80,
                                            top_k: int = 10,
                                            max_links_per_paper: int = 2,
                                            importance_percent: float = 0.2
                                        ):
        """Optimized ADDRESSES_LIMITATION relationship creation using top-k similarity filtering"""
        ## importance_percent=0.4 → top 40% of papers
        ## top_k=10 → top 10 similar newer papers checked per paper
        ## max_links_per_paper=2 → at most 2 edges created per paper
        try:
            self.compute_importance_scores()
            # Step 1: Fetch top N% important papers
            cypher = f"""
                MATCH (p:PAPER)
                WHERE p.embedding IS NOT NULL AND p.importance IS NOT NULL
                WITH p ORDER BY p.importance DESC
                WITH collect(p)[..toInteger(count(p) * $importance_percent)] AS top_papers
                UNWIND top_papers AS p
                RETURN p.id AS id, 
                    p.embedding AS embedding, 
                    p.title AS title,
                    p.keywords AS keywords,
                    p.methodology AS methodology,
                    p.main_findings AS findings,
                    p.year AS year
            """
            papers = self.graph_store.execute(cypher, {"importance_percent": importance_percent})

            if not papers:
                self.logger.error("No papers with embeddings found in create_limitation_relationships")
                raise ValueError("No papers returned from importance-ranked subset")

            embeddings = np.array([paper['embedding'] for paper in papers])
            sim_matrix = cosine_similarity(embeddings)

            relation_count = 0

            for i, base_paper in enumerate(papers[:-1]):
                similarities = sim_matrix[i, i+1:]

                # Top-k similar newer papers
                if top_k > 0 and len(similarities) >= top_k:
                    candidate_indices = np.argpartition(similarities, -top_k)[-top_k:]
                    candidate_indices = candidate_indices[np.argsort(similarities[candidate_indices])]
                else:
                    candidate_indices = np.argsort(similarities)[-top_k:]

                link_created = 0
                for idx in candidate_indices:
                    if similarities[idx] < similarity_threshold:
                        continue

                    candidate_paper = papers[i + 1 + idx]
                    if self._addresses_limitation(base_paper, candidate_paper):
                        self._create_relationship(
                            base_paper['id'], candidate_paper['id'], 
                            'ADDRESSES_LIMITATION'
                        )
                        relation_count += 1
                        link_created += 1
                        self.logger.info(f"Creating {relation_count}th ADDRESSES_LIMITATION")

                        if link_created >= max_links_per_paper:
                            break

            self.logger.info("Finished creating limitation relationships.")

        except Exception as e:
            self.logger.exception(f"Error during create_limitation_relationships: {e}")
            raise

    
    
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
        
        # print(f"prompt has {len(prompt.split())} words.")

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
                # print("YES" in answer)
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
        