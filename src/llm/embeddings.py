import requests
import time

class OllamaEmbedding:
    def __init__(self, model_name: str = "nomic-embed-text", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text"""
        try:
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json={
                    "model": self.model_name,
                    "prompt": text
                }
            )
            if response.status_code == 200:
                return response.json()["embedding"]
            else:
                print(f"Error getting embedding: {response.status_code}")
                return []
        except Exception as e:
            print(f"Error connecting to Ollama: {e}")
            return []
    
    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts"""
        embeddings = []
        for text in texts:
            embedding = self.get_embedding(text)
            embeddings.append(embedding)
            time.sleep(0.1)  # Rate limiting
        return embeddings