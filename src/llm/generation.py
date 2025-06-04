import requests

class OllamaGenerator:
    def __init__(self, model_name: str = "llama3.1", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
    
    def generate(self, prompt: str, max_tokens: int = 2000) -> str:
        """Generate text using Ollama"""
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": 0.7
                    }
                }
            )
            if response.status_code == 200:
                return response.json()["response"]
            else:
                print(f"Error generating text: {response.status_code}")
                return ""
        except Exception as e:
            print(f"Error connecting to Ollama: {e}")
            return ""