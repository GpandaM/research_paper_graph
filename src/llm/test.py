import requests

def test_simple_summary():
    """Test with minimal prompt"""
    prompt = "Summarize: Paper about CFD methods in 2024. Summary:"
    
    request_data = {
        "model": "llama3.2:latest", 
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": 50, "temperature": 0.3}
    }
    
    response = requests.post(
        "http://localhost:11434/api/generate",
        json=request_data,
        timeout=15
    )
    
    print(response.json())


test_simple_summary()