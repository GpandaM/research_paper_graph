

### Command Line Interface

```bash
python -m src.main --file data/raw/Literature_Review_1.xlsx
```

## neo4j setup

1. Install neo4j desktop 
2. Create a project <Project_abc>
3. Create password
4. Create a database <rpaperdb> inside project <Project_abc>
5. Install GDS and APOC plugins in database


## ollama
ollama pull mistral:7b-instruct-v0.2-q4_0
ollama pull nomic-embed-text:latest

## changes
if you use another embedding model; make sure to update embeddings length in neo_schema.py in create_vector_indexes method