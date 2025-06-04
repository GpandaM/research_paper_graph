

### Command Line Interface

```bash
# Run with interactive mode
python -m src.main --file data/raw/Literature_Review_1.xlsx

# Non-interactive mode
python -m src.main --file data/raw/Literature_Review_1.xlsx --no-interactive
```

### Interactive Commands

Once running, you can use these commands:

- `summarize <keyword>` - Get research summary for a specific keyword
- `compare <method1,method2>` - Compare two research methodologies  
- `trends` - Analyze research trends in the dataset
- `author <name>` - Get author expertise summary
- `quit` - Exit the application

### Example Queries

```python
# Summarize CFD research
response = query_engine.summarize_by_keyword("CFD")

# Compare methodologies
response = query_engine.compare_methodologies("CFD-DEM", "Eulerian-Lagrangian")

# Find research trends
response = query_engine.find_research_trends(year_range=(2020, 2024))

# Author expertise
response = query_engine.get_author_expertise("Smith, J.")
```



Phase 1: Basic Graph Construction

Create paper nodes with all attributes
Build direct relationships (authors, keywords, venues)
Establish temporal connections
Calculate basic similarity metrics

Phase 2: Semantic Enhancement

Generate embeddings for findings/abstracts
Cluster papers into research themes
Identify methodology patterns
Create complementary relationships

Phase 3: Knowledge Extraction

Identify research gaps (frequent limitations)
Track methodology evolution
Detect emerging trends
Generate contribution summaries