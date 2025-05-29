# Research Paper Knowledge Graph

A comprehensive system for building knowledge graphs from research paper datasets with integrated LLM-powered querying and summarization capabilities.

## Features

- 📊 **Excel/CSV Data Loading**: Supports various research paper dataset formats
- 🕸️ **Graph Construction**: Builds rich knowledge graphs with papers, authors, keywords, and relationships
- 🤖 **LLM Integration**: Powered by LlamaIndex for intelligent querying and summarization
- 🔍 **Advanced Querying**: Keyword-based summarization, methodology comparison, trend analysis
- ⚡ **Scalable Architecture**: Object-oriented design for easy extension and maintenance
- 📈 **Relationship Analysis**: Automatic similarity detection between papers
- 🎯 **Interactive Interface**: Command-line interface for real-time exploration

## Installation

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/research-paper-graph.git
cd research-paper-graph

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .

# Set up environment variables
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### Development Setup

```bash
# Install with development dependencies
pip install -e ".[dev,viz,advanced]"

# Run tests
pytest tests/

# Format code
black src/
flake8 src/
```

## Usage

### Basic Usage

```python
from src.main import ResearchGraphApplication

# Initialize application
app = ResearchGraphApplication("config/config.yaml")

# Load data and build graph
graph, query_engine = app.run("data/raw/Literature_Review_1.xlsx")
```

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

## Data Format

Your Excel/CSV file should contain these columns:

### Required Columns
- `Title` - Paper title
- `Author(s)` - Comma-separated author names
- `Keywords` - Comma-separated keywords
- `Year` - Publication year

### Optional Columns  
- `Journal/Conference` - Publication venue
- `Methodology` - Research methodology description
- `Main Findings` - Key findings summary
- `Citations (est.)` - Estimated citation count
- `DOI/Link` - Paper DOI or URL
- `Institution (if known)` - Author affiliations

## Architecture

### Core Components

1. **Data Layer** (`src/data/`)
   - `DataLoader`: Handles Excel/CSV file loading
   - `DataPreprocessor`: Cleans and standardizes data
   - `DataValidator`: Validates data integrity

2. **Graph Layer** (`src/graph/`)
   - `GraphBuilder`: Constructs NetworkX graphs
   - `Node Classes`: Structured representations (Paper, Author, Keyword)
   - `Relationship Classes`: Typed relationships between entities
   - `GraphAnalyzer`: Graph analysis utilities

3. **LLM Layer** (`src/llm/`)
   - `LlamaIndexIntegrator`: Converts graphs to LlamaIndex format
   - `QueryEngine`: High-level query interface
   - `Summarizer`: Specialized summarization utilities

4. **Utils** (`src/utils/`)
   - `ConfigManager`: Configuration management
   - `Logger`: Logging utilities
   - `Helpers`: Common utility functions

### Graph Schema

```
Papers ──[HAS_KEYWORD]──→ Keywords
   │
   └──[AUTHORED_BY]──→ Authors
   │
   └──[SIMILAR_TO]──→ Papers
   │
   └──[PUBLISHED_IN]──→ Journals
```

### Scalability Features

- **Modular Design**: Easy to extend with new node/relationship types
- **Configurable Parameters**: Adjustable similarity thresholds and processing options
- **Caching Support**: Saves processed graphs for faster reloading
- **Batch Processing**: Handles large datasets efficiently
- **Plugin Architecture**: Extensible for custom analysis modules

## Configuration

### Main Configuration (`config/config.yaml`)

```yaml
llm:
  model: "gpt-3.5-turbo"
  temperature: 0.1

graph:
  similarity_threshold: 0.3
  max_keywords_per_paper: 10

logging:
  level: "INFO"
```

### LLM Configuration (`config/llm_config.yaml`)

Configure OpenAI settings, query parameters, and custom prompt templates.

## Advanced Features

### Custom Node Types

```python
from src.graph.nodes import BaseNode, NodeType

class InstitutionNode(BaseNode):
    def __init__(self, institution_name: str):
        super().__init__(
            id=f"inst_{institution_name.lower().replace(' ', '_')}",
            type=NodeType.INSTITUTION,
            properties={'name': institution_name}
        )
```

### Custom Relationships

```python
from src.graph.relationships import Relationship, RelationshipType

class CollaborationRelationship(Relationship):
    def __init__(self, author1_id: str, author2_id: str, paper_count: int):
        super().__init__(
            source_id=author1_id,
            target_id=author2_id,
            relationship_type=RelationshipType.COLLABORATES_WITH,
            properties={'paper_count': paper_count},
            weight=paper_count
        )
```

### Custom Query Templates

Modify `config/llm_config.yaml` to add custom prompt templates for specialized queries.

## Performance Optimization

- **Lazy Loading**: Graphs built on-demand
- **Vectorized Operations**: Pandas for efficient data processing  
- **Graph Caching**: Serialized graphs for quick reloading
- **Parallel Processing**: Multi-threaded relationship computation
- **Memory Management**: Efficient node/edge storage

## Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test module
pytest tests/test_graph/test_builder.py
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Roadmap

- [ ] Web interface with interactive graph visualization
- [ ] Support for additional data formats (JSON, XML)
- [ ] Advanced NLP features (entity extraction, sentiment analysis)
- [ ] Integration with academic databases (PubMed, arXiv)
- [ ] Collaborative filtering recommendations
- [ ] Export capabilities (GraphML, GEXF)
- [ ] REST API for programmatic access

## Support

For support, email support@example.com or create an issue on GitHub.

---
# .env.example
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_ORGANIZATION=your_org_id_here  # Optional

# Application Settings
LOG_LEVEL=INFO
CACHE_ENABLED=true
MAX_WORKERS=4

# Database (if using external storage)
# DATABASE_URL=postgresql://user:password@localhost:5432/research_graph