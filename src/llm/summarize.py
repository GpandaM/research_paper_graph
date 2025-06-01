import asyncio
import weaviate
from weaviate.connect import ConnectionParams
from neo4j import GraphDatabase
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import Settings
from typing import List, Dict, Any
import json
from ollama import Client
import requests
from time import sleep

class ResearchSummarizer:
    def __init__(self, neo4j_config: dict):
        # Neo4j setup
        self.neo4j_driver = GraphDatabase.driver(
            neo4j_config["uri"],
            auth=(neo4j_config["user"], neo4j_config["password"]),
            database=neo4j_config.get("database", "neo4j")
        )
        
        # LLM setup
        # Settings.llm = Ollama(
        #     model="mistral:7b-instruct-v0.2-q4_0",
        #     request_timeout=120.0  # 2 minutes timeout
        #     )
        # Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

        self.ollama_client = Client(
                    host='http://localhost:11434',
                    # headers={'x-some-header': 'some-value'}
                    )
        
    
    def get_graph_schema(self) -> Dict:
        """Inspect Neo4j database schema using basic queries"""
        schema_info = {}
        
        with self.neo4j_driver.session(database="paperdb") as session:
            try:
                # Get node counts and labels using basic queries
                node_count_result = session.run("MATCH (n) RETURN count(n) as node_count")
                schema_info['total_nodes'] = node_count_result.single()['node_count']
                
                # Get relationship counts
                rel_count_result = session.run("MATCH ()-[r]->() RETURN count(r) as rel_count")
                schema_info['total_relationships'] = rel_count_result.single()['rel_count']
                
                # Get distinct labels using basic query
                labels_result = session.run("MATCH (n) RETURN DISTINCT labels(n) as labels")
                all_labels = set()
                for record in labels_result:
                    if record['labels']:
                        all_labels.update(record['labels'])
                schema_info['node_labels'] = list(all_labels)
                
                # Get distinct relationship types
                rel_types_result = session.run("MATCH ()-[r]->() RETURN DISTINCT type(r) as rel_type")
                schema_info['relationship_types'] = [record['rel_type'] for record in rel_types_result]
                
                # Get properties for each label
                schema_info['node_properties'] = {}
                for label in schema_info['node_labels']:
                    prop_query = f"MATCH (n:`{label}`) RETURN keys(n) as properties LIMIT 1"
                    prop_result = session.run(prop_query)
                    record = prop_result.single()
                    if record and record['properties']:
                        schema_info['node_properties'][label] = record['properties']
                
                # Get sample data for each label
                schema_info['sample_data'] = {}
                for label in schema_info['node_labels']:
                    sample_query = f"MATCH (n:`{label}`) RETURN n LIMIT 2"
                    sample_result = session.run(sample_query)
                    samples = []
                    for record in sample_result:
                        node_data = dict(record['n'])
                        samples.append(node_data)
                    schema_info['sample_data'][label] = samples
                    
            except Exception as e:
                schema_info['error'] = str(e)
                print(f"Schema inspection error: {e}")
                
        return schema_info
    
    
    def print_schema_info(self):
        """Print detailed schema information"""
        schema = self.get_graph_schema()
        
        print("=== NEO4J DATABASE SCHEMA ===")
        print(f"Total Nodes: {schema.get('total_nodes', 'Unknown')}")
        print(f"Total Relationships: {schema.get('total_relationships', 'Unknown')}")
        print()
        
        print("Node Labels:")
        for label in schema.get('node_labels', []):
            print(f"  - {label}")
        print()
        
        print("Relationship Types:")
        for rel_type in schema.get('relationship_types', []):
            print(f"  - {rel_type}")
        print()
        
        print("Node Properties by Label:")
        for label, properties in schema.get('node_properties', {}).items():
            print(f"  {label}: {properties}")
        print()
        
        print("Sample Data:")
        for label, samples in schema.get('sample_data', {}).items():
            print(f"\n  {label} (showing up to 3 samples):")
            for i, sample in enumerate(samples, 1):
                print(f"    Sample {i}: {sample}")
        
        return schema
    
    def get_all_papers_from_neo4j(self) -> List[Dict]:
        """Retrieve all papers from Neo4j - dynamically based on actual schema"""
        
        # First, get the schema to understand the structure
        schema = self.get_graph_schema()
        
        # Find the most likely paper/document label
        paper_labels = []
        for label in schema.get('node_labels', []):
            label_lower = label.lower()
            if any(keyword in label_lower for keyword in ['paper', 'document', 'article', 'research', 'publication']):
                paper_labels.append(label)
        
        if not paper_labels:
            # If no obvious paper labels, use the first available label
            paper_labels = schema.get('node_labels', [])[:1]
        
        if not paper_labels:
            print("No node labels found in the database!")
            return []
        
        # Use the first paper-like label
        target_label = paper_labels[0]
        print(f"Using label: {target_label}")
        
        # Get available properties for this label
        available_props = schema.get('node_properties', {}).get(target_label, [])
        print(f"Available properties: {available_props}")
        
        # Build dynamic query based on available properties
        property_mappings = {
            'title': ['title', 'name', 'paper_title', 'document_title'],
            'authors': ['authors', 'author', 'author_names', 'creators'],
            'year': ['year', 'publication_year', 'pub_year', 'date'],
            'doi': ['doi', 'DOI', 'identifier'],
            'abstract': ['abstract', 'summary', 'description'],
            'keywords': ['keywords', 'tags', 'subjects', 'topics'],
            'citation_count': ['citation_count', 'citations', 'cite_count']
        }
        
        # Find matching properties
        selected_props = {}
        for target_prop, possible_names in property_mappings.items():
            for prop_name in possible_names:
                if prop_name in available_props:
                    selected_props[target_prop] = prop_name
                    break
        
        # Build RETURN clause
        return_clauses = []
        for target_prop, actual_prop in selected_props.items():
            return_clauses.append(f"n.{actual_prop} as {target_prop}")
        
        # Add any remaining properties not mapped
        for prop in available_props:
            if prop not in selected_props.values():
                return_clauses.append(f"n.{prop} as {prop}")
        
        if not return_clauses:
            return_clauses = ["n"]  # Return entire node if no specific properties
        
        query = f"""
        MATCH (n:`{target_label}`)
        RETURN {', '.join(return_clauses)}
        LIMIT 100
        """
        
        print(f"Executing query: {query}")
        
        with self.neo4j_driver.session(database="paperdb") as session:
            result = session.run(query)
            papers = []
            for record in result:
                paper_data = {}
                for key in record.keys():
                    paper_data[key] = record[key]
                papers.append(paper_data)
            
            print(f"Retrieved {len(papers)} records")
            if papers:
                print(f"Sample record keys: {list(papers[0].keys())}")
                
            return papers
    
    def chunk_papers(self, papers: List[Dict], chunk_size: int = 5) -> List[List[Dict]]:
        """Split papers into chunks for processing"""
        return [papers[i:i + chunk_size] for i in range(0, len(papers), chunk_size)]
    
    
    def summarize_paper_chunk(self, papers: List[Dict], theme: str = "") -> str:
        """Summarize a chunk of papers"""
        paper_texts = []
        for paper in papers:
            paper_text = f"Title: {paper.get('title', 'N/A')}\n"
            paper_text += f"Main_findings: {paper.get('main_findings', 'N/A')[:500]}...\n"
            paper_text += f"equations_models: {paper.get('equations_models', 'N/A')}\n"
            paper_text += f"application_area: {paper.get('application_area', 'N/A')}\n"
            paper_text += f"strengths: {paper.get('strengths', 'N/A')}\n"
            paper_text += f"limitations: {paper.get('limitations', 'N/A')}\n"
            paper_text += f"Keywords: {paper.get('keywords', 'N/A')}\n"
            paper_texts.append(paper_text)
        
        combined_text = "---\n".join(paper_texts)
        
        prompt = f"""
        Summarize the following research papers{' focusing on ' + theme if theme else ''}:
        
        {combined_text}
        
        Provide a concise summary in 3-5 senetences highlighting:
        1. Key research themes and methodologies
        2. Main findings
        """
        
        print(f"      summarize_paper_chunk sending {len(prompt.split())} words in prompt")
        print(prompt)
        # response = Settings.llm.complete(prompt)

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
        
        # Try with retries
        max_retries = 3
        retry_delay = 5  # seconds
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    "http://localhost:11434/api/generate",
                    headers={"Content-Type": "application/json"},
                    data=json.dumps(request_data),
                    timeout=180  # 60 seconds timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if result.get("done", False):
                        print(f"Summary generated successfully {result}")
                        return result.get("response") #, "No summary generated")
                    else:
                        print("Incomplete response from Ollama")
                else:
                    print(f"Ollama API error: {response.status_code} - {response.text}")
                    
            except requests.exceptions.RequestException as e:
                print(f"Attempt {attempt + 1} failed: {str(e)}")
            
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
        
        return "Failed to generate summary after multiple attempts"
        
    
    def format_references(self, papers: List[Dict]) -> str:
        """Format papers as references"""
        references = []
        for i, paper in enumerate(papers, 1):
            authors = paper.get('authors', 'Unknown')
            year = paper.get('year', 'N/A')
            title = paper.get('title', 'Untitled')
            doi = paper.get('doi', '')
            
            ref = f"[{i}] {authors} ({year}). {title}."
            if doi:
                ref += f" DOI: {doi}"
            references.append(ref)
        
        return "\n".join(references)
    
    
    def get_research_themes_from_neo4j(self, papers: List[Dict]) -> List[str]:
        """Extract themes directly from Neo4j paper keywords"""
        all_keywords = []
        for paper in papers:
            keywords = paper.get('keywords', [])
            if isinstance(keywords, list):
                all_keywords.extend(keywords[:3])  # Top 3 per paper
            elif isinstance(keywords, str):
                all_keywords.extend([k.strip() for k in keywords.split(',')[:3]])
        
        # Count frequency and get top themes
        from collections import Counter
        keyword_counts = Counter(all_keywords)
        return [kw for kw, count in keyword_counts.most_common(8)]
    
    
    
    
    def generate_comprehensive_summary(self) -> str:
        """Generate comprehensive summary of all research papers"""
        # First inspect the database schema
        # print("Inspecting Neo4j database schema...")
        schema = self.print_schema_info()
        
        print("\nRetrieving all papers from Neo4j...")
        all_papers = self.get_all_papers_from_neo4j()
        
        if not all_papers:
            return "No papers found in the database."
        
        print(f"Found {len(all_papers)} papers")
        
        # Get main research themes
        print("Identifying research themes...")
        themes = self.get_research_themes_from_neo4j(all_papers)
        print(f" themes are {themes}")
        
        # Split papers into chunks
        paper_chunks = self.chunk_papers(all_papers, chunk_size=2)
        
        paper_chunks = paper_chunks[:3] ## for testing keeping only 3 sets now

        # Generate summaries for each chunk
        print("Generating summaries...")
        chunk_summaries = []
        for i, chunk in enumerate(paper_chunks):
            print(f"Processing chunk {i+1}/{len(paper_chunks)}")
            theme = themes[i % len(themes)] if themes else ""
            print(f"       theme is {theme}")
            summary = self.summarize_paper_chunk(chunk, theme)
            print(f"       summary is {theme}")
            chunk_summaries.append(summary)
        
        # Combine all summaries
        print("Combining summaries...")
        combined_summary_text = "\n\n".join(chunk_summaries)
        
        # Generate final comprehensive summary
        final_prompt = f"""
        Based on the following research summaries covering {len(all_papers)} papers, create a comprehensive literature review:
        
        Research Themes Identified: {', '.join(themes)}
        
        Summaries:
        {combined_summary_text}
        
        Create a well-structured literature review with:
        1. Introduction - Overview of the research domain
        2. Main Research Areas - Organize by themes with key findings
        3. Methodological Approaches - Common methods and innovations
        4. Key Contributions - Major advances and discoveries
        5. Future Directions - Gaps and opportunities
        
        Make it comprehensive but concise.
        """
        
        print(f"prompt has {len(final_prompt.split())} words")

        request_data = {
            "model": "mistral:7b-instruct-v0.2-q4_0",
            "prompt": final_prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "num_predict": 500,
                "repeat_penalty": 1.1
                    }
                }
        
        # final_summary = Settings.llm.complete(final_prompt)
        final_summary_res = requests.post(
                    "http://localhost:11434/api/generate",
                    headers={"Content-Type": "application/json"},
                    data=json.dumps(request_data),
                    timeout=180  # 60 seconds timeout
                )
                
        if final_summary_res.status_code == 200:
            result = final_summary_res.json()
            if result.get("done", False):
                print(f"Summary generated successfully {result}")
                final_summary = result.get("response")
                # return result.get("response") #, "No summary generated")
            else:
                print("Incomplete response from Ollama")
        else:
            print(f"Ollama API error: =======================")
        
        # Format references
        print("Formatting references...")
        references = self.format_references(all_papers)
        
        # Combine everything
        complete_summary = f"""
# Comprehensive Literature Review

{str(final_summary)}

## References

{references}

---
Total Papers Analyzed: {len(all_papers)}
Research Themes: {', '.join(themes)}
        """
        
        return complete_summary
    
    def close(self):
        """Close connections"""
        self.neo4j_driver.close()






# Usage example
def main():
    # Configuration
    neo4j_config = {
            "uri": "neo4j://localhost:7687",
            "user": "neo4j",
            "password": "Sundeep@123",
            "database": "paperdb"  # Add this line
            }
    
    # Create summarizer
    summarizer = ResearchSummarizer(neo4j_config)
    
    try:
        # Generate comprehensive summary
        summary = summarizer.generate_comprehensive_summary()
        print("\n\n\n\n\n")
        print(summary)
        
        ## Save to file
        with open("research_summary.md", "w", encoding="utf-8") as f:
            f.write(summary)
        
        print("\nSummary saved to research_summary.md")
        
    finally:
        summarizer.close()

if __name__ == "__main__":
    main()