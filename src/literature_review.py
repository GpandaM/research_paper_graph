import logging
from typing import Dict, List, Any, Optional
import ollama
from neo4j import GraphDatabase
import pandas as pd
from collections import defaultdict, Counter
import json
import requests
from neo4j.time import Date, Time, DateTime, Duration

from .logger import setup_logger

class LiteratureReviewGenerator:
    """Generate comprehensive literature reviews using Neo4j graph data and Ollama LLM"""
    
    def __init__(self, graph_store, ollama_model: str = "mistral:7b-instruct-v0.2-q4_0", 
                 ollama_url: str = "http://localhost:11434"):
        """
        Initialize the Literature Review Generator
        
        Args:
            graph_store: Neo4j graph database connection
            ollama_model: Ollama model name to use
            ollama_host: Ollama server host
        """
        self.graph_store = graph_store
        self.ollama_model = ollama_model
        self.ollama_url = ollama_url
        self.logger = setup_logger()
        

    
    def _call_ollama(self, prompt: str, context_data: Dict = None) -> str:
        """Call Ollama model with prompt and context"""
        try:
            # Prepare the full prompt with context
            full_prompt = self._prepare_prompt_with_context(prompt, context_data)
            
            # print(f"prompt has {len(full_prompt.split())} words.")

            request_data = {
                "model": "mistral:7b-instruct-v0.2-q4_0",
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "num_predict": 800,
                    "repeat_penalty": 1.1
                        }
                    }
            
            response = requests.post(
                                    f"{self.ollama_url}/api/generate", 
                                    headers={"Content-Type": "application/json"},
                                    data=json.dumps(request_data),
                                    timeout=240  # 240 seconds timeout
                )
            
            if response.status_code == 200:
                result = response.json()
                if result.get("done", False):
                    # print(f"Summary generated successfully {result}")
                    answer = result.get("response")
            
            return answer
        
        except Exception as e:
            self.logger.error(f" {'!'*30} Ollama API call failed: {e}")
            return f"Error generating content: {str(e)}"
    
    
    
    
    def _prepare_prompt_with_context(self, prompt: str, context_data: Dict) -> str:
        """Prepare prompt with structured context data"""
        context_str = ""
        if context_data:
            context_str = f"\n\nContext Data:\n{json.dumps(context_data, indent=2, default=str)}\n\n"
        
        return f"{context_str}{prompt}"
    

    def format_cypher_result(records):
        """
        Converts Neo4j driver records into a JSON-safe list of dictionaries.
        Handles all complex types (nodes, relationships, paths, temporal types).
        """
        def serialize(obj):
            # Handle graph types
            if hasattr(obj, 'items'):  # Nodes, Relationships
                return dict(obj.items())
            
            # Handle temporal types
            elif isinstance(obj, (Date, Time, DateTime)):
                return obj.isoformat()
            elif isinstance(obj, Duration):
                return str(obj)
            
            # Handle paths by extracting start/end nodes
            elif hasattr(obj, 'nodes') and hasattr(obj, 'relationships'):
                return {
                    "start_node": dict(obj.nodes[0].items()),
                    "end_node": dict(obj.nodes[-1].items()),
                    "length": len(obj.relationships)
                }
            
            # Handle lists recursively
            elif isinstance(obj, list):
                return [serialize(item) for item in obj]
            
            # Return primitives as-is
            return obj
        
        return [
            {key: serialize(value) for key, value in record.items()}
            for record in records
        ]




    def _extract_paper_data(self) -> Dict:
        """Extract comprehensive paper data from graph"""
        query = """
        MATCH (p:PAPER)
        OPTIONAL MATCH (p)-[:AUTHORED_BY]->(a:AUTHOR)
        OPTIONAL MATCH (p)-[:PUBLISHED_IN]->(j:JOURNAL)
        OPTIONAL MATCH (p)-[:HAS_KEYWORD]->(k:KEYWORD)
        OPTIONAL MATCH (a)-[:AFFILIATED_WITH]->(i:INSTITUTION)
        RETURN p, 
               collect(DISTINCT a.name) as authors,
               collect(DISTINCT j.name) as journals,
               collect(DISTINCT k.name) as keywords,
               collect(DISTINCT i.name) as institutions
        ORDER BY p.year DESC, p.citations_count DESC
        """
        
        results = self.graph_store.execute(query)
        papers = []
        
        for record in results:
            paper = dict(record['p'])
            paper['authors'] = record['authors']
            paper['journals'] = record['journals']
            paper['keywords'] = record['keywords']
            paper['institutions'] = record['institutions']
            papers.append(paper)
        

        # self.logger.info(f"{'=='*50} {len(papers)} Papers are retrived")
        return {"papers": papers, "total_papers": len(papers)}
    
    
    
    def _extract_temporal_trends(self) -> Dict:
        """Extract temporal evolution patterns"""
        query = """
        MATCH (p1:PAPER)-[:TEMPORAL_SUCCESSOR]->(p2:PAPER)
        RETURN p1.year as earlier_year, p2.year as later_year,
               p1.title as earlier_title, p2.title as later_title,
               p1.main_findings as earlier_findings, p2.main_findings as later_findings
        ORDER BY p1.year, p2.year
        """
        
        temporal_data = self.graph_store.execute(query)
        
        # Year-wise paper distribution
        year_query = """
        MATCH (p:PAPER)
        WHERE p.year IS NOT NULL
        RETURN p.year as year, count(p) as paper_count
        ORDER BY year
        """
        
        year_distribution = self.graph_store.execute(year_query)
        
        return {
            "temporal_relationships": temporal_data,
            "year_distribution": year_distribution
        }


    # def _extract_temporal_trends(self) -> Dict:
    #     """Extract comprehensive temporal patterns from available data"""
        
    #     # 1. Year-wise paper distribution (ALWAYS include this)
    #     year_query = """
    #     MATCH (p:PAPER)
    #     WHERE p.year IS NOT NULL
    #     RETURN p.year as year, count(p) as paper_count,
    #            collect(p.title)[0..3] as sample_titles
    #     ORDER BY year
    #     """
        
    #     # 2. Research evolution by application areas over time
    #     area_evolution_query = """
    #     MATCH (p:PAPER)
    #     WHERE p.year IS NOT NULL AND p.application_area IS NOT NULL AND p.application_area <> ''
    #     WITH p.year as year, p.application_area as area, count(p) as paper_count
    #     RETURN year, area, paper_count
    #     ORDER BY year, paper_count DESC
    #     """
        
    #     # 3. Methodology adoption over time
    #     methodology_timeline_query = """
    #     MATCH (p:PAPER)
    #     WHERE p.year IS NOT NULL AND p.methodology IS NOT NULL AND p.methodology <> ''
    #     WITH p.year as year, p.methodology as methodology, count(p) as usage_count
    #     RETURN year, methodology, usage_count
    #     ORDER BY year, usage_count DESC
    #     """
        
    #     # 4. Citation trends over time
    #     citation_trends_query = """
    #     MATCH (p:PAPER)
    #     WHERE p.year IS NOT NULL AND p.citations_count IS NOT NULL
    #     RETURN p.year as year, 
    #            avg(toFloat(p.citations_count)) as avg_citations,
    #            max(p.citations_count) as max_citations,
    #            count(p) as paper_count
    #     ORDER BY year
    #     """
        
        
    #     area_evolution = self.graph_store.execute(area_evolution_query)
    #     self.logger.info(f"Extracted area_evolution")
    #     methodology_timeline = self.graph_store.execute(methodology_timeline_query)
    #     self.logger.info(f"Extracted methodology_timeline")
    #     year_distribution = self.graph_store.execute(year_query)
    #     self.logger.info(f"Extracted year_distribution")
    #     citation_trends = self.graph_store.execute(citation_trends_query)
    #     self.logger.info(f"Extracted citation_trends")
        
        
    #     return {
    #         "year_distribution": year_distribution,
    #         "area_evolution": area_evolution,
    #         "methodology_timeline": methodology_timeline,
    #         "citation_trends": citation_trends,
    #         "temporal_method": "comprehensive_analysis"
    #     }
    
    
    
    def _extract_research_communities(self) -> Dict:
        """Extract research communities using GDS algorithms"""
        remove_projection = """
        CALL gds.graph.exists('literature-network') YIELD exists
        WITH exists
        CALL apoc.do.when(
        exists,
        'CALL gds.graph.drop("literature-network", false)',
        'RETURN "Graph does not exist" AS message',
        {}
        ) YIELD value
        RETURN value;
        """

        projection_query = """
        CALL gds.graph.project(
            'literature-network',
            ['PAPER', 'AUTHOR', 'KEYWORD'],
            {
                AUTHORED_BY: {orientation: 'UNDIRECTED'},
                HAS_KEYWORD: {orientation: 'UNDIRECTED'},
                SEMANTIC_SIMILAR: {orientation: 'UNDIRECTED'}
            }
        )
        """

        # Executing separately, catching projection errors specifically
        try:
            self.graph_store.execute(remove_projection)
            self.graph_store.execute(projection_query)
        except Exception as e:
            self.logger.error(f"Issue with projection: {e}")

        
        # Run Louvain community detection
        community_query = """
        CALL gds.louvain.stream('literature-network')
        YIELD nodeId, communityId
        WITH gds.util.asNode(nodeId) as node, communityId
        RETURN labels(node)[0] as nodeType, 
               node.name as nodeName,
               CASE 
                   WHEN 'PAPER' IN labels(node) THEN node.title
                   WHEN 'AUTHOR' IN labels(node) THEN node.name
                   WHEN 'KEYWORD' IN labels(node) THEN node.name
               END as nodeIdentifier,
               communityId
        ORDER BY communityId, nodeType
        """
        
        communities = self.graph_store.execute(community_query)
        
        # Group by community
        community_groups = defaultdict(lambda: {"papers": [], "authors": [], "keywords": []})
        for record in communities:
            community_id = record['communityId']
            node_type = record['nodeType']
            
            if node_type == 'PAPER':
                community_groups[community_id]['papers'].append(record['nodeIdentifier'])
            elif node_type == 'AUTHOR':
                community_groups[community_id]['authors'].append(record['nodeIdentifier'])
            elif node_type == 'KEYWORD':
                community_groups[community_id]['keywords'].append(record['nodeIdentifier'])
        
        return {"communities": dict(community_groups)}
    
    
    
    def _extract_influential_works(self) -> Dict:
        """Extract most influential papers and authors"""
        # High-impact papers
        papers_query = """
        MATCH (p:PAPER)
        RETURN p.title, p.citations_count, p.year, p.main_findings, p.authors
        ORDER BY p.citations_count DESC
        LIMIT 20
        """
        
        # Central authors (by PageRank)
        authors_query = """
        MATCH (a:AUTHOR)-[:AUTHORED_BY]-(p:PAPER)
        WITH a, count(p) as paper_count, sum(p.citations_count) as total_citations
        RETURN a.name, paper_count, total_citations
        ORDER BY total_citations DESC, paper_count DESC
        LIMIT 15
        """
        
        influential_papers = self.graph_store.execute(papers_query)
        influential_authors = self.graph_store.execute(authors_query)
        
        return {
            "influential_papers": influential_papers,
            "influential_authors": influential_authors
        }
    
    
    
    def _extract_methodologies(self) -> Dict:
        """Extract methodology patterns and evolution"""
        methodology_query = """
        MATCH (p:PAPER)
        WHERE p.methodology IS NOT NULL AND p.methodology <> ''
        RETURN p.methodology, p.year, p.title, count(*) as frequency
        ORDER BY frequency DESC
        """
        
        methodologies = self.graph_store.execute(methodology_query)
        
        # Methodology evolution over time
        methodology_evolution_query = """
        MATCH (p:PAPER)
        WHERE p.methodology IS NOT NULL AND p.methodology <> '' AND p.year IS NOT NULL
        RETURN p.year, p.methodology, count(*) as usage_count
        ORDER BY p.year, usage_count DESC
        """
        
        methodology_evolution = self.graph_store.execute(methodology_evolution_query)
        
        return {
            "methodologies": methodologies,
            "methodology_evolution": methodology_evolution
        }
    
    
    
    
    def _extract_limitations_and_gaps(self) -> Dict:
        """Extract research limitations and gaps"""
        limitations_query = """
        MATCH (p:PAPER)
        WHERE p.limitations IS NOT NULL AND p.limitations <> ''
        RETURN p.title, p.limitations, p.year, p.application_area
        ORDER BY p.year DESC
        """
        
        # Papers that address limitations of others
        limitation_relationships_query = """
        MATCH (p1:PAPER)-[:ADDRESSES_LIMITATION]->(p2:PAPER)
        RETURN p1.title as addressing_paper, p1.year as addressing_year,
               p2.title as addressed_paper, p2.year as addressed_year,
               p1.main_findings as solution_findings,
               p2.limitations as original_limitations
        ORDER BY p1.year DESC
        """
        
        limitations = self.graph_store.execute(limitations_query)
        limitation_relationships = self.graph_store.execute(limitation_relationships_query)
        
        return {
            "limitations": limitations,
            "limitation_relationships": limitation_relationships
        }
    
    
    
    def _generate_introduction(self) -> str:
        
        try:
            """Generate literature review introduction"""
            paper_data = self._extract_paper_data()
            temporal_data = self._extract_temporal_trends()
            
            # Get year distribution based on the temporal method used
            year_distribution = []
            if "year_distribution" in temporal_data:
                year_distribution = temporal_data["year_distribution"]
            elif "area_evolution" in temporal_data:
                # Extract year info from area evolution data
                years = set()
                for item in temporal_data["area_evolution"]:
                    years.add(item["year"])
                year_distribution = [{"year": year, "paper_count": 0} for year in sorted(years)]
            
            # Safe year range calculation
            paper_years = [p['year'] for p in paper_data['papers'] if p.get('year') is not None]
            year_range = f"{min(paper_years)}-{max(paper_years)}" if paper_years else "N/A"
            
            context = {
                "total_papers": paper_data["total_papers"],
                "year_range": year_range,
                "recent_papers": paper_data["papers"][:5],
                "temporal_trends": temporal_data,
                "year_distribution": year_distribution
            }
            
            prompt = """
            Based on the provided context data about research papers, write a comprehensive introduction in an academic tone
            approximately in 100-150 words. The introduction should:
            
            1. Provide an overview of the research domain
            2. Highlight the temporal scope and evolution of research
            3. Mention key trends and developments
            4. Set the stage for detailed analysis
            
            """
            
            return (self._call_ollama(prompt, context), paper_data)

        except Exception as e:
            self.logger.exception(f"There is an error while generating the Introduction {e}")
            raise
    
    
    
    def _generate_research_areas(self) -> str:
        
        try:
            """Generate research areas analysis"""
            communities = self._extract_research_communities()
            self.logger.info(f"communities created")
            paper_data = self._extract_paper_data()
            
            # Extract application areas and keywords
            application_areas = Counter()
            all_keywords = Counter()
            
            for paper in paper_data["papers"]:
                if paper.get('application_area'):
                    application_areas[paper['application_area']] += 1
                for keyword in paper.get('keywords', []):
                    if keyword:
                        all_keywords[keyword] += 1
            
            context = {
                "communities": communities,
                "top_application_areas": dict(application_areas.most_common(10)),
                "top_keywords": dict(all_keywords.most_common(20)),
                "sample_papers": paper_data["papers"][:10]
            }
            
            prompt = """
            Based on the research communities, application areas, and keywords data;
            analyze and describe the major research areas in academic style, approximately in 80-100 words 
            
            Structure your response to cover:
            1. Main research themes and clusters
            2. Interdisciplinary connections
            3. Emerging areas of focus
            4. Geographic or institutional concentrations
            
            """
            
            return self._call_ollama(prompt, context)
        except Exception as e:
            self.logger.exception(f"There is an exception while extracting research areas {e}")
            raise
    
    
    
    def _generate_methodologies(self) -> str:
        try:
            """Generate methodologies analysis"""
            methodology_data = self._extract_methodologies()
            
            context = {
                "methodologies": methodology_data["methodologies"],
                "methodology_evolution": methodology_data["methodology_evolution"]
            }
            
            prompt = """
            Analyze the methodological approaches used in this research domain based on the provided data.
            
            Your analysis should cover:
            1. Dominant methodological approaches
            2. Evolution of methods over time
            3. Methodological trends and patterns
            4. Strengths and limitations of different approaches
            
            Write in academic style, approximately 150-200 words.
            """
            
            return self._call_ollama(prompt, context)
        
        except Exception as e:
            self.logger.exception(f"There is an error while generating methodology summary {e}")
            raise
    
    
    
    def _generate_key_contributions(self) -> str:
        try:
            """Generate key contributions analysis"""
            influential_works = self._extract_influential_works()
            paper_data = self._extract_paper_data()
            
            # Extract key findings and contributions
            key_findings = []
            for paper in paper_data["papers"][:15]:
                if paper.get('main_findings'):
                    key_findings.append({
                        "title": paper.get('title'),
                        "year": paper.get('year'),
                        "findings": paper.get('main_findings'),
                        "citations": paper.get('citations_count', 0)
                    })
            
            context = {
                "influential_papers": influential_works["influential_papers"],
                "influential_authors": influential_works["influential_authors"],
                "key_findings": key_findings
            }
            
            prompt = """
                Below is the data you need. Use it to write a single, self-contained academic-style synthesis of about 400 words. Organize strictly under these five headings (in this order) and do not add any extra sections or commentary:

                1. Breakthrough discoveries and innovations  
                2. Theoretical contributions and frameworks  
                3. Methodological advances  
                4. Practical applications and implementations  
                5. Most influential works and their impact  

                Instructions:
                • Stay within ~400 words (≈2,500 tokens at most).
                • Use an academic tone but keep it concise.
                • Don’t repeat the prompt or include any extra preamble or closing.
                • Do not include any citation number or references such as [1], [2] etc.
            """
            
            return self._call_ollama(prompt, context)
        
        except Exception as e:
            self.logger.error(f"There is an exception while generating key contributors {e}")
            raise
    
    
    def _generate_future_directions(self) -> str:
        try:
            """Generate future research directions"""
            limitations_data = self._extract_limitations_and_gaps()
            recent_papers = self._extract_paper_data()["papers"][:10]
            
            context = {
                "limitations": limitations_data["limitations"],
                "limitation_relationships": limitations_data["limitation_relationships"],
                "recent_developments": recent_papers
            }
            
            prompt = """
            Based on the identified limitations, research gaps, and recent developments, 
            propose future research directions for this domain in 100-150 words in academic style.
            Do not include any citation number or references such as [1], [2] etc.
            
            Your analysis should cover:
            1. Unresolved research questions and gaps
            2. Emerging opportunities and challenges
            3. Technological and methodological opportunities
            4. Interdisciplinary research potential
            5. Practical applications and societal impact
            
            """
            
            return self._call_ollama(prompt, context)
        
        except Exception as e:
            self.logger.error(f"There is an exception while generating future directions {e}")
            raise
    
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
    
    
    
    def generate_review(self) -> Dict[str, str]:
        """Generate comprehensive literature review"""
        self.logger.info("Generating literature review...")
        
        try:
            intro, paper_data = self._generate_introduction()
            top_papers = sorted(
                paper_data["papers"], 
                key=lambda x: (x.get('citations_count', 0), x.get('year', 0)), 
                reverse=True
            )

            top_40_percent = top_papers[:int(len(top_papers) * 0.4)]
            
            self.logger.info(f"{'=='*20} Final Summary: Introduction generated.")

            research_areas = self._generate_research_areas()
            self.logger.info(f"{'=='*20} Final Summary: Research Areas generated.")
            
            methodologies = self._generate_methodologies()
            self.logger.info(f"{'=='*20} Final Summary: Methodologies generated.")
            
            key_contributions = self._generate_key_contributions()
            self.logger.info(f"{'=='*20} Final Summary: Key Contributions generated.")

            future_directions = self._generate_future_directions()
            self.logger.info(f"{'=='*20} Final Summary: Future Direction generated.")
            
            review = {
                "introduction": intro,
                "research_areas": research_areas,
                "methodologies": methodologies,
                "contributions": key_contributions,
                "future_directions": future_directions,
                "references": self.format_references(top_40_percent)
            }
            
            self.logger.info("Literature review generation completed successfully")
            self.export_review(review=review)
            return review
            
        except Exception as e:
            self.logger.error(f"Literature review generation failed: {e}")
            return {
                "error": f"Failed to generate literature review: {str(e)}",
                "introduction": "",
                "research_areas": "",
                "methodologies": "",
                "contributions": "",
                "future_directions": "",
                "references":""
            }
    
    def export_review(self, review: Dict[str, str], output_path: str = "literature_review.md"):
        """Export the generated review to a markdown file"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("# Literature Review\n\n")
                
                sections = [
                    ("Introduction", "introduction"),
                    ("Research Areas and Themes", "research_areas"),
                    ("Methodological Approaches", "methodologies"),
                    ("Key Contributions and Findings", "contributions"),
                    ("Future Research Directions", "future_directions"),
                    ("References", "references")
                ]
                
                for section_title, section_key in sections:
                    f.write(f"## {section_title}\n\n")
                    f.write(f"{review.get(section_key, 'Content not available')}\n\n")
            
            self.logger.info(f"Literature review exported to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to export literature review: {e}")
    
    def cleanup_graph_projections(self):
        """Clean up GDS graph projections"""
        try:
            self._query_graph("CALL gds.graph.drop('literature-network')")
        except:
            pass  # Projection might not exist