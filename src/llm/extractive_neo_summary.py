import asyncio
import weaviate
from weaviate.connect import ConnectionParams
from neo4j import GraphDatabase
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import Settings
from typing import List, Dict, Any
import json
import os
import time
import re
from ollama import Client
import requests
from time import sleep
from graphdatascience import GraphDataScience

class SummarizeByCluster:
    def __init__(self, neo4j_config: dict):
        NEO4J_URI = neo4j_config["uri"]
        NEO4J_AUTH = (neo4j_config["user"], neo4j_config["password"])
        self.paper_gds = GraphDataScience(NEO4J_URI, auth=NEO4J_AUTH, database="paperdb")


    def compute_importance_scores(self, current_year=2025):
        """Calculate paper importance scores combining citations and recency"""
        query = """
        MATCH (p:PAPER)
        WITH p, 
            p.citations_count AS citations,
            $current_year - p.year AS age
        SET p.importance = citations + (5 - CASE WHEN age > 5 THEN 5 ELSE age END)
        RETURN count(p) AS updated
        """
        self.paper_gds.run_cypher(query, {"current_year": current_year})

    def create_similarity_relationships(self):
        """Create virtual relationships for community detection"""

        # Shared methodologies
        self.paper_gds.run_cypher("""
        MATCH (p1:PAPER)-[:USES_METHODOLOGY]->(m:METHODOLOGY)<-[:USES_METHODOLOGY]-(p2:PAPER)
        WHERE id(p1) < id(p2)
        MERGE (p1)-[r:SIMILAR_TO {type: "methodology", weight: 1.0}]->(p2)
        """)

        # Shared application areas
        self.paper_gds.run_cypher("""
        MATCH (p1:PAPER)-[:APPLIED_IN]->(a:APPLICATION_AREA)<-[:APPLIED_IN]-(p2:PAPER)
        WHERE id(p1) < id(p2)
        MERGE (p1)-[r:SIMILAR_TO {type: "application", weight: 0.8}]->(p2)
        """)

        # Shared keywords (minimum 2 common keywords)
        self.paper_gds.run_cypher("""
        MATCH (p1:PAPER)-[:HAS_KEYWORD]->(k:KEYWORD)<-[:HAS_KEYWORD]-(p2:PAPER)
        WHERE id(p1) < id(p2)
        WITH p1, p2, count(k) AS common_keywords
        WHERE common_keywords >= 2
        MERGE (p1)-[r:SIMILAR_TO {type: "keywords", weight: 0.6 * common_keywords}]->(p2)
        """)

        # Shared equations or models
        self.paper_gds.run_cypher("""
        MATCH (p1:PAPER)-[:USES_EQUATION]->(e:USES_EQUATION)<-[:USES_EQUATION]-(p2:PAPER)
        WHERE id(p1) < id(p2)
        MERGE (p1)-[r:SIMILAR_TO {type: "equations_models", weight: 0.9}]->(p2)
        """)

        # # Shared main findings (minimum 2 common findings)
        # self.paper_gds.run_cypher("""
        # MATCH (p1:PAPER)-[:HAS_FINDING]->(f:FINDING)<-[:HAS_FINDING]-(p2:PAPER)
        # WHERE id(p1) < id(p2)
        # WITH p1, p2, count(f) AS common_findings
        # WHERE common_findings >= 2
        # MERGE (p1)-[r:SIMILAR_TO {type: "main_findings", weight: 0.7 * common_findings}]->(p2)
        # """)


    def run_community_detection(self):
        """Perform Louvain community detection and store results"""
        # Create graph projection
        
        if self.paper_gds.graph.exists("papers_graph").get("exists"):
            self.paper_gds.graph.drop("papers_graph")
        
        G, _ = self.paper_gds.graph.project(
            "papers_graph",
            ["PAPER"],
            {
                # "CITES": {"orientation": "UNDIRECTED"}, --> weights not found error
                "SIMILAR_TO": {"properties": ["weight"]}
            }
        )
        
        # Run Louvain algorithm
        result = self.paper_gds.louvain.write(
            G,
            relationshipWeightProperty="weight",
            writeProperty="community"
        )
        
        self.paper_gds.graph.drop(G)
        return result

    def summarize_community(self, community_id):
        """Generate summary for a community cluster"""
        query = """
        MATCH (p:PAPER {community: $community_id})
        WITH p
        ORDER BY p.importance DESC
        LIMIT 5

        OPTIONAL MATCH (p)-[:USES_METHODOLOGY]->(m)
        OPTIONAL MATCH (p)-[:APPLIED_IN]->(a)
        OPTIONAL MATCH (p)-[:HAS_KEYWORD]->(k)
        OPTIONAL MATCH (p)-[:USES_EQUATION]->(e)

        WITH collect(DISTINCT {title: p.title, id: p.id}) AS top_papers,
            collect(DISTINCT m.name) AS methodologies,
            collect(DISTINCT a.name) AS applications,
            collect(DISTINCT k.name) AS keywords,
            collect(DISTINCT e.name) AS equations_models

        RETURN {
            top_papers: top_papers,
            core_methodologies: methodologies[..3],
            primary_applications: applications[..3],
            trending_keywords: keywords[..5],
            equations_models: equations_models[..5]
        } AS summary
        """
        return self.paper_gds.run_cypher(query, {"community_id": community_id}).iloc[0]["summary"]


    

    def summarize_all_communities(self):
        """Main workflow to summarize all paper clusters"""
        # Step 1: Compute importance scores
        self.compute_importance_scores()
        
        # Step 2: Create similarity relationships
        self.create_similarity_relationships()
        
        # Step 3: Run community detection
        communities_result = self.run_community_detection()
        print(f"Detected {communities_result['communityCount']} communities")
        
        # Step 4: Summarize each community
        community_ids = self.paper_gds.run_cypher(
            "MATCH (p:PAPER) RETURN DISTINCT p.community AS id ORDER BY id"
        )["id"].tolist()
        
        summaries = {}
        for cid in community_ids:
            summaries[cid] = self.summarize_community(cid)
        
        # Step 5: Generate final report
        final_report = {
            "total_communities": len(community_ids),
            "community_summaries": summaries
        }
        return final_report


    def get_reference_properties_by_paper_ids(self, paper_ids: List[str]) -> List[str]:
        """Fetch paper node details by paper ID using GDS and return formatted references as a list of strings."""
        references = []
        
        for i, paper_id in enumerate(paper_ids, 1):
            query = """
            MATCH (p:PAPER {id: $nodeId})
            RETURN p.authors AS authors, p.year AS year, p.title AS title, p.doi AS doi
            """
            result = self.paper_gds.run_cypher(query, params={"nodeId": paper_id})
            # print(result)
            
            if result is not None and not result.empty:
                row = result.iloc[0]
                authors = row.get('authors', 'Unknown')
                title = row.get('title', 'Untitled')
                doi = row.get('doi', '')
                
                # Clean up authors if it's a list (convert to string)
                if isinstance(authors, list):
                    authors = ', '.join(authors)
                
                # Clean up title (remove newlines and extra spaces)
                if isinstance(title, str):
                    title = title.replace('\n', ' ').strip()
                
                # Format reference with numbering and semicolon separators
                ref = f"[{i}] {authors}; {title}; {doi}"
                references.append(ref)
        
        return '\n'.join(references)

    
    def extract_paper_ids(self, data):
        paper_ids = []

        for summary in data.get("community_summaries", {}).values():
            for paper in summary.get("top_papers", []):
                paper_ids.append(paper.get("id", "paper id not found"))

        # print(paper_ids)
        return paper_ids

    
    def generate_comprehensive_summary(self) -> str:

        combined_summary_text = self.summarize_all_communities()

        # print(combined_summary_text)
        references = self.get_reference_properties_by_paper_ids(self.extract_paper_ids(combined_summary_text))
        
        # print(references)

        if not combined_summary_text:
            return "No papers found in the database."
        
        # Generate final comprehensive summary
        final_prompt = f"""
        You are provided with extractive summary of research papers data.
        Summaries:
        {combined_summary_text}
        
        Write a final abstractive and elaborative summary.
        
        """
        
        print(f"********************************* prompt has {len(final_prompt.split())} words")

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
                    timeout=180  # 180 seconds timeout
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
        
        
        ## Combine everything
        complete_summary = f"""
                        # Comprehensive Literature Review

                        {str(final_summary)}

                        ## References

                        {references}

                        ---
                                """
        
        return complete_summary


if __name__ == "__main__":
    neo4j_config = {
            "uri": "neo4j://localhost:7687",
            "user": "neo4j",
            "password": "Sundeep@123",
            "database": "paperdb"  # Add this line
            }

    def print_community_summaries(data):
        from textwrap import indent

        def format_list(items, bullet="- "):
            return "\n".join(f"{bullet}{item}" for item in items)

        print(f"Total Communities: {data.get('total_communities', 'N/A')}\n")
        print("=" * 80)

        for community_id, summary in data.get("community_summaries", {}).items():
            print(f"\n🔹 Community ID: {community_id}")
            print("-" * 80)

            if 'core_methodologies' in summary:
                print("Core Methodologies:")
                print(indent(format_list(summary['core_methodologies']), prefix="  "))

            if 'trending_keywords' in summary:
                print("\nTrending Keywords:")
                print(indent(", ".join(summary['trending_keywords']), prefix="  "))

            if 'primary_applications' in summary:
                print("\nPrimary Applications:")
                print(indent(format_list(summary['primary_applications']), prefix="  "))

            if 'top_papers' in summary:
                print("\nTop Papers:")
                print(indent(format_list(summary['top_papers']), prefix="  "))

            print("=" * 80)
    

    summarizer = SummarizeByCluster(neo4j_config)
    # report = summarizer.summarize_all_communities()
    report = summarizer.generate_comprehensive_summary()

    print(report)

    # print_community_summaries(report)



    # print("Summarization complete!")