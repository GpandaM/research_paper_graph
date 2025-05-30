import json
from typing import List, Dict, Any
from nebula3.gclient.net import ConnectionPool
from nebula3.Config import Config

class NebulaGraphStore:
    def __init__(self, config: dict):
        self.config = config
        self.pool = ConnectionPool()
        cfg = Config()
        self.pool.init([(config['host'], config['port'])], cfg)

    def execute(self, query: str):
        with self.pool.session_context(self.config['user'], self.config['password']) as session:
            return session.execute(query)

    def insert_node(self, node: Dict):
        """Insert any type of node into Nebula Graph with proper schema handling."""
        try:
            node_type = node.get('type', '')
            node_id = node['id']
            
            # Helper function to clean and escape strings for Nebula
            def clean_string(value):
                if value is None:
                    return ''
                if isinstance(value, list):
                    return json.dumps(value)
                return str(value).replace('"', '\\"').replace('\n', '\\n')
            
            # Handle different node types
            if node_type == 'paper':
                query = f"""
                INSERT VERTEX paper (
                    title, year, authors, institutions, 
                    journal, keywords, methodology, main_findings,
                    equations_models, application_area, strengths,
                    limitations, citations_count, doi
                ) VALUES "{node_id}":(
                    "{clean_string(node.get('title', ''))}", 
                    {int(node.get('year', 0))}, 
                    {clean_string(node.get('authors', []))}, 
                    {clean_string(node.get('institutions', []))},
                    "{clean_string(node.get('journal', ''))}", 
                    {clean_string(node.get('keywords', []))}, 
                    "{clean_string(node.get('methodology', ''))}", 
                    "{clean_string(node.get('main_findings', ''))}",
                    "{clean_string(node.get('equations_models', ''))}",
                    "{clean_string(node.get('application_area', ''))}",
                    "{clean_string(node.get('strengths', ''))}",
                    "{clean_string(node.get('limitations', ''))}",
                    {int(node.get('citations_count', 0))},
                    "{clean_string(node.get('doi', ''))}"
                )
                """
            elif node_type == 'author':
                query = f"""
                INSERT VERTEX author (
                    name, display_name, metadata
                ) VALUES "{node_id}":(
                    "{clean_string(node.get('name', ''))}",
                    "{clean_string(node.get('properties', {}).get('display_name', ''))}",
                    "{clean_string(node.get('properties', {}))}"
                )
                """
            elif node_type == 'institution':
                query = f"""
                INSERT VERTEX institution (
                    name, display_name, metadata
                ) VALUES "{node_id}":(
                    "{clean_string(node.get('name', ''))}",
                    "{clean_string(node.get('properties', {}).get('display_name', ''))}",
                    "{clean_string(node.get('properties', {}))}"
                )
                """
            elif node_type == 'journal':
                query = f"""
                INSERT VERTEX journal (
                    name, display_name, metadata
                ) VALUES "{node_id}":(
                    "{clean_string(node.get('name', ''))}",
                    "{clean_string(node.get('properties', {}).get('display_name', ''))}",
                    "{clean_string(node.get('properties', {}))}"
                )
                """
            elif node_type == 'keyword':
                query = f"""
                INSERT VERTEX keyword (
                    keyword, display_name, metadata
                ) VALUES "{node_id}":(
                    "{clean_string(node.get('properties', {}).get('keyword', ''))}",
                    "{clean_string(node.get('properties', {}).get('display_name', ''))}",
                    "{clean_string(node.get('properties', {}))}"
                )
                """
            else:
                print(f"Unknown node type: {node_type}")
                return None
            
            print("Generated Query:", query)  # For debugging
            return self.execute(query)
        
        except Exception as e:
            print(f"Error inserting node {node.get('id', 'unknown')}: {str(e)}")
            print("Problematic node data:", node)
            return None
    
    
    
    def insert_relationship(self, relationship: Dict):
        """Insert a relationship into Nebula Graph."""
        try:
            rel_type = relationship.get('relationship_type', '')
            source_id = relationship['source_id']
            target_id = relationship['target_id']
            properties = relationship.get('properties', {})
            weight = relationship.get('weight', 1.0)
            
            # Helper to format properties
            def format_props(props):
                if not props:
                    return ""
                prop_str = ", ".join([f"{k}: {json.dumps(v)}" for k, v in props.items()])
                return f"@{prop_str}"
            
            if rel_type == 'AUTHORED_BY':
                query = f"""
                INSERT EDGE AUTHORED_BY() VALUES "{source_id}" -> "{target_id}":()
                """
            elif rel_type == 'AFFILIATED_WITH':
                query = f"""
                INSERT EDGE AFFILIATED_WITH() VALUES "{source_id}" -> "{target_id}":()
                """
            elif rel_type == 'PUBLISHED_IN':
                query = f"""
                INSERT EDGE PUBLISHED_IN() VALUES "{source_id}" -> "{target_id}":()
                """
            elif rel_type == 'HAS_KEYWORD':
                query = f"""
                INSERT EDGE HAS_KEYWORD() VALUES "{source_id}" -> "{target_id}":()
                """
            elif rel_type == 'SIMILAR_TO':
                query = f"""
                INSERT EDGE SIMILAR_TO(similarity_score) VALUES "{source_id}" -> "{target_id}":({properties.get('similarity_score', 0.0)})
                """
            elif rel_type == 'COLLABORATES_WITH':
                query = f"""
                INSERT EDGE COLLABORATES_WITH(shared_papers, weight) VALUES "{source_id}" -> "{target_id}":(
                    "{clean_string(properties.get('shared_papers', []))}",
                    {float(weight)}
                )
                """
            else:
                print(f"Unknown relationship type: {rel_type}")
                return None
            
            return self.execute(query)
        
        except Exception as e:
            print(f"Error inserting relationship: {str(e)}")
            print("Problematic relationship:", relationship)
            return None