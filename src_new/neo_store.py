from typing import Dict, Any, List
from neo4j import GraphDatabase
import logging
from .neo_schema import Neo4jSchemaManager
from .nodes import NodeType
from .relationships import RelationshipType
import json
from collections import defaultdict

class Neo4jGraphStore:
    def __init__(self, config: dict):
        self.config = config
        self.driver = GraphDatabase.driver(
            config['uri'],
            auth=(config['user'], config['password'])
        )
        self.database = config.get('database', 'neo4j')
        self.schema_manager = Neo4jSchemaManager(self)
        self.logger = logging.getLogger(__name__)
        
        if config.get('initialize_schema', True):
            if not self.schema_manager.initialize_schema():
                raise RuntimeError("Failed to initialize database schema")

    def execute(self, query: str, parameters: dict = None):
        """Execute a Cypher query with parameters"""
        with self.driver.session(database=self.database) as session:
            try:
                result = session.run(query, parameters or {})
                return list(result)
            except Exception as e:
                self.logger.error(f"Query failed: {query[:100]}... - {str(e)}")
                raise

    def insert_node(self, node: Dict):
        """Insert or update a node and return success status"""
        try:
            # Extract labels and id
            labels = node.get('labels', [])
            if not labels:
                raise ValueError("Node must have labels")
            
            node_id = node['properties']['id']
            properties = node['properties']
            
            # Create label string for query
            label_str = ':'.join(labels)
            
            query = f"""
            MERGE (n:`{label_str}` {{id: $id}})
            SET n += $properties
            RETURN n.id as node_id
            """
            
            result = self.execute(query, {
                'id': node_id,
                'properties': properties
            })
            
            return result[0]['node_id'] if result else None
            
        except Exception as e:
            self.logger.error(f"Node insertion failed: {str(e)}")
            raise

    def insert_relationship(self, relationship: Dict):
        """Insert or update a relationship using node IDs instead of element IDs"""
        try:
            rel_type = relationship.get('relationship_type', '')
            if not RelationshipType.has_value(rel_type):
                raise ValueError(f"Invalid relationship type: {rel_type}")
            
            source_node_id = relationship['source_id']
            target_node_id = relationship['target_id']
            properties = relationship.get('properties', {})
            
            # Use node IDs instead of element IDs for more reliable matching
            query = f"""
            MATCH (src {{id: $source_node_id}}), (tgt {{id: $target_node_id}})
            MERGE (src)-[r:`{rel_type}`]->(tgt)
            SET r += $properties
            RETURN r
            """
            
            result = self.execute(query, {
                'source_node_id': source_node_id,
                'target_node_id': target_node_id,
                'properties': properties
            })
            
            if result:
                self.logger.debug(f"Created relationship {rel_type}: {source_node_id} -> {target_node_id}")
                return True
            else:
                self.logger.warning(f"Failed to create relationship {rel_type}: {source_node_id} -> {target_node_id}")
                return False
            
        except Exception as e:
            self.logger.error(f"Relationship insertion failed: {str(e)}")
            raise

    def get_node_counts(self):
        """Get count of nodes by type"""
        query = """
        CALL db.labels() YIELD label
        CALL apoc.cypher.run('MATCH (n:`' + label + '`) RETURN count(n) as count', {}) 
        YIELD value
        RETURN label, value.count as count
        """
        try:
            result = self.execute(query)
            return {record['label']: record['count'] for record in result}
        except:
            # Fallback if APOC is not available
            counts = {}
            for node_type in ['PAPER', 'AUTHOR', 'INSTITUTION', 'JOURNAL', 'KEYWORD']:
                query = f"MATCH (n:`{node_type}`) RETURN count(n) as count"
                result = self.execute(query)
                counts[node_type] = result[0]['count'] if result else 0
            return counts

    def get_relationship_counts(self):
        """Get count of relationships by type"""
        query = """
        CALL db.relationshipTypes() YIELD relationshipType
        CALL apoc.cypher.run('MATCH ()-[r:`' + relationshipType + '`]->() RETURN count(r) as count', {}) 
        YIELD value
        RETURN relationshipType, value.count as count
        """
        try:
            result = self.execute(query)
            return {record['relationshipType']: record['count'] for record in result}
        except:
            # Fallback if APOC is not available
            counts = {}
            for rel_type in ['AUTHORED_BY', 'AFFILIATED_WITH', 'PUBLISHED_IN', 'HAS_KEYWORD', 
                             'TEMPORAL_SUCCESSOR', 'SEMANTIC_SIMILAR', 'ADDRESSES_LIMITATION']:
                query = f"MATCH ()-[r:`{rel_type}`]->() RETURN count(r) as count"
                result = self.execute(query)
                counts[rel_type] = result[0]['count'] if result else 0
            return counts

    def close(self):
        """Close the database connection"""
        if self.driver:
            self.driver.close()
