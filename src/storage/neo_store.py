from typing import Dict, Any, List
from neo4j import GraphDatabase
import logging
from .neo_schema import Neo4jSchemaManager
from ..graph.nodes import NodeType
from ..graph.relationships import RelationshipType
import json

class Neo4jGraphStore:
    def __init__(self, config: dict):
        self.config = config
        self.driver = GraphDatabase.driver(
            config['uri'],
            auth=(config['user'], config['password'])
        )
        self.database = config.get('database', 'neo4j')
        self.schema_manager = Neo4jSchemaManager(self)
        
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
                self.schema_manager.logger.error(f"Query failed: {query[:100]}... - {str(e)}")
                raise

    def insert_node(self, node: Dict):
        """Insert a node with flattened properties for Neo4j"""
        try:
            node_type = node.get('type', '').upper()
            if not hasattr(NodeType, node_type):
                raise ValueError(f"Invalid node type: {node_type}")
            
            node_id = node['id']
            properties = {
                k: v
                for k, v in node.items()
                if k not in ['id', 'type'] and v is not None and not isinstance(v, dict)
            }
            
            # Optionally: Flatten dict properties (like metadata) into top-level keys
            for k, v in node.items():
                if isinstance(v, dict):
                    for sub_k, sub_v in v.items():
                        properties[f"{k}_{sub_k}"] = sub_v
            
            query = f"""
            MERGE (n:{node_type} {{id: $id}})
            SET n += $properties
            """
            
            return self.execute(query, {
                'id': node_id,
                'properties': properties
            })
            
        except Exception as e:
            self.schema_manager.logger.error(f"Node insertion failed: {str(e)}")
            raise


    def insert_relationship(self, relationship: Dict):
        """Insert a relationship with type and properties"""
        rel_type = relationship.get('relationship_type', '').upper()
        source_id = relationship['source_id']
        target_id = relationship['target_id']
        properties = relationship.get('properties', {})

        # 1) enum‐membership check
        if rel_type not in RelationshipType._value2member_map_:
            raise ValueError(f"Invalid relationship type: {rel_type}")

        query = """
        MATCH (src), (tgt)
        WHERE src.id = $source_id AND tgt.id = $target_id
        CREATE (src)-[r:%s $properties]->(tgt)
        """ % rel_type

        params = {
            'source_id': source_id,
            'target_id': target_id,
            'properties': properties
        }

        return self.execute(query, params)


    def batch_insert_nodes(self, nodes: List[Dict], batch_size: int = 1000):
        """Batch insert nodes for better performance"""
        for i in range(0, len(nodes), batch_size):
            batch = nodes[i:i + batch_size]
            query = """
            UNWIND $batch AS node
            CREATE (n:Node {id: node.id})
            SET n += node.properties, n:Label
            """
            
            # Group by type for more efficient insertion
            for node_type, group in self._group_by_type(batch):
                typed_query = query.replace("Label", node_type)
                params = {
                    'batch': [{
                        'id': n['id'],
                        'properties': {k: v for k, v in n.items() if k not in ['id', 'type']}
                    } for n in group]
                }
                self.execute(typed_query, params)

    def batch_insert_relationships(self, relationships: List[Dict], batch_size: int = 1000):
        """Batch insert relationships"""
        for i in range(0, len(relationships), batch_size):
            batch = relationships[i:i + batch_size]
            query = """
            UNWIND $batch AS rel
            MATCH (src {id: rel.source_id}), (tgt {id: rel.target_id})
            CREATE (src)-[r:REL_TYPE $rel.properties]->(tgt)
            """
            
            # Group by relationship type
            for rel_type, group in self._group_by_rel_type(batch):
                typed_query = query.replace("REL_TYPE", rel_type)
                params = {
                    'batch': [{
                        'source_id': r['source_id'],
                        'target_id': r['target_id'],
                        'properties': r.get('properties', {})
                    } for r in group]
                }
                self.execute(typed_query, params)

    def _group_by_type(self, nodes: List[Dict]):
        """Group nodes by their type"""
        groups = {}
        for node in nodes:
            node_type = node.get('type', '')
            if node_type not in groups:
                groups[node_type] = []
            groups[node_type].append(node)
        return groups.items()

    def _group_by_rel_type(self, relationships: List[Dict]):
        """Group relationships by their type"""
        groups = {}
        for rel in relationships:
            rel_type = rel.get('relationship_type', '')
            if rel_type not in groups:
                groups[rel_type] = []
            groups[rel_type].append(rel)
        return groups.items()

    def close(self):
        """Close the Neo4j driver connection"""
        self.driver.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()