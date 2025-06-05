import logging

class Neo4jSchemaManager:
    """Handles database schema creation, indexes, and constraints"""
    
    def __init__(self, neo4j_store):
        self.store = neo4j_store
        self.logger = logging.getLogger(__name__)
    
    def initialize_schema(self):
        """Create all required constraints and indexes"""
        try:
            self.create_constraints()
            self.create_indexes()
            self.create_fulltext_indexes()
            # Try vector index but continue if fails
            try:
                if self.create_vector_indexes():
                    self.logger.info("Vector indexes enabled")
            except Exception as e:
                self.logger.warning(f"Vector indexes not available: {e}")
            
            self.logger.info("Database schema initialized successfully")
            return True
        
        except Exception as e:
            self.logger.error(f"Schema initialization failed: {str(e)}")
            return False
    
    def create_constraints(self):
        """Create uniqueness constraints"""
        constraints = [
            ("Paper", "id", "paper_id_unique"),
            ("Paper", "doi", "paper_doi_unique"),
            ("Author", "id", "author_id_unique"),
            ("Institution", "id", "institution_id_unique"),
            ("Journal", "id", "journal_id_unique"),
            ("Keyword", "keyword", "keyword_value_unique")
        ]
        
        for label, prop, name in constraints:
            query = f"""
            CREATE CONSTRAINT {name} IF NOT EXISTS 
            FOR (n:{label}) REQUIRE n.{prop} IS UNIQUE
            """
            self.store.execute(query)
    
    def create_indexes(self):
        """Create single-property indexes"""
        indexes = [
            ("Paper", "title"),
            ("Paper", "year"),
            ("Paper", "citations_count"),
            ("Author", "name"),
            ("Institution", "name"),
            ("Journal", "name"),
            ("Keyword", "display_name")
        ]
        
        for label, prop in indexes:
            query = f"""
            CREATE INDEX {label.lower()}_{prop}_index IF NOT EXISTS 
            FOR (n:{label}) ON (n.{prop})
            """
            self.store.execute(query)
    
    def create_fulltext_indexes(self):
        """Create full-text search indexes with correct syntax"""
        fulltext_indexes = [
            {
                "name": "paper_content_index",
                "labels": ["Paper"],
                "properties": ["title", "abstract", "keywords", "main_findings"],
                "analyzer": "english"
            }
        ]
        
        for index in fulltext_indexes:
            labels = '|'.join(index['labels'])
            props = ', '.join([f"n.{p}" for p in index['properties']])
            
            query = f"""
            CREATE FULLTEXT INDEX {index['name']} IF NOT EXISTS
            FOR (n:{labels}) ON EACH [{props}]
            """
            
            if index.get("analyzer"):
                query += f"""
                OPTIONS {{
                    indexConfig: {{
                        `fulltext.analyzer`: '{index["analyzer"]}'
                    }}
                }}
                """
            
            self.store.execute(query)


    def create_vector_indexes(self):
        """Create vector indexes for embedding-based search (Neo4j 5.11+)"""
        try:
            # Check if vector index is supported
            version_query = "CALL dbms.components() YIELD versions"
            result = self.store.execute(version_query)
            neo4j_version = result[0]["versions"][0]  # Gets first version number
            
            if not neo4j_version.startswith("5."):
                self.logger.warning(f"Vector indexes require Neo4j 5.11+. Current version: {neo4j_version}")
                return False

            ## Ollama's nomic-embed-text uses 768 by default --> change it you change the model
            ## cosine similarity is most common for text
            query = """
            CREATE VECTOR INDEX paper_embeddings_index IF NOT EXISTS
            FOR (p:Paper) ON (p.embedding)
            OPTIONS {
                indexConfig: {
                    `vector.dimensions`: 768,
                    `vector.similarity_function`: 'cosine'
                }
            }
            """
            self.store.execute(query)
            self.logger.info("Vector index created successfully")
            return True
        except Exception as e:
            self.logger.error(f"Vector index creation failed: {str(e)}")
            return False
