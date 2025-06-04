from neo4j import GraphDatabase


class ResearchKnowledgeGraph:
    def __init__(self, neo4j_config: dict):
        self.driver  = GraphDatabase.driver(
            neo4j_config["uri"],
            auth=(neo4j_config["user"], neo4j_config["password"]),
            database=neo4j_config.get("database", "neo4j")
        )
        # self.embedding_model = OllamaEmbedding()
        # self.generator = OllamaGenerator()
        self.similarity_threshold = 0.7
        
    def close(self):
        self.driver.close()
    
    def clear_database(self):
        """Clear the entire database"""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")