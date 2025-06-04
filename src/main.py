import logging
from pathlib import Path
import argparse
import asyncio
from contextlib import asynccontextmanager
import pandas as pd
import networkx as nx
from .data.loader import DataLoader
from .data.preprocessor import DataPreprocessor
from .graph.builder import GraphBuilder
from .llm.integrator import GraphQueryEngine
from .llm.neo_retrieval import Neo4jRetrievalEngine
from .utils.config import ConfigManager
from .utils.logger import setup_logger
from .storage.neo_store import Neo4jGraphStore
from .storage.weaviate_schema import WeaviateSetup
from .storage.weaviate_store import WeaviateVectorStore
from .graph.nodes import RichPaperNode

import warnings
warnings.filterwarnings("ignore")  # Ignore all warnings

class ResearchGraphApplication:
    """Main application class for the research paper knowledge graph."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = ConfigManager(config_path)
        self.logger = setup_logger()
        
        # Initialize components
        self.data_loader = DataLoader(self.logger)
        self.preprocessor = DataPreprocessor()
        self.graph_builder = GraphBuilder()
        self.llm_integrator = None
        self.query_engine = None
        
    def load_and_preprocess_data(self, file_path: str):
        """Load and preprocess research paper data."""
        self.logger.info(f"Loading data from {file_path}")
        
        # Load data
        if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            df = self.data_loader.load_excel(file_path)
        else:
            df = self.data_loader.load_csv(file_path)
        
        # Preprocess
        df_processed = self.preprocessor.preprocess_dataframe(df)
        self.logger.info("Data preprocessing completed")
        
        return df_processed
    
    def build_graph(self, df):
        """Build knowledge graph from preprocessed data."""
        self.logger.info("Building knowledge graph")
        graph = self.graph_builder.build_from_dataframe(df)

        # Get importance scores for ranking
        # importance_scores = self.graph_builder.get_node_importance_scores()
        # print(f"\nMain: testing importance score : {importance_scores}")

        # Access rich paper information
        paper_node = graph.nodes['paper_0']
        print(f"\nMain: {paper_node['title'], paper_node['authors'], paper_node['main_findings']}")
        
        # Log graph statistics
        num_nodes = len(graph.nodes())
        num_edges = len(graph.edges())
        self.logger.info(f"Graph built: {num_nodes} nodes, {num_edges} edges")
        
        return graph
    
    # def setup_llm_integration(self, graph):
    #     """Setup LLM integration with LlamaIndex."""
    #     self.logger.info("Setting up LLM integration")
        
    #     self.llm_integrator = LlamaIndexIntegrator()
    #     self.llm_integrator.build_index(graph)
        
    #     self.query_engine = GraphQueryEngine(graph, self.llm_integrator)
    #     self.logger.info("LLM integration completed")
    
    
    def run_interactive_mode(self):
        """Run interactive query mode."""
        print("\n=== Research Paper Knowledge Graph Query Interface ===")
        print("Available commands:")
        print("1. summarize <keyword> - Summarize research by keyword")
        print("2. compare <method1,method2> - Compare methodologies")
        print("3. trends - Find research trends")
        print("4. author <name> - Get author expertise")
        print("5. quit - Exit")
        
        while True:
            try:
                command = input("\nEnter command: ").strip()
                
                if command.lower() == 'quit':
                    break
                elif command.startswith('summarize '):
                    keyword = command[10:]
                    result = self.query_engine.summarize_by_keyword(keyword)
                    print(f"\nSummary for '{keyword}':\n{result}")
                    
                elif command.startswith('compare '):
                    methods = command[8:].split(',')
                    if len(methods) == 2:
                        result = self.query_engine.compare_methodologies(
                            methods[0].strip(), methods[1].strip()
                        )
                        print(f"\nMethodology Comparison:\n{result}")
                    else:
                        print("Please provide two methods separated by comma")
                        
                elif command == 'trends':
                    result = self.query_engine.find_research_trends()
                    print(f"\nResearch Trends:\n{result}")
                    
                elif command.startswith('author '):
                    author = command[7:]
                    result = self.query_engine.get_author_expertise(author)
                    print(f"\nAuthor Expertise for '{author}':\n{result}")
                    
                else:
                    print("Unknown command. Please try again.")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
    

    async def run(self, file_path: str, interactive: bool = True):
        
        ## --------------------------------------------------------------------------------------------- ##
        
        # Load and preprocess data
        # df = self.load_and_preprocess_data(file_path)
        
        ## --------------------------------------------------------------------------------------------- ##
        
        # # Build and export to Neo4j
        # graph = self.graph_builder.build_from_dataframe(df)
        # self.export_to_neo4j(graph)

        ## --------------------------------------------------------------------------------------------- ##

        # Export to Weaviate - AFTER building the graph
        # schema_manager = WeaviateSetup()
        # schema_manager.create_research_paper_schema()
        # self.export_to_weaviate(graph)  

        ## --------------------------------------------------------------------------------------------- ##


        ## ------------------------------ verify the insertion of weaviate ------------------------------ ##
        # weaviate_store = WeaviateVectorStore()
        # weaviate_store.test_single_insert()
        # weaviate_store.peek_a_boo()

        # 1. Check sample data
        # weaviate_store.verify_weaviate_data(sample_size=3)

        # 2. Verify total count
        # inserted_count = weaviate_store.count_inserted_papers()
        # print(f"Expected: 100, Inserted: {inserted_count}")

        # 3. Search for specific papers
        # weaviate_store.search_paper_by_title("Euler-Lagrange")

        # 4. Check vectors (use an ID from your sample)
        # sample_id = "100"  # Get from verify_weaviate_data()
        # weaviate_store.check_vector_embeddings(sample_id)

        # 5. Full quality report
        # weaviate_store.generate_data_quality_report()
        ## --------------------------------------------------------------------------------------------- ##
    
     
        # Initialize query engine
        neo4j_config = self.config.get('neo4j')
        weaviate_config = self.config.get('weaviate')

        async with self.get_query_engine(neo4j_config, weaviate_config) as query_engine:
            res = await query_engine.summarize("machine learning")
            print(res)

        # print(self.query_engine.summarize("machine learning"))

        
        # try:
        #     # Initialize the engine
        #     engine = Neo4jRetrievalEngine(neo4j_config)
            
        #     # Test connection
        #     if engine.test_connection():
        #         print("✅ Neo4j connection successful")
                
        #         # Get graph statistics
        #         stats = engine.get_graph_stats()
        #         print(f"📊 Graph stats: {stats}")
                
        #         # Example query
        #         result = await engine.query("What papers are about machine learning?")
        #         print(f"🔍 Query result: {result}")
                
        #     else:
        #         print("❌ Neo4j connection failed")
                
        # except Exception as e:
        #     print(f"❌ Error: {e}")
        
        # finally:
        #     if 'engine' in locals():
        #         engine.close()

        ## --------------------------------------------------------------------------------------------- ##


    @asynccontextmanager
    async def get_query_engine(self, neo_config: dict, weav_config: dict):
        engine = GraphQueryEngine(neo_config, weav_config)
        try:
            await engine.initialize()
            yield engine
        finally:
            await engine.close()
    

    def export_to_neo4j(self, graph: nx.MultiDiGraph):
        neo4j_store = Neo4jGraphStore(self.config.get('neo4j'))
        
        # First pass: Create all nodes
        for node_id, data in graph.nodes(data=True):
            node_data = {
                'id': node_id,
                'type': data.get('type', ''),
                **{k: v for k, v in data.items() if k != 'node'}  # Exclude the RichPaperNode object
                # 'properties': {
                #     **data,
                #     'id': node_id  # Ensure ID is included in properties
                # }
            }
            print("\n\n")
            print(f"{node_id}, and the type is {data.get('type', '')}")
            print("\n\n")
            neo4j_store.insert_node(node_data)
        
        print("\n\n\n\n---------------------------- FIRST PASS COMPLETED ------------------------------\n\n\n\n\n")
        
        # Second pass: Create relationships after all nodes exist
        for src, dst, data in graph.edges(data=True):
            relationship_data = {
                'source_id': src,
                'target_id': dst,
                'relationship_type': data.get('type', 'RELATED_TO'),
                'properties': {
                    'weight': data.get('weight', 1.0)
                }
            }
            neo4j_store.insert_relationship(relationship_data)
        
        print("\n\n\n\n---------------------------- SECOND PASS COMPLETED ------------------------------\n\n\n\n\n")


    def export_to_weaviate(self, graph):
        """Export processed paper nodes to Weaviate"""
        try:
            weaviate_store = WeaviateVectorStore()
            
            # Get all paper nodes from the graph using the node data
            paper_nodes = []
            for node_id in graph.nodes:
                node_data = graph.nodes[node_id]
                print(node_data)
                print("\n")
                if 'node' in node_data and isinstance(node_data['node'], RichPaperNode):
                    paper_nodes.append(node_data['node'])
            
            print(f"Exporting {len(paper_nodes)} papers to Weaviate...")
            
            if paper_nodes:
                weaviate_store.batch_insert_papers(paper_nodes)
                print(f"Successfully exported {len(paper_nodes)} papers to Weaviate")
            else:
                print("No valid paper nodes found to export")

            '''  
            # if not paper_nodes:
            #     print("No valid paper nodes found to export")
            #     return
                
            # Insert papers one by one with progress tracking
            # success_count = 0
            # from tqdm import tqdm
            # for i, paper_node in enumerate(tqdm(paper_nodes, desc="Inserting papers"), 1):
            #     if weaviate_store.insert_paper_to_weaviate(paper_node):
            #         success_count += 1
            #         if i % 10 == 0 or i == len(paper_nodes):
            #             print(f"✅ Added paper {i}: {paper_node.title[:50]}...")
            #     else:
            #         print(f"❌ Failed to add paper {i}: {paper_node.title[:50]}...")
            '''
                    
            
            # print(f"Successfully exported {success_count}/{len(paper_nodes)} papers to Weaviate")
            
        except Exception as e:
            print(f"Error exporting to Weaviate: {e}")
            raise


    def export_to_weaviate_from_df(self, df: pd.DataFrame):
        """Export directly from DataFrame to Weaviate"""
        weaviate_store = WeaviateVectorStore()

        print(df.columns)
        
        for idx, row in df.iterrows():
            data_object = {
                "title": str(row.get('Title', '')),
                "authors": self._parse_authors(row.get('authors_list', '')),  
                "year": self._safe_int_conversion(row.get('Year'), 0),
                "keywords": self._parse_keywords(row.get('keywords_cleaned', '')),
                "abstract": str(row.get('Abstract', '')), 
                "methodology": str(row.get('Methodology', '')),
                "main_findings": str(row.get('Main Findings', ''))
            }
            
            weaviate_store.client.data_object.create(
                data_object=data_object,
                class_name="ResearchPaper"
            )

    # def export_to_neo4j(self, graph: nx.MultiDiGraph):
    #     neo4j_store = Neo4jGraphStore(self.config.get('neo4j'))
        
    #     # First pass: Create all nodes
    #     for node_id, data in graph.nodes(data=True):
    #         # Use the full node object if available, else fall back to flattened data
    #         node_obj = data.get('node', None)
            
    #         node_data = {
    #             'id': node_id,
    #             'type': node_obj.node_type if node_obj else data.get('type', 'PAPER'),  # Fallback ## 
    #             'properties': {
    #                 **(node_obj.to_dict() if node_obj else data),  # Prefer RichPaperNode data
    #                 'id': node_id  # Ensure ID is included
    #             }
    #         }
    #         neo4j_store.insert_node(node_data)
        
    #     # Second pass: Relationships (unchanged)
    #     for src, dst, data in graph.edges(data=True):
    #         relationship_data = {
    #             'source_id': src,
    #             'target_id': dst,
    #             'relationship_type': data.get('type', 'CITES'),  # e.g., "CITES" for citations
    #             'properties': {
    #                 'weight': data.get('weight', 1.0)
    #             }
    #         }
    #         neo4j_store.insert_relationship(relationship_data)



def main():
    parser = argparse.ArgumentParser(description="Research Paper Knowledge Graph")
    parser.add_argument("--file", required=True, help="Path to Excel/CSV file")
    parser.add_argument("--config", default="config/config.yaml", help="Config file path")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--interactive", action="store_true", default=True, 
                           help="Run in interactive mode (default)")
    mode_group.add_argument("--no-interactive", action="store_true", 
                           help="Disable interactive mode")
    
    args = parser.parse_args()
    print(args)
    
    app = ResearchGraphApplication(args.config)

    if args.no_interactive:
        interactive_mode = False
        print("Running in NON-INTERACTIVE mode")
    else:
        interactive_mode = True
        print("Running in INTERACTIVE mode")

    print(f"Interactive mode: {interactive_mode}")
    # app.run(args.file, interactive=interactive_mode)
    asyncio.run(app.run(args.file, interactive=interactive_mode))


if __name__ == "__main__":
    main()