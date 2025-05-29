import logging
from pathlib import Path
import argparse
from .data.loader import DataLoader
from .data.preprocessor import DataPreprocessor
from .graph.builder import GraphBuilder
from .llm.integrator import LlamaIndexIntegrator
from .llm.query_engine import GraphQueryEngine
from .utils.config import ConfigManager
from .utils.logger import setup_logger

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
        importance_scores = self.graph_builder.get_node_importance_scores()
        # print(f"\nMain: testing importance score : {importance_scores}")

        # Access rich paper information
        paper_node = graph.nodes['paper_0']
        print(f"\nMain: {paper_node['title'], paper_node['authors'], paper_node['main_findings']}")
        
        # Log graph statistics
        num_nodes = len(graph.nodes())
        num_edges = len(graph.edges())
        self.logger.info(f"Graph built: {num_nodes} nodes, {num_edges} edges")
        
        return graph
    
    def setup_llm_integration(self, graph):
        """Setup LLM integration with LlamaIndex."""
        self.logger.info("Setting up LLM integration")
        
        self.llm_integrator = LlamaIndexIntegrator()
        self.llm_integrator.build_index(graph)
        
        self.query_engine = GraphQueryEngine(graph, self.llm_integrator)
        self.logger.info("LLM integration completed")
    
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
    
    def run(self, file_path: str, interactive: bool = True):
        """Main execution method."""
        try:
            # Load and preprocess data
            df = self.load_and_preprocess_data(file_path)
            print(f"\nMain: data after load_and_preprocess is \n")
            print(f"\n{df.columns}")
            print(f"\n{df.head()}")
            

            ## Build graph
            graph = self.build_graph(df)
            print(f"\nMain: build_graph is completed.")

            
            # Setup LLM integration
            self.setup_llm_integration(graph)
            
            # if interactive:
            #     self.run_interactive_mode()
            
            # return graph, self.query_engine
            
        except Exception as e:
            self.logger.error(f"Application error: {e}")
            raise

def main():
    parser = argparse.ArgumentParser(description="Research Paper Knowledge Graph")
    parser.add_argument("--file", required=True, help="Path to Excel/CSV file")
    parser.add_argument("--config", default="config/config.yaml", help="Config file path")
    parser.add_argument("--no-interactive", action="store_true", help="Disable interactive mode")
    
    args = parser.parse_args()
    print(args)
    
    app = ResearchGraphApplication(args.config)
    app.run(args.file, interactive=not args.no_interactive)


if __name__ == "__main__":
    main()