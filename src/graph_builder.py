import hashlib
import re
import pandas as pd
import logging
from collections import defaultdict
from typing import List, Dict


from .neo_store import Neo4jGraphStore
from .nodes import NodeType, RichPaperNode, KeywordNode, AuthorNode, JournalNode, InstitutionNode, BaseNode
from .relationships import RelationshipType, Relationship
from .logger import setup_logger

import traceback


class GraphBuilder:
    """Builds the knowledge graph from preprocessed research data"""
    
    def __init__(self, graph_store):
        self.graph_store = graph_store
        self.logger = setup_logger()
        
        # Use dictionary to track created nodes by their custom IDs
        self.created_nodes = set()  # Track custom node IDs that have been created
        
        # Statistics tracking
        self.node_creation_stats = defaultdict(lambda: {'success': 0, 'failed': 0, 'skipped': 0})
        self.relationship_stats = defaultdict(lambda: {'success': 0, 'failed': 0})

    def _generate_paper_id(self, row) -> str:
        """Generate unique paper ID"""
        title = str(row.get('Title', '')).strip()
        year = str(row.get('Year', '')).strip()
        key = f"paper_{title}_{year}"
        return hashlib.md5(key.encode()).hexdigest()[:16]

    def _generate_author_id(self, author_name: str) -> str:
        """Generate unique author ID"""
        key = f"author_{author_name.lower().strip()}"
        return hashlib.md5(key.encode()).hexdigest()[:16]

    def _generate_institution_id(self, institution_name: str) -> str:
        """Generate unique institution ID"""
        key = f"institution_{institution_name.lower().strip()}"
        return hashlib.md5(key.encode()).hexdigest()[:16]

    def _generate_journal_id(self, journal_name: str) -> str:
        """Generate unique journal ID"""
        key = f"journal_{journal_name.lower().strip()}"
        return hashlib.md5(key.encode()).hexdigest()[:16]

    def _generate_keyword_id(self, keyword_name: str) -> str:
        """Generate unique keyword ID"""
        key = f"keyword_{keyword_name.lower().strip()}"
        return hashlib.md5(key.encode()).hexdigest()[:16]

    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare dataframe for processing"""
        self.logger.info("Cleaning dataframe...")
        
        # Handle authors
        if 'authors_list' not in df.columns and 'Authors' in df.columns:
            df['authors_list'] = df['Authors'].apply(
                lambda x: [author.strip() for author in str(x).split(',') if author.strip()] 
                if pd.notna(x) else []
            )
        
        # Handle keywords
        if 'keywords_cleaned' not in df.columns and 'Keywords' in df.columns:
            df['keywords_cleaned'] = df['Keywords'].apply(
                lambda x: [kw.strip() for kw in str(x).split(',') if kw.strip()] 
                if pd.notna(x) else []
            )
        
        return df

    def _parse_institutions(self, inst_string: str) -> List[str]:
        """Parse institution string into list of institutions"""
        if not inst_string or pd.isna(inst_string):
            return []
        
        # Split by semicolon, comma, or numbered list
        institutions = []
        parts = re.split(r'[;,]|\d+\.', str(inst_string))
        
        for part in parts:
            part = part.strip()
            if part and len(part) > 3:  # Filter out very short strings
                institutions.append(part)
        
        return institutions

    def _prepare_paper_node(self, row) -> Dict:
        """Prepare paper node data"""
        paper_id = self._generate_paper_id(row)
        
        return {
            'labels': ['PAPER'],
            'properties': {
                'id': paper_id,
                'title': str(row.get('Title', '')).strip(),
                'year': int(row.get('Year', 0)) if pd.notna(row.get('Year')) else None,
                'doi': str(row.get('DOI', '')).strip() if pd.notna(row.get('DOI')) else None,
                'main_findings': str(row.get('Main Findings', '')).strip() if pd.notna(row.get('Main Findings')) else None,
                'equations_models': str(row.get('Equations/Models', '')).strip() if pd.notna(row.get('Equations/Models')) else None,
                'application_area': str(row.get('Application Area', '')).strip() if pd.notna(row.get('Application Area')) else None,
                'strengths': str(row.get('Strengths', '')).strip() if pd.notna(row.get('Strengths')) else None,
                'limitations': str(row.get('Limitations', '')).strip() if pd.notna(row.get('Limitations')) else None,
                'journal': str(row.get('Journal/Conference', '')).strip() if pd.notna(row.get('Journal/Conference')) else None,
                'citations_count': int(row.get('Citations', 0)) if pd.notna(row.get('Citations')) else 0,
                'methodology': str(row.get('Methodology', '')).strip() if pd.notna(row.get('Methodology')) else None,
                'authors': str(row.get('Authors', '')).strip() if pd.notna(row.get('Authors')) else None,
                'keywords': str(row.get('Keywords', '')).strip() if pd.notna(row.get('Keywords')) else None,
                'institutions': str(row.get('Institution (if known)', '')).strip() if pd.notna(row.get('Institution (if known)')) else None
            }
        }

    def _create_node_if_not_exists(self, custom_id: str, node_data: Dict, node_type: str) -> bool:
        """Create a node if it doesn't exist and return success status"""
        if custom_id not in self.created_nodes:
            try:
                result = self.graph_store.insert_node(node_data)
                if result:
                    self.created_nodes.add(custom_id)
                    self.node_creation_stats[node_type]['success'] += 1
                    self.logger.debug(f"Created {node_type}: {custom_id}")
                    return True
                else:
                    self.node_creation_stats[node_type]['failed'] += 1
                    return False
            except Exception as e:
                self.logger.error(f"Failed to create {node_type} {custom_id}: {e}")
                self.node_creation_stats[node_type]['failed'] += 1
                return False
        else:
            self.node_creation_stats[node_type]['skipped'] += 1
            return True

    def _create_all_nodes(self, df: pd.DataFrame):
        """Create all nodes with proper tracking"""
        self.logger.info("Creating nodes...")
        
        # Create paper nodes
        for idx, row in df.iterrows():
            paper_id = self._generate_paper_id(row)
            node_data = self._prepare_paper_node(row)
            self._create_node_if_not_exists(paper_id, node_data, 'PAPER')
        
        # Create author nodes
        for idx, row in df.iterrows():
            authors_list = row.get('authors_list', [])
            if isinstance(authors_list, list):
                for author in authors_list:
                    if author and str(author).strip():
                        author_name = str(author).strip()
                        author_id = self._generate_author_id(author_name)
                        node_data = {
                            'labels': ['AUTHOR'],
                            'properties': {'id': author_id, 'name': author_name}
                        }
                        self._create_node_if_not_exists(author_id, node_data, 'AUTHOR')
        
        # Create institution nodes
        for idx, row in df.iterrows():
            inst_str = row.get('Institution (if known)', '')
            if inst_str and pd.notna(inst_str):
                institutions = self._parse_institutions(str(inst_str))
                for institution in institutions:
                    inst_id = self._generate_institution_id(institution)
                    node_data = {
                        'labels': ['INSTITUTION'],
                        'properties': {'id': inst_id, 'name': institution}
                    }
                    self._create_node_if_not_exists(inst_id, node_data, 'INSTITUTION')
        
        # Create journal nodes
        for idx, row in df.iterrows():
            journal = row.get('Journal/Conference', '')
            if journal and pd.notna(journal) and str(journal).strip():
                journal_name = str(journal).strip()
                journal_id = self._generate_journal_id(journal_name)
                node_data = {
                    'labels': ['JOURNAL'],
                    'properties': {'id': journal_id, 'name': journal_name}
                }
                self._create_node_if_not_exists(journal_id, node_data, 'JOURNAL')
        
        # Create keyword nodes
        for idx, row in df.iterrows():
            keywords_list = row.get('keywords_cleaned', [])
            if isinstance(keywords_list, list):
                for keyword in keywords_list:
                    if keyword and str(keyword).strip():
                        keyword_name = str(keyword).strip()
                        keyword_id = self._generate_keyword_id(keyword_name)
                        node_data = {
                            'labels': ['KEYWORD'],
                            'properties': {'id': keyword_id, 'name': keyword_name}
                        }
                        self._create_node_if_not_exists(keyword_id, node_data, 'KEYWORD')

    def _create_all_relationships(self, df: pd.DataFrame):
        """Create all relationships with detailed logging"""
        self.logger.info("Creating relationships...")
        
        for idx, row in df.iterrows():
            paper_id = self._generate_paper_id(row)
            
            if paper_id not in self.created_nodes:
                self.logger.warning(f"Skipping relationships for non-existent paper: {paper_id}")
                continue
            
            # Author relationships
            authors_list = row.get('authors_list', [])
            if isinstance(authors_list, list):
                for author in authors_list:
                    if author and str(author).strip():
                        author_name = str(author).strip()
                        author_id = self._generate_author_id(author_name)
                        self._create_relationship(paper_id, author_id, 'AUTHORED_BY')
            
            # Institution relationships
            inst_str = row.get('Institution (if known)', '')
            if inst_str and pd.notna(inst_str):
                institutions = self._parse_institutions(str(inst_str))
                for institution in institutions:
                    inst_id = self._generate_institution_id(institution)
                    self._create_relationship(paper_id, inst_id, 'AFFILIATED_WITH')
            
            # Journal relationships
            journal = row.get('Journal/Conference', '')
            if journal and pd.notna(journal) and str(journal).strip():
                journal_name = str(journal).strip()
                journal_id = self._generate_journal_id(journal_name)
                self._create_relationship(paper_id, journal_id, 'PUBLISHED_IN')
            
            # Keyword relationships
            keywords_list = row.get('keywords_cleaned', [])
            if isinstance(keywords_list, list):
                for keyword in keywords_list:
                    if keyword and str(keyword).strip():
                        keyword_name = str(keyword).strip()
                        keyword_id = self._generate_keyword_id(keyword_name)
                        self._create_relationship(paper_id, keyword_id, 'HAS_KEYWORD')
            
            ## Temporal relationships (papers published in consecutive years by same authors)
            ## # Create relationships for papers in same domain with year gap < 3
            # # This is simplified - should be enhanced with semantic similarity later
            if idx < len(df) - 1:  # Not the last row
                next_row = df.iloc[idx + 1]
                next_paper_id = self._generate_paper_id(next_row)
                
                # Check if papers share authors and are consecutive years
                current_year = row.get('Year')
                next_year = next_row.get('Year')
                
                if (pd.notna(current_year) and pd.notna(next_year) and 
                    abs(int(next_year) - int(current_year)) < 3):
                    
                    current_authors = set(row.get('authors_list', []))
                    next_authors = set(next_row.get('authors_list', []))

                    # self.logger.info("CREATING TEMPORAL_SUCCESSOR")
                    
                    if current_authors & next_authors:  # If they share any authors
                        self._create_relationship(paper_id, next_paper_id, 'TEMPORAL_SUCCESSOR')


    def _create_relationship(self, source_id: str, target_id: str, rel_type: str):
        """Create a single relationship with error handling"""
        try:
            # Check if both nodes exist in our tracking
            if source_id not in self.created_nodes:
                self.logger.warning(f"Source node {source_id} not found for {rel_type}")
                self.relationship_stats[rel_type]['failed'] += 1
                return False
            
            if target_id not in self.created_nodes:
                self.logger.warning(f"Target node {target_id} not found for {rel_type}")
                self.relationship_stats[rel_type]['failed'] += 1
                return False
            
            # Create relationship using node IDs directly
            rel_data = {
                'source_id': source_id,
                'target_id': target_id,
                'relationship_type': rel_type,
                'properties': {}
            }
            
            success = self.graph_store.insert_relationship(rel_data)
            if success:
                self.relationship_stats[rel_type]['success'] += 1
                return True
            else:
                self.relationship_stats[rel_type]['failed'] += 1
                return False
            
        except Exception as e:
            self.logger.error(f"Failed to create {rel_type} relationship {source_id} -> {target_id}: {str(e)}")
            self.relationship_stats[rel_type]['failed'] += 1
            return False

    def _log_node_creation_stats(self):
        """Log node creation statistics"""
        self.logger.info("NODE CREATION STATS:")
        for node_type, stats in self.node_creation_stats.items():
            total = stats['success'] + stats['failed'] + stats['skipped']
            self.logger.info(f"  {node_type}: {stats['success']} created, {stats['skipped']} skipped, {stats['failed']} failed (Total: {total})")

    def _log_relationship_stats(self):
        """Log relationship creation statistics"""
        self.logger.info("RELATIONSHIP CREATION STATS:")
        for rel_type, stats in self.relationship_stats.items():
            total = stats['success'] + stats['failed']
            self.logger.info(f"  {rel_type}: {stats['success']} created, {stats['failed']} failed (Total: {total})")

    # def _log_database_stats(self):
    #     """Log current database statistics"""
    #     self.logger.info("DATABASE STATS:")
        
    #     # Node counts
    #     node_counts = self.graph_store.get_node_counts()
    #     self.logger.info("NODES IN DATABASE:")
    #     for node_type, count in node_counts.items():
    #         self.logger.info(f"  {node_type}: {count}")
        
    #     # Relationship counts
    #     rel_counts = self.graph_store.get_relationship_counts()
    #     self.logger.info("RELATIONSHIPS IN DATABASE:")
    #     if any(count > 0 for count in rel_counts.values()):
    #         for rel_type, count in rel_counts.items():
    #             if count > 0:
    #                 self.logger.info(f"  {rel_type}: {count}")
        # else:
        #     self.logger.warning("  NO RELATIONSHIPS FOUND!")

    def build_graph(self, df: pd.DataFrame):
        """Main method to orchestrate graph construction"""
        self.logger.info("="*50)
        self.logger.info("STARTING GRAPH CONSTRUCTION")
        self.logger.info("="*50)
        
        try:
            # Log DataFrame info
            self.logger.info(f"DataFrame shape: {df.shape}")
            self.logger.info(f"DataFrame columns: {list(df.columns)}")
            
            # Clean DataFrame
            df = self._clean_dataframe(df)
            self.logger.info("DataFrame cleaned successfully")
            
            # Create nodes
            self.logger.info("PHASE 1: Creating nodes...")
            self._create_all_nodes(df)
            self._log_node_creation_stats()
            
            # Debug created nodes
            self.logger.info(f"Total tracked nodes: {len(self.created_nodes)}")
            
            # Create relationships
            self.logger.info("PHASE 2: Creating basic relationships...")
            self._create_all_relationships(df)
            # self._log_relationship_stats()
            
            # # Final database stats
            # self.logger.info("PHASE 2: First layer database statistics...")
            # self._log_database_stats()
            
            # self.logger.info("="*50)
            # self.logger.info("First Phase GRAPH CONSTRUCTION COMPLETED SUCCESSFULLY")
            # self.logger.info("="*50)
            
        except Exception as e:
            self.logger.error(f"Graph construction failed: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            raise

