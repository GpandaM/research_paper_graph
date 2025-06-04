
def debug_graph_state(self):
    """Debug method to check the current state of the graph"""
    self.logger.info("=== GRAPH DEBUG INFO ===")
    
    # Check nodes in database
    try:
        node_counts = self.graph_store.execute("""
        MATCH (n)
        RETURN labels(n)[0] as node_type, count(n) as count
        ORDER BY node_type
        """)
        self.logger.info("Nodes in database:")
        for record in node_counts:
            self.logger.info(f"  {record['node_type']}: {record['count']}")
    except Exception as e:
        self.logger.error(f"Failed to get node counts: {str(e)}")
    
    # Check relationships in database
    try:
        rel_counts = self.graph_store.execute("""
        MATCH ()-[r]->()
        RETURN type(r) as rel_type, count(r) as count
        ORDER BY rel_type
        """)
        self.logger.info("Relationships in database:")
        for record in rel_counts:
            self.logger.info(f"  {record['rel_type']}: {record['count']}")
    except Exception as e:
        self.logger.error(f"Failed to get relationship counts: {str(e)}")
    
    # Check sample nodes
    try:
        sample_nodes = self.graph_store.execute("""
        MATCH (n)
        RETURN n.id as id, labels(n)[0] as type
        LIMIT 10
        """)
        self.logger.info("Sample node IDs:")
        for record in sample_nodes:
            self.logger.info(f"  {record['type']}: {record['id']}")
    except Exception as e:
        self.logger.error(f"Failed to get sample nodes: {str(e)}")
    
    # Internal state
    self.logger.info(f"Tracked existing nodes: {len(self.existing_nodes)}")
    self.logger.info(f"Tracked failed nodes: {len(self.failed_nodes)}")
    
    # Sample of existing nodes
    if self.existing_nodes:
        sample_existing = list(self.existing_nodes)[:10]
        self.logger.info(f"Sample existing node IDs: {sample_existing}")

def verify_node_relationships(self, df: pd.DataFrame):
    """Verify that nodes exist before trying to create relationships"""
    verification_results = {
        'papers': {'exist': 0, 'missing': 0},
        'authors': {'exist': 0, 'missing': 0},
        'institutions': {'exist': 0, 'missing': 0},
        'journals': {'exist': 0, 'missing': 0},
        'keywords': {'exist': 0, 'missing': 0}
    }
    
    for _, row in df.iterrows():
        # Check paper
        paper_id = self._generate_paper_id(row)
        if paper_id in self.existing_nodes:
            verification_results['papers']['exist'] += 1
        else:
            verification_results['papers']['missing'] += 1
        
        # Check authors
        if 'authors_list' in row and isinstance(row['authors_list'], list):
            for author in row['authors_list']:
                if author and str(author).strip():
                    author_id = self._normalize_author_id(str(author).strip())
                    if author_id in self.existing_nodes:
                        verification_results['authors']['exist'] += 1
                    else:
                        verification_results['authors']['missing'] += 1
        
        # Check institutions
        if 'Institution (if known)' in row and pd.notna(row['Institution (if known)']):
            institutions = self._parse_institutions(row['Institution (if known)'])
            for institution in institutions:
                if institution and str(institution).strip():
                    inst_id = self._normalize_institution_id(str(institution).strip())
                    if inst_id in self.existing_nodes:
                        verification_results['institutions']['exist'] += 1
                    else:
                        verification_results['institutions']['missing'] += 1
        
        # Check journals
        if 'Journal/Conference' in row and pd.notna(row['Journal/Conference']):
            journal = str(row['Journal/Conference']).strip()
            if journal:
                journal_id = self._normalize_journal_id(journal)
                if journal_id in self.existing_nodes:
                    verification_results['journals']['exist'] += 1
                else:
                    verification_results['journals']['missing'] += 1
        
        # Check keywords
        if 'keywords_cleaned' in row and isinstance(row['keywords_cleaned'], list):
            for keyword in row['keywords_cleaned']:
                if keyword and str(keyword).strip():
                    keyword_id = self._normalize_keyword_id(str(keyword).strip())
                    if keyword_id in self.existing_nodes:
                        verification_results['keywords']['exist'] += 1
                    else:
                        verification_results['keywords']['missing'] += 1
    
    self.logger.info("=== NODE VERIFICATION RESULTS ===")
    for node_type, counts in verification_results.items():
        total = counts['exist'] + counts['missing']
        if total > 0:
            success_rate = (counts['exist'] / total) * 100
            self.logger.info(f"{node_type}: {counts['exist']}/{total} exist ({success_rate:.1f}%)")

def test_single_relationship(self):
    """Test creating a single relationship to debug the process"""
    try:
        # Find two nodes that should exist
        sample_query = """
        MATCH (p:Paper), (a:Author)
        RETURN p.id as paper_id, a.id as author_id
        LIMIT 1
        """
        result = self.graph_store.execute(sample_query)
        
        if result:
            paper_id = result[0]['paper_id']
            author_id = result[0]['author_id']
            
            self.logger.info(f"Testing relationship: {paper_id} -> {author_id}")
            
            # Try to create relationship
            rel_dict = {
                'source_id': paper_id,
                'target_id': author_id,
                'relationship_type': 'AUTHORED_BY',
                'properties': {}
            }
            
            self.graph_store.insert_relationship(rel_dict)
            self.logger.info("Test relationship created successfully!")
            
            # Verify it exists
            verify_query = """
            MATCH (p:Paper {id: $paper_id})-[r:AUTHORED_BY]->(a:Author {id: $author_id})
            RETURN count(r) as count
            """
            verify_result = self.graph_store.execute(verify_query, 
                                                   {'paper_id': paper_id, 'author_id': author_id})
            
            if verify_result and verify_result[0]['count'] > 0:
                self.logger.info("Test relationship verified in database!")
            else:
                self.logger.error("Test relationship not found in database!")
                
        else:
            self.logger.error("No nodes found for relationship test")
            
    except Exception as e:
        self.logger.error(f"Test relationship failed: {str(e)}")

# Add these methods to your GraphBuilder class
GraphBuilder.debug_graph_state = debug_graph_state
GraphBuilder.verify_node_relationships = verify_node_relationships
GraphBuilder.test_single_relationship = test_single_relationship