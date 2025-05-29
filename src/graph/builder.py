import re
import pandas as pd
from typing import Dict, List, Set, Tuple
from collections import defaultdict
import networkx as nx
from .nodes import KeywordNode, AuthorNode, NodeType, EntityNode, RichPaperNode
from .relationships import Relationship, RelationshipType



class GraphBuilder:
    """Optimized graph builder for LLM integration with consolidated information."""
    
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.nodes_registry = {}
        self.entity_counters = defaultdict(int)  # Track entity frequencies
        self.paper_lookup = {}  # Fast paper lookup by title/doi
        
    def add_node(self, node) -> None:
        """Add a node to the graph with duplicate prevention."""
        if node.id not in self.nodes_registry:
            self.nodes_registry[node.id] = node
            self.graph.add_node(node.id, **node.to_dict())
    
    def add_relationship(self, relationship: Relationship) -> None:
        """Add a weighted relationship to the graph."""
        self.graph.add_edge(
            relationship.source_id,
            relationship.target_id,
            key=relationship.relationship_type.value,
            **relationship.to_dict()
        )
    
    def build_from_dataframe(self, df: pd.DataFrame) -> nx.MultiDiGraph:
        """Build optimized graph from DataFrame with all fields."""
        
        # First pass: Create all paper nodes with rich information
        for idx, row in df.iterrows():
            paper_node = self._create_rich_paper_node(idx, row)
            self.add_node(paper_node)
            
            # Build lookup mappings
            self.paper_lookup[row['Title'].lower().strip()] = paper_node.id
            if pd.notna(row.get('DOI/Link')):
                self.paper_lookup[row['DOI/Link'].lower().strip()] = paper_node.id
        
        # Second pass: Create entity nodes and relationships
        for idx, row in df.iterrows():
            paper_id = f"paper_{idx}"
            self._create_entity_relationships(paper_id, row)
        
        # Third pass: Create computed relationships
        self._build_similarity_relationships()
        self._build_collaboration_relationships()
        self._build_institutional_relationships()
        
        return self.graph
    
    def _create_rich_paper_node(self, idx: int, row: pd.Series) -> RichPaperNode:
        """Create a rich paper node with all consolidated information."""
        
        def safe_int_conversion(value, default=0):
            """Safely convert value to int, handling non-numeric strings."""
            if pd.isna(value):
                return default
            try:
                # Handle string values that might be '-', 'N/A', etc.
                if isinstance(value, str):
                    value = value.strip()
                    if value in ['-', '', 'N/A', 'n/a', 'NA']:
                        return default
                return int(float(value))  # Use float first to handle decimal strings
            except (ValueError, TypeError):
                return default
        
        return RichPaperNode(
            id=f"paper_{idx}",
            title=str(row.get('Title', '')),
            year=safe_int_conversion(row.get('Year'), 0),
            authors=self._parse_authors(row.get('Author(s)', '')),
            institutions=self._parse_institutions(row.get('Institution (if known)', '')),
            journal=str(row.get('Journal/Conference', '')),
            keywords=self._parse_keywords(row.get('Keywords', '')),
            methodology=str(row.get('Methodology', '')),
            main_findings=str(row.get('Main Findings', '')),
            equations_models=str(row.get('Equations/Models', '')),
            application_area=str(row.get('Application Area', '')),
            strengths=str(row.get('Strengths', '')),
            limitations=str(row.get('Limitations', '')),
            citations_count=safe_int_conversion(row.get('Citations (est.)'), 0),
            doi=str(row.get('DOI/Link', ''))
        )
    
    def _create_entity_relationships(self, paper_id: str, row: pd.Series) -> None:
        """Create entity nodes and their relationships to papers."""
        
        # Author nodes and relationships
        authors = self._parse_authors(row.get('Author(s)', ''))
        for author in authors:
            author_node = EntityNode(
                id=f"author_{self._clean_id(author)}",
                name=author,
                node_type=NodeType.AUTHOR
            )
            self.add_node(author_node)
            self.entity_counters[author_node.id] += 1
            
            rel = Relationship(
                source_id=paper_id,
                target_id=author_node.id,
                relationship_type=RelationshipType.AUTHORED_BY
            )
            self.add_relationship(rel)
        
        # Institution nodes and relationships
        institutions = self._parse_institutions(row.get('Institution (if known)', ''))
        for institution in institutions:
            institution_node = EntityNode(
                id=f"institution_{self._clean_id(institution)}",
                name=institution,
                node_type=NodeType.INSTITUTION
            )
            self.add_node(institution_node)
            self.entity_counters[institution_node.id] += 1
            
            rel = Relationship(
                source_id=paper_id,
                target_id=institution_node.id,
                relationship_type=RelationshipType.AFFILIATED_WITH
            )
            self.add_relationship(rel)
        
        # Journal node and relationship
        if pd.notna(row.get('Journal/Conference')):
            journal_node = EntityNode(
                id=f"journal_{self._clean_id(row['Journal/Conference'])}",
                name=row['Journal/Conference'],
                node_type=NodeType.JOURNAL
            )
            self.add_node(journal_node)
            self.entity_counters[journal_node.id] += 1
            
            rel = Relationship(
                source_id=paper_id,
                target_id=journal_node.id,
                relationship_type=RelationshipType.PUBLISHED_IN
            )
            self.add_relationship(rel)
        
        # Keyword nodes and relationships
        keywords = self._parse_keywords(row.get('Keywords', ''))
        for keyword in keywords:
            keyword_node = EntityNode(
                id=f"keyword_{self._clean_id(keyword)}",
                name=keyword,
                node_type=NodeType.KEYWORD
            )
            self.add_node(keyword_node)
            self.entity_counters[keyword_node.id] += 1
            
            rel = Relationship(
                source_id=paper_id,
                target_id=keyword_node.id,
                relationship_type=RelationshipType.HAS_KEYWORD
            )
            self.add_relationship(rel)
        
        # Methodology nodes and relationships
        methodologies = self._parse_methodologies(row.get('Methodology', ''))
        for methodology in methodologies:
            methodology_node = EntityNode(
                id=f"methodology_{self._clean_id(methodology)}",
                name=methodology,
                node_type=NodeType.METHODOLOGY
            )
            self.add_node(methodology_node)
            self.entity_counters[methodology_node.id] += 1
            
            rel = Relationship(
                source_id=paper_id,
                target_id=methodology_node.id,
                relationship_type=RelationshipType.USES_METHODOLOGY
            )
            self.add_relationship(rel)
        
        # Application area nodes and relationships
        app_areas = self._parse_application_areas(row.get('Application Area', ''))
        for app_area in app_areas:
            app_area_node = EntityNode(
                id=f"app_area_{self._clean_id(app_area)}",
                name=app_area,
                node_type=NodeType.APPLICATION_AREA
            )
            self.add_node(app_area_node)
            self.entity_counters[app_area_node.id] += 1
            
            rel = Relationship(
                source_id=paper_id,
                target_id=app_area_node.id,
                relationship_type=RelationshipType.APPLIED_IN
            )
            self.add_relationship(rel)
        
        # Equation/model nodes and relationships
        equations = self._parse_equations_models(row.get('Equations/Models', ''))
        for equation in equations:
            equation_node = EntityNode(
                id=f"equation_{self._clean_id(equation)}",
                name=equation,
                node_type=NodeType.EQUATION_MODEL
            )
            self.add_node(equation_node)
            self.entity_counters[equation_node.id] += 1
            
            rel = Relationship(
                source_id=paper_id,
                target_id=equation_node.id,
                relationship_type=RelationshipType.USES_EQUATION
            )
            self.add_relationship(rel)
    
    def _parse_authors(self, authors_str: str) -> List[str]:
        """Parse author string into individual author names."""
        if not authors_str or pd.isna(authors_str):
            return []
        
        # Handle different author formats
        authors = []
        for separator in [',', ';', ' and ', ' & ']:
            if separator in authors_str:
                authors = [author.strip() for author in authors_str.split(separator)]
                break
        
        if not authors:
            authors = [authors_str.strip()]
        
        return [author for author in authors if author]
    
    def _parse_institutions(self, institutions_str: str) -> List[str]:
        """Parse institution string into individual institutions."""
        if not institutions_str or pd.isna(institutions_str):
            return []
        
        # Split by numbers (1., 2., etc.) or semicolons
        institutions = []
        
        # Remove numbering and split
        cleaned = re.sub(r'\d+\.\s*', '|', institutions_str)
        institutions = [inst.strip() for inst in cleaned.split('|') if inst.strip()]
        
        if not institutions:
            # Fallback to semicolon split
            institutions = [inst.strip() for inst in institutions_str.split(';') if inst.strip()]
        
        return institutions
    
    def _parse_keywords(self, keywords_str: str) -> List[str]:
        """Parse keywords string into individual keywords."""
        if not keywords_str or pd.isna(keywords_str):
            return []
        
        keywords = [kw.strip() for kw in keywords_str.split(',')]
        return [kw for kw in keywords if kw]
    
    def _parse_methodologies(self, methodology_str: str) -> List[str]:
        """Parse methodology string into individual methodologies."""
        if not methodology_str or pd.isna(methodology_str):
            return []
        
        # Split by common separators
        methodologies = []
        for separator in [';', ',', ' + ', ' and ', '–', '-']:
            if separator in methodology_str:
                methodologies = [m.strip() for m in methodology_str.split(separator)]
                break
        
        if not methodologies:
            methodologies = [methodology_str.strip()]
        
        return [m for m in methodologies if m and len(m) > 2]  # Filter very short strings
    
    def _parse_application_areas(self, app_area_str: str) -> List[str]:
        """Parse application area string into individual areas."""
        if not app_area_str or pd.isna(app_area_str):
            return []
        
        # Split by common separators
        areas = []
        for separator in [',', ';', ' and ', ' & ', '(', ')']:
            if separator in app_area_str:
                parts = app_area_str.split(separator)
                areas = [area.strip() for area in parts if area.strip()]
                break
        
        if not areas:
            areas = [app_area_str.strip()]
        
        return [area for area in areas if area and len(area) > 3]
    
    def _parse_equations_models(self, equations_str: str) -> List[str]:
        """Parse equations/models string into individual equations."""
        if not equations_str or pd.isna(equations_str):
            return []
        
        # Split by semicolons or common separators
        equations = []
        for separator in [';', ',']:
            if separator in equations_str:
                equations = [eq.strip() for eq in equations_str.split(separator)]
                break
        
        if not equations:
            equations = [equations_str.strip()]
        
        return [eq for eq in equations if eq and len(eq) > 2]
    
    def _clean_id(self, text: str) -> str:
        """Clean text for use as node ID."""
        if not text:
            return ""
        
        # Convert to lowercase and replace special characters
        cleaned = re.sub(r'[^\w\s-]', '', text.lower())
        cleaned = re.sub(r'[-\s]+', '_', cleaned)
        return cleaned.strip('_')
    
    def _build_similarity_relationships(self, threshold: float = 0.3) -> None:
        """Build similarity relationships between papers based on shared keywords."""
        paper_keywords = defaultdict(set)
        
        # Collect keywords for each paper
        for node_id, node_data in self.graph.nodes(data=True):
            if node_data.get('type') == 'paper':
                paper_keywords[node_id] = set(node_data.get('keywords', []))
        
        # Calculate similarities and create relationships
        papers = list(paper_keywords.keys())
        for i, paper1 in enumerate(papers):
            for paper2 in papers[i+1:]:
                keywords1 = paper_keywords[paper1]
                keywords2 = paper_keywords[paper2]
                
                if keywords1 and keywords2:
                    # Jaccard similarity
                    intersection = len(keywords1.intersection(keywords2))
                    union = len(keywords1.union(keywords2))
                    similarity = intersection / union if union > 0 else 0
                    
                    if similarity >= threshold:
                        rel = Relationship(
                            source_id=paper1,
                            target_id=paper2,
                            relationship_type=RelationshipType.SIMILAR_TO,
                            properties={'similarity_score': similarity},
                            weight=similarity
                        )
                        self.add_relationship(rel)
    
    def _build_collaboration_relationships(self) -> None:
        """Build collaboration relationships between authors."""
        author_papers = defaultdict(set)
        
        # Collect papers for each author
        for edge in self.graph.edges(data=True):
            if edge[2].get('type') == 'AUTHORED_BY':
                paper_id = edge[0]
                author_id = edge[1]
                author_papers[author_id].add(paper_id)
        
        # Create collaboration edges between authors
        authors = list(author_papers.keys())
        for i, author1 in enumerate(authors):
            for author2 in authors[i+1:]:
                shared_papers = author_papers[author1].intersection(author_papers[author2])
                if shared_papers:
                    collaboration_strength = len(shared_papers)
                    rel = Relationship(
                        source_id=author1,
                        target_id=author2,
                        relationship_type=RelationshipType.COLLABORATES_WITH,
                        properties={'shared_papers': list(shared_papers)},
                        weight=collaboration_strength
                    )
                    self.add_relationship(rel)
    
    def _build_institutional_relationships(self) -> None:
        """Build relationships between institutions based on collaborations."""
        institution_authors = defaultdict(set)
        
        # Collect authors for each institution
        for edge in self.graph.edges(data=True):
            if edge[2].get('type') == 'AFFILIATED_WITH':
                paper_id = edge[0]
                institution_id = edge[1]
                
                # Find authors of this paper
                for author_edge in self.graph.edges(paper_id, data=True):
                    if author_edge[2].get('type') == 'AUTHORED_BY':
                        institution_authors[institution_id].add(author_edge[1])
        
        # Create institutional collaboration edges
        institutions = list(institution_authors.keys())
        for i, inst1 in enumerate(institutions):
            for inst2 in institutions[i+1:]:
                shared_authors = institution_authors[inst1].intersection(institution_authors[inst2])
                if shared_authors:
                    collaboration_strength = len(shared_authors)
                    rel = Relationship(
                        source_id=inst1,
                        target_id=inst2,
                        relationship_type=RelationshipType.SAME_INSTITUTION,
                        properties={'shared_authors': list(shared_authors)},
                        weight=collaboration_strength
                    )
                    self.add_relationship(rel)
    
    def get_node_importance_scores(self) -> Dict[str, float]:
        """Calculate importance scores for all nodes based on centrality and frequency."""
        importance_scores = {}
        
        # Calculate different centrality measures
        try:
            degree_centrality = nx.degree_centrality(self.graph)
            betweenness_centrality = nx.betweenness_centrality(self.graph)
            
            for node_id in self.graph.nodes():
                # Combine different factors for importance
                degree_score = degree_centrality.get(node_id, 0)
                betweenness_score = betweenness_centrality.get(node_id, 0)
                frequency_score = self.entity_counters.get(node_id, 1) / max(self.entity_counters.values()) if self.entity_counters else 0
                
                # Weighted combination
                importance_scores[node_id] = (
                    0.4 * degree_score + 
                    0.3 * betweenness_score + 
                    0.3 * frequency_score
                )
        except:
            # Fallback to simple degree-based scoring
            for node_id in self.graph.nodes():
                importance_scores[node_id] = self.graph.degree(node_id) / max(dict(self.graph.degree()).values()) if self.graph.degree() else 0
        
        return importance_scores

'''
class GraphBuilder:
    """Builds knowledge graph from research paper data."""
    
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.nodes_registry = {}
        self.relationships = []
        
    def add_node(self, node) -> None:
        """Add a node to the graph."""
        if node.id not in self.nodes_registry:
            self.nodes_registry[node.id] = node
            self.graph.add_node(node.id, **node.to_dict())
    
    def add_relationship(self, relationship: Relationship) -> None:
        """Add a relationship to the graph."""
        self.relationships.append(relationship)
        self.graph.add_edge(
            relationship.source_id,
            relationship.target_id,
            **relationship.to_dict()
        )
    
    def build_from_dataframe(self, df: pd.DataFrame) -> nx.MultiDiGraph:
        """Build complete graph from preprocessed DataFrame."""
        
        
        for idx, row in df.iterrows():
            
            # Create paper nodes
            paper_node = PaperNode(
                paper_id=f"paper_{idx}",
                title=row['Title'],
                year=row['Year'],
                journal=row.get('Journal/Conference'),
                doi=row.get('DOI/Link'),
                methodology=row.get('Methodology'),
                main_findings=row.get('Main Findings'),
                citations_count=row.get('citations_count', 0)
            )
            self.add_node(paper_node)
            
            # Create keyword nodes and relationships
            if 'keywords_cleaned' in row and row['keywords_cleaned']:
                for keyword in row['keywords_cleaned']:
                    keyword_node = KeywordNode(keyword)
                    self.add_node(keyword_node)
                    
                    # Create HAS_KEYWORD relationship
                    rel = Relationship(
                        source_id=paper_node.id,
                        target_id=keyword_node.id,
                        relationship_type=RelationshipType.HAS_KEYWORD,
                        properties={}
                    )
                    self.add_relationship(rel)
            
            # Create author nodes and relationships
            if 'authors_list' in row and row['authors_list']:
                for author_name in row['authors_list']:
                    author_node = AuthorNode(author_name)
                    self.add_node(author_node)
                    
                    # Create AUTHORED_BY relationship
                    rel = Relationship(
                        source_id=paper_node.id,
                        target_id=author_node.id,
                        relationship_type=RelationshipType.AUTHORED_BY,
                        properties={}
                    )
                    self.add_relationship(rel)
        
        # Build keyword similarity relationships
        self._build_keyword_similarities()
        
        return self.graph
    
    def _build_keyword_similarities(self, threshold: float = 0.3) -> None:
        """Build similarity relationships between papers based on shared keywords."""
        paper_keywords = defaultdict(set)
        
        # Collect keywords for each paper
        for node_id, node_data in self.graph.nodes(data=True):
            if node_data['type'] == 'paper':
                paper_id = node_id
                # Find connected keywords
                for neighbor in self.graph.neighbors(node_id):
                    neighbor_data = self.graph.nodes[neighbor]
                    if neighbor_data['type'] == 'keyword':
                        paper_keywords[paper_id].add(neighbor_data['keyword'])
        
        # Calculate similarities and create relationships
        papers = list(paper_keywords.keys())
        for i, paper1 in enumerate(papers):
            for paper2 in papers[i+1:]:
                keywords1 = paper_keywords[paper1]
                keywords2 = paper_keywords[paper2]
                
                if keywords1 and keywords2:
                    # Jaccard similarity
                    intersection = len(keywords1.intersection(keywords2))
                    union = len(keywords1.union(keywords2))
                    similarity = intersection / union if union > 0 else 0
                    
                    if similarity >= threshold:
                        rel = Relationship(
                            source_id=paper1,
                            target_id=paper2,
                            relationship_type=RelationshipType.SIMILAR_TO,
                            properties={'similarity_score': similarity},
                            weight=similarity
                        )
                        self.add_relationship(rel)
'''