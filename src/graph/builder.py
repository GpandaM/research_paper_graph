import re
import pandas as pd
from typing import Dict, List, Set, Tuple
from collections import defaultdict
import networkx as nx
from .nodes import KeywordNode, AuthorNode, NodeType, EntityNode, RichPaperNode
from .relationships import Relationship, RelationshipType
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



class GraphBuilder:
    """Optimized graph builder for LLM integration with consolidated information."""
    
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.nodes_registry = {}
        self.entity_counters = defaultdict(int)  # Track entity frequencies
        self.paper_lookup = {}  # Fast paper lookup by title/doi
        self.paper_features = {} 
        
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
        """Build graph with paper-to-paper relationships."""
        
        # First pass: Create all paper nodes
        for idx, row in df.iterrows():
            paper_node = self._create_rich_paper_node(idx, row)
            self.add_node(paper_node)
            
            # Store paper features for similarity calculation
            self.paper_features[paper_node.id] = {
                'authors': set(paper_node.authors),
                'keywords': set(paper_node.keywords),
                'institutions': set(paper_node.institutions),
                'year': paper_node.year,
                'journal': paper_node.journal,
                'methodology': paper_node.methodology,
                'application_area': paper_node.application_area,
                'text_content': f"{paper_node.title} {paper_node.main_findings} {paper_node.methodology}"
            }
        
        # Second pass: Create entity nodes and relationships
        for idx, row in df.iterrows():
            paper_id = f"paper_{idx}"
            self._create_entity_relationships(paper_id, row)
        
        # Third pass: Create paper-to-paper relationships
        self._build_paper_similarity_relationships()
        self._build_paper_collaboration_relationships()
        self._build_paper_citation_relationships(df)
        self._build_paper_temporal_relationships()
        self._build_paper_journal_relationships()
        
        # Fourth pass: Create other computed relationships
        self._build_collaboration_relationships()
        self._build_institutional_relationships()
        
        return self.graph
    
    def _build_paper_similarity_relationships(self):
        """Create SIMILAR_TO relationships between papers based on content similarity."""
        print("Building paper similarity relationships...")
        
        paper_ids = [pid for pid in self.paper_features.keys()]
        if len(paper_ids) < 2:
            return
        
        # Method 1: Text-based similarity using TF-IDF
        text_contents = [self.paper_features[pid]['text_content'] for pid in paper_ids]
        
        try:
            vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2
            )
            tfidf_matrix = vectorizer.fit_transform(text_contents)
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            # Create relationships for papers with high similarity
            similarity_threshold = 0.3  # Adjust as needed
            relationships_added = 0
            
            for i, paper1_id in enumerate(paper_ids):
                for j, paper2_id in enumerate(paper_ids):
                    if i < j and similarity_matrix[i][j] > similarity_threshold:
                        # Create bidirectional similarity relationship
                        rel1 = Relationship(
                            source_id=paper1_id,
                            target_id=paper2_id,
                            relationship_type=RelationshipType.SIMILAR_TO,
                            weight=float(similarity_matrix[i][j]),
                            properties={
                                'similarity_score': float(similarity_matrix[i][j]),
                                'similarity_type': 'content'
                            }
                        )
                        rel2 = Relationship(
                            source_id=paper2_id,
                            target_id=paper1_id,
                            relationship_type=RelationshipType.SIMILAR_TO,
                            weight=float(similarity_matrix[i][j]),
                            properties={
                                'similarity_score': float(similarity_matrix[i][j]),
                                'similarity_type': 'content'
                            }
                        )
                        
                        if self.add_relationship(rel1) and self.add_relationship(rel2):
                            relationships_added += 2
            
            print(f"Added {relationships_added} content similarity relationships")
            
        except Exception as e:
            print(f"Error in text similarity calculation: {e}")
        
        # Method 2: Feature-based similarity (keywords, authors, etc.)
        self._build_feature_based_similarity(paper_ids)
    
    def _build_feature_based_similarity(self, paper_ids: List[str]):
        """Build similarity based on shared features like keywords, authors."""
        relationships_added = 0
        
        for i, paper1_id in enumerate(paper_ids):
            for j, paper2_id in enumerate(paper_ids):
                if i < j:
                    paper1_features = self.paper_features[paper1_id]
                    paper2_features = self.paper_features[paper2_id]
                    
                    # Calculate different types of similarity
                    keyword_similarity = self._jaccard_similarity(
                        paper1_features['keywords'],
                        paper2_features['keywords']
                    )
                    
                    author_similarity = self._jaccard_similarity(
                        paper1_features['authors'],
                        paper2_features['authors']
                    )
                    
                    institution_similarity = self._jaccard_similarity(
                        paper1_features['institutions'],
                        paper2_features['institutions']
                    )
                    
                    # Combined similarity score
                    combined_score = (
                        keyword_similarity * 0.4 +
                        author_similarity * 0.3 +
                        institution_similarity * 0.3
                    )
                    
                    # Create relationship if similarity is high enough
                    if combined_score > 0.2:  # Threshold for feature similarity
                        rel = Relationship(
                            source_id=paper1_id,
                            target_id=paper2_id,
                            relationship_type=RelationshipType.SIMILAR_TO,
                            weight=combined_score,
                            properties={
                                'similarity_score': combined_score,
                                'similarity_type': 'features',
                                'keyword_similarity': keyword_similarity,
                                'author_similarity': author_similarity,
                                'institution_similarity': institution_similarity
                            }
                        )
                        
                        if self.add_relationship(rel):
                            relationships_added += 1
        
        print(f"Added {relationships_added} feature-based similarity relationships")
    
    def _build_paper_collaboration_relationships(self):
        """Create COLLABORATES_WITH relationships between papers sharing authors."""
        print("Building paper collaboration relationships...")
        
        # Group papers by shared authors
        author_papers = defaultdict(set)
        
        for paper_id, features in self.paper_features.items():
            for author in features['authors']:
                if author:  # Skip empty authors
                    author_papers[author].add(paper_id)
        
        relationships_added = 0
        
        # Create collaboration relationships between papers with shared authors
        for author, papers in author_papers.items():
            if len(papers) > 1:
                papers_list = list(papers)
                for i, paper1_id in enumerate(papers_list):
                    for j, paper2_id in enumerate(papers_list):
                        if i < j:
                            rel = Relationship(
                                source_id=paper1_id,
                                target_id=paper2_id,
                                relationship_type=RelationshipType.COLLABORATES_WITH,
                                weight=1.0,
                                properties={
                                    'shared_author': author,
                                    'collaboration_type': 'author'
                                }
                            )
                            
                            if self.add_relationship(rel):
                                relationships_added += 1
        
        print(f"Added {relationships_added} paper collaboration relationships")
    
    def _build_paper_citation_relationships(self, df: pd.DataFrame):
        """Build citation relationships if citation data is available."""
        print("Building paper citation relationships...")
        
        # This would need actual citation data
        # For now, we'll create hypothetical citations based on year and similarity
        relationships_added = 0
        
        paper_years = {}
        for idx, row in df.iterrows():
            paper_id = f"paper_{idx}"
            year = self._safe_int_conversion(row.get('Year'), 0)
            if year > 0:
                paper_years[paper_id] = year
        
        # Create citation relationships (newer papers cite older ones)
        for paper1_id, year1 in paper_years.items():
            for paper2_id, year2 in paper_years.items():
                if paper1_id != paper2_id and year1 > year2:
                    # Check if papers are similar enough to warrant citation
                    if paper1_id in self.paper_features and paper2_id in self.paper_features:
                        features1 = self.paper_features[paper1_id]
                        features2 = self.paper_features[paper2_id]
                        
                        keyword_similarity = self._jaccard_similarity(
                            features1['keywords'],
                            features2['keywords']
                        )
                        
                        # Create citation if there's topical similarity and reasonable time gap
                        if keyword_similarity > 0.3 and (year1 - year2) <= 10:
                            rel = Relationship(
                                source_id=paper1_id,
                                target_id=paper2_id,
                                relationship_type=RelationshipType.CITES,
                                weight=1.0,
                                properties={
                                    'year_gap': year1 - year2,
                                    'topical_similarity': keyword_similarity
                                }
                            )
                            
                            if self.add_relationship(rel):
                                relationships_added += 1
        
        print(f"Added {relationships_added} citation relationships")
    
    def _build_paper_temporal_relationships(self):
        """Build temporal relationships between papers in the same field."""
        print("Building temporal relationships...")
        
        # Group papers by application area and year
        area_year_papers = defaultdict(lambda: defaultdict(list))
        
        for paper_id, features in self.paper_features.items():
            year = features['year']
            app_area = features['application_area']
            if year > 0 and app_area:
                area_year_papers[app_area][year].append(paper_id)
        
        relationships_added = 0
        
        # Create temporal relationships within the same application area
        for app_area, year_papers in area_year_papers.items():
            years = sorted(year_papers.keys())
            
            for i, year1 in enumerate(years):
                for year2 in years[i+1:i+3]:  # Connect to next 2 years only
                    if year2 - year1 <= 3:  # Within 3 years
                        for paper1_id in year_papers[year1]:
                            for paper2_id in year_papers[year2]:
                                rel = Relationship(
                                    source_id=paper1_id,
                                    target_id=paper2_id,
                                    relationship_type=RelationshipType.TEMPORAL_SUCCESSOR,
                                    weight=1.0 / (year2 - year1),  # Closer years have higher weight
                                    properties={
                                        'year_gap': year2 - year1,
                                        'application_area': app_area
                                    }
                                )
                                
                                if self.add_relationship(rel):
                                    relationships_added += 1
        
        print(f"Added {relationships_added} temporal relationships")
    
    def _build_paper_journal_relationships(self):
        """Build relationships between papers in the same journal/conference."""
        print("Building journal-based relationships...")
        
        # Group papers by journal
        journal_papers = defaultdict(list)
        
        for paper_id, features in self.paper_features.items():
            journal = features['journal']
            if journal and journal.strip():
                journal_papers[journal].append(paper_id)
        
        relationships_added = 0
        
        # Create relationships between papers in the same journal
        for journal, papers in journal_papers.items():
            if len(papers) > 1:
                for i, paper1_id in enumerate(papers):
                    for j, paper2_id in enumerate(papers):
                        if i < j:
                            rel = Relationship(
                                source_id=paper1_id,
                                target_id=paper2_id,
                                relationship_type=RelationshipType.SAME_VENUE,
                                weight=1.0,
                                properties={
                                    'journal': journal,
                                    'relationship_type': 'same_venue'
                                }
                            )
                            
                            if self.add_relationship(rel):
                                relationships_added += 1
        
        print(f"Added {relationships_added} journal-based relationships")
    
    def _jaccard_similarity(self, set1: Set, set2: Set) -> float:
        """Calculate Jaccard similarity between two sets."""
        if not set1 and not set2:
            return 1.0
        if not set1 or not set2:
            return 0.0
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    def _safe_int_conversion(self, value, default=0):
        """Safely convert value to int."""
        if pd.isna(value):
            return default
        try:
            if isinstance(value, str):
                value = value.strip()
                if value.lower() in ['-', '', 'n/a', 'na', 'none', 'null']:
                    return default
            return int(float(value))
        except (ValueError, TypeError):
            return default


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
            node_type=NodeType.PAPER,
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
        """Optimized similarity relationship builder using inverted index"""
        keyword_papers = defaultdict(set)
        paper_keywords = {}
        
        # Build inverted index
        for node_id, node_data in self.graph.nodes(data=True):
            if node_data.get('type') == 'paper':
                keywords = set(node_data.get('keywords', []))
                paper_keywords[node_id] = keywords
                for kw in keywords:
                    keyword_papers[kw].add(node_id)
        
        # Find similar papers
        similar_pairs = defaultdict(int)
        for kw, papers in keyword_papers.items():
            papers = list(papers)
            for i in range(len(papers)):
                for j in range(i+1, len(papers)):
                    pair = tuple(sorted([papers[i], papers[j]]))
                    similar_pairs[pair] += 1
        
        # Create relationships
        for (paper1, paper2), intersection in similar_pairs.items():
            union = len(paper_keywords[paper1] | paper_keywords[paper2])
            similarity = intersection / union
            
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
