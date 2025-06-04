from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum
import re

class NodeType(Enum):
    PAPER = "PAPER"
    AUTHOR = "AUTHOR"
    INSTITUTION = "INSTITUTION"
    JOURNAL = "JOURNAL"
    KEYWORD = "KEYWORD"
    METHODOLOGY = "METHODOLOGY"
    APPLICATION_AREA = "APPLICATION_AREA"
    EQUATION_MODEL = "EQUATION_MODEL"

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_


@dataclass
class BaseNode:
    id: str
    node_type: NodeType
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'type': self.node_type.value,
            **self.properties
        }

@dataclass
class RichPaperNode(BaseNode):
    """Comprehensive representation of a research paper"""
    title: str
    year: int
    authors: List[str]
    keywords: List[str]
    institutions: List[str]
    
    journal: Optional[str] = None
    methodology: Optional[str] = None
    main_findings: Optional[str] = None
    equations_models: Optional[str] = None
    application_area: Optional[str] = None
    strengths: Optional[str] = None
    limitations: Optional[str] = None
    citations_count: Optional[int] = None
    doi: Optional[str] = None
    abstract: Optional[str] = None
    publication_date: Optional[str] = None
    references: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        self.node_type = NodeType.PAPER
        self.properties = {
            'title': self.title,
            'year': self.year,
            'authors': self.authors,
            'keywords': self.keywords,
            'institutions': self.institutions,
            'journal': self.journal,
            'methodology': self.methodology,
            'main_findings': self.main_findings,
            'equations_models': self.equations_models,
            'application_area': self.application_area,
            'strengths': self.strengths,
            'limitations': self.limitations,
            'citations_count': self.citations_count,
            'doi': self.doi,
            'abstract': self.abstract,
            'publication_date': self.publication_date,
            'references': self.references
        }

@dataclass
class KeywordNode(BaseNode):
    """Representation of a research keyword/concept"""
    keyword: str
    display_name: Optional[str] = None
    frequency: Optional[int] = None
    normalized_form: Optional[str] = None
    category: Optional[str] = None
    
    def __post_init__(self):
        self.node_type = NodeType.KEYWORD
        if self.display_name is None:
            self.display_name = self.keyword
        if self.normalized_form is None:
            self.normalized_form = re.sub(r'\W+', '_', self.keyword).lower()
            
        self.properties = {
            'keyword': self.keyword,
            'display_name': self.display_name,
            'frequency': self.frequency,
            'normalized_form': self.normalized_form,
            'category': self.category
        }

@dataclass
class AuthorNode(BaseNode):
    """Representation of a research author"""
    name: str
    normalized_name: Optional[str] = None
    institution: Optional[str] = None
    orcid: Optional[str] = None
    h_index: Optional[int] = None
    publications_count: Optional[int] = None
    
    def __post_init__(self):
        self.node_type = NodeType.AUTHOR
        if self.normalized_name is None:
            self.normalized_name = re.sub(r'[^a-zA-Z0-9]', '', self.name).lower()
            
        self.properties = {
            'name': self.name,
            'normalized_name': self.normalized_name,
            'institution': self.institution,
            'orcid': self.orcid,
            'h_index': self.h_index,
            'publications_count': self.publications_count
        }

@dataclass
class InstitutionNode(BaseNode):
    """Representation of a research institution"""
    name: str
    normalized_name: Optional[str] = None
    country: Optional[str] = None
    type: Optional[str] = None  # e.g., "University", "Company", "Research Lab"
    rank: Optional[int] = None
    
    def __post_init__(self):
        self.node_type = NodeType.INSTITUTION
        if self.normalized_name is None:
            self.normalized_name = re.sub(r'[^a-zA-Z0-9]', '', self.name).lower()
            
        self.properties = {
            'name': self.name,
            'normalized_name': self.normalized_name,
            'country': self.country,
            'type': self.type,
            'rank': self.rank
        }

@dataclass
class JournalNode(BaseNode):
    """Representation of a journal/conference"""
    name: str
    normalized_name: Optional[str] = None
    issn: Optional[str] = None
    impact_factor: Optional[float] = None
    publisher: Optional[str] = None
    is_conference: bool = False
    
    def __post_init__(self):
        self.node_type = NodeType.JOURNAL
        if self.normalized_name is None:
            self.normalized_name = re.sub(r'[^a-zA-Z0-9]', '', self.name).lower()
            
        self.properties = {
            'name': self.name,
            'normalized_name': self.normalized_name,
            'issn': self.issn,
            'impact_factor': self.impact_factor,
            'publisher': self.publisher,
            'is_conference': self.is_conference
        }