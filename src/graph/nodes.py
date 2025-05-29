from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum

class NodeType(Enum):
    PAPER = "paper"
    AUTHOR = "author"
    KEYWORD = "keyword"
    INSTITUTION = "institution"
    JOURNAL = "journal"
    METHODOLOGY = "methodology"
    APPLICATION_AREA = "application_area"
    EQUATION_MODEL = "equation_model"

@dataclass
class BaseNode(ABC):
    """Base class for all graph nodes."""
    id: str
    type: NodeType
    properties: Dict[str, Any]
    
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary representation."""
        pass

@dataclass
class RichPaperNode:
    """Rich paper node with all information consolidated for efficient LLM queries."""
    id: str
    title: str
    year: int
    authors: List[str] = field(default_factory=list)
    institutions: List[str] = field(default_factory=list)
    journal: str = ""
    keywords: List[str] = field(default_factory=list)
    methodology: str = ""
    main_findings: str = ""
    equations_models: str = ""
    application_area: str = ""
    strengths: str = ""
    limitations: str = ""
    citations_count: int = 0
    doi: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for graph storage."""
        return {
            'id': self.id,
            'type': NodeType.PAPER.value,
            'title': self.title,
            'year': self.year,
            'authors': self.authors,
            'institutions': self.institutions,
            'journal': self.journal,
            'keywords': self.keywords,
            'methodology': self.methodology,
            'main_findings': self.main_findings,
            'equations_models': self.equations_models,
            'application_area': self.application_area,
            'strengths': self.strengths,
            'limitations': self.limitations,
            'citations_count': self.citations_count,
            'doi': self.doi
        }

@dataclass
class EntityNode:
    """Generic entity node for authors, institutions, etc."""
    id: str
    name: str
    node_type: NodeType
    properties: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'type': self.node_type.value,
            'name': self.name,
            **self.properties
        }



@dataclass
class KeywordNode(BaseNode):
    """Represents a keyword node."""
    
    def __init__(self, keyword: str, **kwargs):
        super().__init__(
            id=f"keyword_{keyword.lower().replace(' ', '_')}",
            type=NodeType.KEYWORD,
            properties={
                'keyword': keyword.lower(),
                'display_name': keyword,
                **kwargs
            }
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'type': self.type.value,
            'keyword': self.properties['keyword'],
            'display_name': self.properties['display_name'],
            'metadata': self.properties
        }

@dataclass
class AuthorNode(BaseNode):
    """Represents an author node."""
    
    def __init__(self, author_name: str, **kwargs):
        super().__init__(
            id=f"author_{author_name.lower().replace(' ', '_')}",
            type=NodeType.AUTHOR,
            properties={
                'name': author_name,
                **kwargs
            }
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'type': self.type.value,
            'name': self.properties['name'],
            'metadata': self.properties
        }