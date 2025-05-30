from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum

class NodeType(Enum):
    PAPER = "PAPER"
    AUTHOR = "AUTHOR"
    INSTITUTION = "INSTITUTION"
    JOURNAL = "JOURNAL"
    KEYWORD = "KEYWORD"
    METHODOLOGY = "METHODOLOGY"
    APPLICATION_AREA = "APPLICATION_AREA"
    EQUATION_MODEL = "EQUATION_MODEL"

@dataclass
class BaseNode:
    id: str
    node_type: NodeType
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'type': self.node_type.value
        }

@dataclass
class EntityNode(BaseNode):
    name: str
    
    def to_dict(self) -> Dict:
        return {
            **super().to_dict(),
            'name': self.name
        }

@dataclass
class RichPaperNode(BaseNode):
    title: str
    year: int
    authors: List[str]
    institutions: List[str]
    journal: str
    keywords: List[str]
    methodology: str
    main_findings: str
    equations_models: str
    application_area: str
    strengths: str
    limitations: str
    citations_count: int
    doi: str
    
    def __post_init__(self):
        self.node_type = NodeType.PAPER
    
    def to_dict(self) -> Dict:
        return {
            **super().to_dict(),
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