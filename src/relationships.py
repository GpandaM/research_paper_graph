from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from enum import Enum

class RelationshipType(Enum):
    AUTHORED_BY = "AUTHORED_BY"
    HAS_KEYWORD = "HAS_KEYWORD"
    USES_METHODOLOGY = "USES_METHODOLOGY"
    APPLIED_IN = "APPLIED_IN"
    USES_EQUATION = "USES_EQUATION"
    SIMILAR_TO = "SIMILAR_TO"
    CITES = "CITES"
    TEMPORAL_SUCCESSOR = "TEMPORAL_SUCCESSOR"
    SAME_VENUE = "SAME_VENUE"
    RELATED_TO = "RELATED_TO"
    BELONGS_TO_THEME = "BELONGS_TO_THEME"
    SHARES_AUTHOR = "SHARES_AUTHOR"
    SHARES_INSTITUTION = "SHARES_INSTITUTION"
    PUBLISHED_IN = "PUBLISHED_IN"
    AFFILIATED_WITH = "AFFILIATED_WITH"
    SEMANTIC_SIMILAR = "SEMANTIC_SIMILAR"
    ADDRESSES_LIMITATION = "ADDRESSES_LIMITATION"

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_

@dataclass
class Relationship:
    """Representation of a relationship between nodes"""
    source_id: str
    target_id: str
    relationship_type: RelationshipType
    weight: float = 1.0
    explanation: Optional[str] = None
    confidence: float = 1.0
    evidence: List[str] = field(default_factory=list)
    properties: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'source_id': self.source_id,
            'target_id': self.target_id,
            'relationship_type': self.relationship_type.value,
            'weight': self.weight,
            'explanation': self.explanation,
            'confidence': self.confidence,
            'evidence': self.evidence,
            'properties': self.properties
        }