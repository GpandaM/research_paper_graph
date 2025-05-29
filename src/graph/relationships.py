from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from enum import Enum

class RelationshipType(Enum):
    AUTHORED_BY = "AUTHORED_BY"
    AFFILIATED_WITH = "AFFILIATED_WITH"
    PUBLISHED_IN = "PUBLISHED_IN"
    HAS_KEYWORD = "HAS_KEYWORD"
    USES_METHODOLOGY = "USES_METHODOLOGY"
    APPLIED_IN = "APPLIED_IN"
    USES_EQUATION = "USES_EQUATION"
    CITES = "CITES"
    SIMILAR_TO = "SIMILAR_TO"
    COLLABORATES_WITH = "COLLABORATES_WITH"
    SAME_INSTITUTION = "SAME_INSTITUTION"

@dataclass
class Relationship:
    """Represents a relationship between two nodes."""
    
    source_id: str
    target_id: str
    relationship_type: RelationshipType
    properties: Dict[str, Any] = field(default_factory=dict)  # Add default empty dict
    weight: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'source': self.source_id,
            'target': self.target_id,
            'type': self.relationship_type.value,
            'weight': self.weight,
            'properties': self.properties
        }
