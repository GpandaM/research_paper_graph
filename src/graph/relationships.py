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
    COLLABORATES_WITH = "COLLABORATES_WITH"
    SIMILAR_TO = "SIMILAR_TO"
    CITES = "CITES"

    # New paper-to-paper relationships
    TEMPORAL_SUCCESSOR = "TEMPORAL_SUCCESSOR"
    SAME_VENUE = "SAME_VENUE"
    RELATED_TO = "RELATED_TO"
    SAME_INSTITUTION = "SAME_INSTITUTION"


@dataclass
class Relationship:
    source_id: str
    target_id: str
    relationship_type: RelationshipType
    weight: float = 1.0
    properties: Dict = None
    
    def __post_init__(self):
        if self.properties is None:
            self.properties = {}
    
    def to_dict(self) -> Dict:
        return {
            'source_id': self.source_id,
            'target_id': self.target_id,
            'type': self.relationship_type.value,
            'weight': self.weight,
            **self.properties
        }