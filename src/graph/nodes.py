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
    THEME = "THEME"
    CONTRIBUTION = "CONTRIBUTION"
    GAP = "GAP"
    CLUSTER = "CLUSTER"