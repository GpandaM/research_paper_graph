from typing import List
import pandas as pd


def parse_list_field(field_value: str) -> List[str]:
    """Parse comma-separated string into list"""
    if pd.isna(field_value) or not field_value:
        return []
    
    # Handle various separators
    separators = [';', ',', '|', '\n']
    items = [field_value]
    
    for sep in separators:
        new_items = []
        for item in items:
            new_items.extend([x.strip() for x in item.split(sep) if x.strip()])
        items = new_items
    
    return [item for item in items if item and len(item) > 1]