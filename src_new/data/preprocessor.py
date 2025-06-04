import re
from typing import List, Set
import pandas as pd

class DataPreprocessor:
    """Preprocesses research paper data for graph construction."""
    
    def __init__(self):
        self.keyword_separators = [',', ';', '|', '\n']
        
    def clean_keywords(self, keywords_str: str) -> List[str]:
        """Extract and clean keywords from string."""
        if pd.isna(keywords_str):
            return []
        
        # Split by common separators
        for sep in self.keyword_separators:
            keywords_str = keywords_str.replace(sep, ',')
        
        keywords = [kw.strip().lower() for kw in keywords_str.split(',')]
        return [kw for kw in keywords if kw and len(kw) > 1]
    
    def extract_authors(self, authors_str: str) -> List[str]:
        if pd.isna(authors_str):
            return []

        # Normalize all whitespace/newlines
        authors_str = re.sub(r'\s+', ' ', authors_str).strip()

        # Split by semicolon OR newline
        raw_authors = re.split(r'[;\n]+', authors_str)

        cleaned_authors = []
        for author in raw_authors:
            author = author.strip()
            if not author:
                continue
            author = re.sub(r'^(Dr\.|Prof\.|Mr\.|Ms\.|Mrs\.)\s*', '', author)
            cleaned_authors.append(author)

        return cleaned_authors


    
    def estimate_citations(self, citations_str: str) -> int:
        """Convert citation string to integer."""
        if pd.isna(citations_str):
            return 0
        
        try:
            return int(citations_str)
        except (ValueError, TypeError):
            return 0
    
    def preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Complete preprocessing of the research paper DataFrame."""
        df_processed = df.copy()
        
        # Clean and standardize columns
        df_processed['keywords_cleaned'] = df_processed['Keywords'].apply(self.clean_keywords)
        df_processed['authors_list'] = df_processed['Author(s)'].apply(self.extract_authors)
        df_processed['citations_count'] = df_processed['Citations (est.)'].apply(self.estimate_citations)
        
        # Fill missing values
        df_processed['Year'] = df_processed['Year'].fillna(0).astype(int)
        df_processed['Title'] = df_processed['Title'].fillna('Unknown Title')
        
        return df_processed