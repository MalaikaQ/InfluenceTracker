"""
Text processing utilities for misogyny detection.
"""
import os
import re
import string
import nltk
from typing import List, Dict, Set
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class TextProcessor:
    """Text processing utilities for social media content."""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        # Compile regex patterns for efficiency
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.mention_pattern = re.compile(r'@\w+')
        self.hashtag_pattern = re.compile(r'#\w+')
        self.number_pattern = re.compile(r'\d+')
        self.whitespace_pattern = re.compile(r'\s+')
        
    def clean_text(self, text: str, 
                   remove_urls: bool = True,
                   remove_mentions: bool = True,
                   remove_hashtags: bool = False,
                   remove_numbers: bool = True,
                   lowercase: bool = True,
                   remove_punctuation: bool = True,
                   remove_stopwords: bool = False,
                   lemmatize: bool = False) -> str:
        """
        Clean and preprocess text data.
        
        Args:
            text: Input text to clean
            remove_urls: Remove URLs
            remove_mentions: Remove @mentions
            remove_hashtags: Remove #hashtags
            remove_numbers: Remove numbers
            lowercase: Convert to lowercase
            remove_punctuation: Remove punctuation
            remove_stopwords: Remove English stopwords
            lemmatize: Apply lemmatization
            
        Returns:
            Cleaned text string
        """
        if not isinstance(text, str):
            return ""
        
        # Remove URLs
        if remove_urls:
            text = self.url_pattern.sub('', text)
            
        # Remove mentions
        if remove_mentions:
            text = self.mention_pattern.sub('', text)
            
        # Remove hashtags (but keep the text if remove_hashtags is False)
        if remove_hashtags:
            text = self.hashtag_pattern.sub('', text)
        else:
            text = re.sub(r'#(\w+)', r'\1', text)  # Remove # but keep word
            
        # Remove numbers
        if remove_numbers:
            text = self.number_pattern.sub('', text)
            
        # Convert to lowercase
        if lowercase:
            text = text.lower()
            
        # Remove punctuation
        if remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))
            
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        if remove_stopwords:
            tokens = [token for token in tokens if token not in self.stop_words]
            
        # Lemmatize
        if lemmatize:
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
            
        # Join tokens back to string
        text = ' '.join(tokens)
        
        # Clean up whitespace
        text = self.whitespace_pattern.sub(' ', text).strip()
        
        return text
    
    def extract_features(self, text: str) -> Dict[str, float]:
        """
        Extract basic text features for analysis.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of text features
        """
        if not isinstance(text, str):
            return {}
            
        # Basic metrics
        features = {
            'length': len(text),
            'word_count': len(text.split()),
            'sentence_count': len(re.split(r'[.!?]+', text)),
            'avg_word_length': sum(len(word) for word in text.split()) / max(len(text.split()), 1),
            'caps_ratio': sum(1 for c in text if c.isupper()) / max(len(text), 1),
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),
            'url_count': len(self.url_pattern.findall(text)),
            'mention_count': len(self.mention_pattern.findall(text)),
            'hashtag_count': len(self.hashtag_pattern.findall(text))
        }
        
        return features
    
    def batch_process(self, texts: List[str], **kwargs) -> List[str]:
        """
        Process a batch of texts.
        
        Args:
            texts: List of text strings
            **kwargs: Arguments to pass to clean_text
            
        Returns:
            List of processed texts
        """
        return [self.clean_text(text, **kwargs) for text in texts]
    
    def filter_by_length(self, df: pd.DataFrame, 
                        text_column: str,
                        min_length: int = 10, 
                        max_length: int = 1000) -> pd.DataFrame:
        """
        Filter DataFrame by text length.
        
        Args:
            df: Input DataFrame
            text_column: Name of text column
            min_length: Minimum text length
            max_length: Maximum text length
            
        Returns:
            Filtered DataFrame
        """
        df = df.copy()
        df['text_length'] = df[text_column].str.len()
        return df[(df['text_length'] >= min_length) & (df['text_length'] <= max_length)]
    
    def remove_duplicates(self, df: pd.DataFrame, 
                         text_column: str,
                         similarity_threshold: float = 0.9) -> pd.DataFrame:
        """
        Remove near-duplicate texts based on Jaccard similarity.
        
        Args:
            df: Input DataFrame
            text_column: Name of text column
            similarity_threshold: Similarity threshold for removal
            
        Returns:
            DataFrame with duplicates removed
        """
        def jaccard_similarity(text1: str, text2: str) -> float:
            """Calculate Jaccard similarity between two texts."""
            set1 = set(text1.lower().split())
            set2 = set(text2.lower().split())
            intersection = len(set1.intersection(set2))
            union = len(set1.union(set2))
            return intersection / union if union > 0 else 0
        
        df = df.copy()
        to_remove = set()
        
        for i in range(len(df)):
            if i in to_remove:
                continue
            for j in range(i + 1, len(df)):
                if j in to_remove:
                    continue
                similarity = jaccard_similarity(df.iloc[i][text_column], df.iloc[j][text_column])
                if similarity > similarity_threshold:
                    to_remove.add(j)
        
        return df.drop(df.index[list(to_remove)])

def create_misogyny_lexicon() -> Set[str]:
    """
    Create a lexicon of misogynistic terms and phrases.
    
    Returns:
        Set of misogynistic terms
    """
    # Base misogynistic terms (this is a starter set - you should expand this)
    base_terms = {
        # Derogatory terms for women
        'bitch', 'slut', 'whore', 'cunt', 'skank', 'hoe', 'thot',
        
        # Red-pill/manosphere specific terms
        'femoid', 'foid', 'roastie', 'becky', 'stacy', 'hypergamy',
        'cock carousel', 'alpha widow', 'beta bux', 'riding the cock carousel',
        
        # Incel terminology
        'blackpill', 'looksmaxing', 'chad', 'virgin shaming', 
        'heightpill', 'gymcel', 'mentalcel',
        
        # MGTOW terms
        'awalt', 'gynocentrism', 'simp', 'white knight', 'beta male',
        'pussy pass', 'false accusation',
        
        # General misogynistic concepts
        'women are', 'females are', 'all women', 'typical female',
        'attention whore', 'gold digger', 'daddy issues'
    }
    
    # Add variations and common misspellings
    expanded_terms = set(base_terms)
    
    # Add plural forms
    for term in base_terms:
        if not term.endswith('s'):
            expanded_terms.add(term + 's')
    
    return expanded_terms

def load_hate_speech_lexicon(file_path: str = None) -> Set[str]:
    """
    Load additional hate speech terms from external sources.
    
    Args:
        file_path: Path to external lexicon file
        
    Returns:
        Set of hate speech terms
    """
    # This could load from Hatebase.org or other research lexicons
    # For now, return empty set - implement based on available resources
    if file_path and os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return set(line.strip().lower() for line in f)
    return set()

if __name__ == "__main__":
    # Example usage
    processor = TextProcessor()
    
    sample_text = "This is a @user sample text with #hashtags and URLs https://example.com!"
    cleaned = processor.clean_text(sample_text)
    print(f"Original: {sample_text}")
    print(f"Cleaned: {cleaned}")
    
    features = processor.extract_features(sample_text)
    print(f"Features: {features}")
    
    lexicon = create_misogyny_lexicon()
    print(f"Misogyny lexicon size: {len(lexicon)}")
