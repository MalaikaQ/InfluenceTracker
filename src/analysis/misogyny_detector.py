"""
Misogyny detection using both lexicon-based and machine learning approaches.
"""
import pandas as pd
import numpy as np
import re
from typing import List, Dict, Tuple, Set
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import pickle
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from utils.config import MISOGYNY_DETECTION, LEXICONS_DIR, PROCESSED_DATA_DIR
from utils.text_processing import TextProcessor, create_misogyny_lexicon

class MisogynyDetector:
    """Hybrid misogyny detection using lexicon and ML approaches."""
    
    def __init__(self):
        """Initialize the detector."""
        self.text_processor = TextProcessor()
        self.misogyny_lexicon = create_misogyny_lexicon()
        self.vectorizer = None
        self.ml_classifier = None
        self.is_trained = False
        
        # Load configuration
        self.confidence_threshold = MISOGYNY_DETECTION['confidence_threshold']
        self.lexicon_weight = MISOGYNY_DETECTION['lexicon_weight']
        self.ml_weight = MISOGYNY_DETECTION['ml_weight']
    
    def _lexicon_score(self, text: str) -> float:
        """
        Calculate misogyny score based on lexicon matching.
        
        Args:
            text: Input text
            
        Returns:
            Lexicon-based score (0-1)
        """
        if not isinstance(text, str):
            return 0.0
        
        # Clean text for matching
        cleaned_text = self.text_processor.clean_text(
            text, 
            remove_stopwords=False,  # Keep all words for lexicon matching
            remove_punctuation=False
        ).lower()
        
        # Count misogynistic terms
        misogyny_count = 0
        total_words = len(cleaned_text.split())
        
        if total_words == 0:
            return 0.0
        
        # Direct term matching
        for term in self.misogyny_lexicon:
            if term in cleaned_text:
                # Weight longer phrases more heavily
                weight = len(term.split())
                misogyny_count += weight
        
        # Additional pattern matching
        patterns = [
            r'\bwomen are\b.*\b(stupid|inferior|objects|property)\b',
            r'\bfemales?\b.*\b(hypergamous|sluts?|whores?)\b',
            r'\ball women\b.*\b(lie|cheat|manipulate)\b',
            r'\btypical\b.*\bfemale\b.*\bbehavior\b'
        ]
        
        for pattern in patterns:
            matches = len(re.findall(pattern, cleaned_text, re.IGNORECASE))
            misogyny_count += matches * 2  # Weight patterns heavily
        
        # Normalize by text length
        score = min(misogyny_count / total_words, 1.0)
        return score
    
    def _extract_features(self, texts: List[str]) -> np.ndarray:
        """
        Extract features for ML classification.
        
        Args:
            texts: List of texts
            
        Returns:
            Feature matrix
        """
        # Clean texts
        cleaned_texts = [
            self.text_processor.clean_text(
                text, 
                remove_stopwords=True,
                remove_punctuation=True,
                lowercase=True
            ) for text in texts
        ]
        
        # TF-IDF features
        if self.vectorizer is None:
            self.vectorizer = TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 3),
                min_df=2,
                max_df=0.95
            )
            tfidf_features = self.vectorizer.fit_transform(cleaned_texts)
        else:
            tfidf_features = self.vectorizer.transform(cleaned_texts)
        
        # Additional features
        additional_features = []
        for text in texts:
            features = self.text_processor.extract_features(text)
            lexicon_score = self._lexicon_score(text)
            
            feature_vector = [
                features.get('length', 0),
                features.get('word_count', 0),
                features.get('caps_ratio', 0),
                features.get('exclamation_count', 0),
                features.get('avg_word_length', 0),
                lexicon_score
            ]
            additional_features.append(feature_vector)
        
        # Combine TF-IDF with additional features
        additional_features = np.array(additional_features)
        combined_features = np.hstack([tfidf_features.toarray(), additional_features])
        
        return combined_features
    
    def train_classifier(self, 
                        training_data: pd.DataFrame,
                        text_column: str = 'text',
                        label_column: str = 'is_misogynistic') -> Dict:
        """
        Train the ML classifier.
        
        Args:
            training_data: DataFrame with text and labels
            text_column: Name of text column
            label_column: Name of label column (0/1)
            
        Returns:
            Training results dictionary
        """
        texts = training_data[text_column].tolist()
        labels = training_data[label_column].tolist()
        
        # Extract features
        features = self._extract_features(texts)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Train classifier
        self.ml_classifier = LogisticRegression(random_state=42, max_iter=1000)
        self.ml_classifier.fit(X_train, y_train)
        
        # Evaluate
        train_score = self.ml_classifier.score(X_train, y_train)
        test_score = self.ml_classifier.score(X_test, y_test)
        
        # Cross-validation
        cv_scores = cross_val_score(self.ml_classifier, features, labels, cv=5)
        
        # Predictions for detailed evaluation
        y_pred = self.ml_classifier.predict(X_test)
        
        self.is_trained = True
        
        return {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
    
    def predict_misogyny(self, texts: List[str]) -> pd.DataFrame:
        """
        Predict misogyny for a list of texts.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            DataFrame with predictions and scores
        """
        results = []
        
        for text in texts:
            lexicon_score = self._lexicon_score(text)
            
            if self.is_trained and self.ml_classifier:
                # ML prediction
                features = self._extract_features([text])
                ml_proba = self.ml_classifier.predict_proba(features)[0][1]  # Probability of positive class
                
                # Combined score
                combined_score = (
                    self.lexicon_weight * lexicon_score +
                    self.ml_weight * ml_proba
                )
            else:
                ml_proba = 0.0
                combined_score = lexicon_score
            
            # Final prediction
            is_misogynistic = combined_score >= self.confidence_threshold
            
            results.append({
                'text': text,
                'lexicon_score': lexicon_score,
                'ml_score': ml_proba,
                'combined_score': combined_score,
                'is_misogynistic': is_misogynistic,
                'confidence': combined_score
            })
        
        return pd.DataFrame(results)
    
    def analyze_dataset(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """
        Analyze an entire dataset for misogyny.
        
        Args:
            df: Input DataFrame
            text_column: Name of text column
            
        Returns:
            DataFrame with misogyny analysis results
        """
        texts = df[text_column].fillna('').tolist()
        predictions = self.predict_misogyny(texts)
        
        # Merge with original data
        result_df = df.copy()
        for col in predictions.columns:
            if col != 'text':
                result_df[col] = predictions[col].values
        
        return result_df
    
    def get_misogynistic_patterns(self, df: pd.DataFrame, 
                                 text_column: str = 'text',
                                 top_n: int = 20) -> Dict:
        """
        Identify common patterns in misogynistic text.
        
        Args:
            df: DataFrame with misogyny analysis
            text_column: Name of text column
            top_n: Number of top patterns to return
            
        Returns:
            Dictionary with pattern analysis
        """
        misogynistic_texts = df[df['is_misogynistic']][text_column]
        
        # Most common terms in misogynistic text
        all_misogynistic_text = ' '.join(misogynistic_texts.fillna(''))
        
        # Clean and tokenize
        cleaned_text = self.text_processor.clean_text(
            all_misogynistic_text,
            remove_stopwords=True,
            remove_punctuation=True
        )
        
        words = cleaned_text.split()
        word_freq = pd.Series(words).value_counts().head(top_n)
        
        # Most common lexicon terms found
        lexicon_found = {}
        for text in misogynistic_texts:
            for term in self.misogyny_lexicon:
                if term in text.lower():
                    lexicon_found[term] = lexicon_found.get(term, 0) + 1
        
        lexicon_freq = pd.Series(lexicon_found).sort_values(ascending=False).head(top_n)
        
        return {
            'most_common_words': word_freq.to_dict(),
            'most_common_lexicon_terms': lexicon_freq.to_dict(),
            'total_misogynistic_comments': len(misogynistic_texts),
            'misogyny_rate': len(misogynistic_texts) / len(df)
        }
    
    def save_model(self, filename: str = 'misogyny_detector'):
        """Save the trained model."""
        if not self.is_trained:
            print("No trained model to save")
            return
        
        model_data = {
            'vectorizer': self.vectorizer,
            'classifier': self.ml_classifier,
            'lexicon': self.misogyny_lexicon,
            'config': {
                'confidence_threshold': self.confidence_threshold,
                'lexicon_weight': self.lexicon_weight,
                'ml_weight': self.ml_weight
            }
        }
        
        filepath = PROCESSED_DATA_DIR / f"{filename}.pkl"
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filename: str = 'misogyny_detector'):
        """Load a previously saved model."""
        filepath = PROCESSED_DATA_DIR / f"{filename}.pkl"
        
        if not filepath.exists():
            print(f"Model file {filepath} not found")
            return
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.vectorizer = model_data['vectorizer']
        self.ml_classifier = model_data['classifier']
        self.misogyny_lexicon = model_data['lexicon']
        
        config = model_data['config']
        self.confidence_threshold = config['confidence_threshold']
        self.lexicon_weight = config['lexicon_weight']
        self.ml_weight = config['ml_weight']
        
        self.is_trained = True
        print(f"Model loaded from {filepath}")

def create_synthetic_training_data() -> pd.DataFrame:
    """
    Create synthetic training data for demonstration.
    In practice, you would use labeled datasets like:
    - Davidson et al. (2017) Hate Speech Dataset
    - Founta et al. (2018) Twitter dataset
    - Custom annotated data
    """
    # Positive examples (misogynistic)
    positive_examples = [
        "women are only good for one thing",
        "typical female behavior, always lying",
        "she's just another attention seeking whore",
        "females these days are all about hypergamy",
        "women hit the wall at 30",
        "all women are the same, they only want chad",
        "she belongs in the kitchen",
        "women can't drive or think logically",
        "femoids always go for the bad boys",
        "awalt - all women are like that"
    ]
    
    # Negative examples (not misogynistic)
    negative_examples = [
        "I think women deserve equal rights and opportunities",
        "my girlfriend is really smart and funny",
        "women in STEM fields are doing amazing work",
        "I have a lot of respect for my female colleagues",
        "gender equality is important for society",
        "women's contributions to science are valuable",
        "my mom is my role model",
        "female artists create beautiful work",
        "women's rights are human rights",
        "diversity makes teams stronger"
    ]
    
    # Create DataFrame
    data = []
    for text in positive_examples:
        data.append({'text': text, 'is_misogynistic': 1})
    for text in negative_examples:
        data.append({'text': text, 'is_misogynistic': 0})
    
    return pd.DataFrame(data)

def main():
    """Example usage of MisogynyDetector."""
    # Initialize detector
    detector = MisogynyDetector()
    
    # Create training data (in practice, use real labeled data)
    training_data = create_synthetic_training_data()
    print(f"Created training data with {len(training_data)} examples")
    
    # Train classifier
    results = detector.train_classifier(training_data)
    print("Training Results:")
    print(f"Test Accuracy: {results['test_accuracy']:.3f}")
    print(f"CV Mean: {results['cv_mean']:.3f} (+/- {results['cv_std']*2:.3f})")
    
    # Test on sample texts
    test_texts = [
        "women are inferior to men in every way",
        "I love spending time with my female friends",
        "typical female logic right there",
        "women make great leaders and scientists"
    ]
    
    predictions = detector.predict_misogyny(test_texts)
    print("\nPredictions:")
    for _, row in predictions.iterrows():
        print(f"Text: {row['text'][:50]}...")
        print(f"Misogynistic: {row['is_misogynistic']} (confidence: {row['confidence']:.3f})")
        print()
    
    # Save model
    detector.save_model('example_misogyny_detector')

if __name__ == "__main__":
    main()
