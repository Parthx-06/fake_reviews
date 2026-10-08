"""
Text Preprocessing Pipeline for Fake Review Detection.
Handles cleaning, tokenization, and feature extraction from review text.
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download required NLTK data (only needed once)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)


class TextPreprocessor:
    """Preprocesses review text for ML model input."""

    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        # Keep some negation words as they carry sentiment meaning
        self.stop_words -= {'not', 'no', 'nor', 'never', 'neither', 'nobody',
                            'nothing', 'nowhere', 'hardly', 'scarcely', 'barely'}

    def clean_text(self, text):
        """
        Clean and normalize review text.

        Args:
            text (str): Raw review text.

        Returns:
            str: Cleaned and preprocessed text.
        """
        if not isinstance(text, str):
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)

        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)

        # Remove special characters and numbers (keep letters and spaces)
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Tokenize
        tokens = word_tokenize(text)

        # Remove stopwords and lemmatize
        tokens = [
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if token not in self.stop_words and len(token) > 2
        ]

        return ' '.join(tokens)

    def extract_features(self, text):
        """
        Extract additional features from review text.

        Args:
            text (str): Raw review text.

        Returns:
            dict: Dictionary of extracted features.
        """
        if not isinstance(text, str):
            text = ""

        features = {
            'text_length': len(text),
            'word_count': len(text.split()),
            'avg_word_length': 0,
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),
            'uppercase_ratio': 0,
            'digit_count': sum(c.isdigit() for c in text),
            'punctuation_ratio': 0,
        }

        words = text.split()
        if words:
            features['avg_word_length'] = sum(len(w) for w in words) / len(words)

        if len(text) > 0:
            features['uppercase_ratio'] = sum(1 for c in text if c.isupper()) / len(text)
            features['punctuation_ratio'] = sum(
                1 for c in text if c in string.punctuation
            ) / len(text)

        return features
