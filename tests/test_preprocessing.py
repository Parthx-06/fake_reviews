"""Tests for the text preprocessing pipeline."""

import pytest
from app.preprocessing import TextPreprocessor


@pytest.fixture
def preprocessor():
    return TextPreprocessor()


class TestCleanText:
    """Tests for TextPreprocessor.clean_text()"""

    def test_lowercase_conversion(self, preprocessor):
        result = preprocessor.clean_text("THIS IS UPPERCASE")
        assert result == result.lower()

    def test_url_removal(self, preprocessor):
        result = preprocessor.clean_text("Check out https://example.com for more")
        assert "https" not in result
        assert "example" not in result

    def test_html_removal(self, preprocessor):
        result = preprocessor.clean_text("This is <b>bold</b> and <i>italic</i>")
        assert "<b>" not in result
        assert "<i>" not in result

    def test_special_chars_removal(self, preprocessor):
        result = preprocessor.clean_text("Price is $99.99!!!")
        assert "$" not in result
        assert "!" not in result

    def test_empty_input(self, preprocessor):
        assert preprocessor.clean_text("") == ""
        assert preprocessor.clean_text(None) == ""
        assert preprocessor.clean_text(123) == ""

    def test_stopword_removal(self, preprocessor):
        result = preprocessor.clean_text("This is a very good product")
        assert "this" not in result.split()
        # "good" should remain
        assert "good" in result

    def test_short_word_removal(self, preprocessor):
        result = preprocessor.clean_text("I am so ok with it")
        # Words with 2 or fewer characters should be removed
        tokens = result.split()
        for token in tokens:
            assert len(token) > 2


class TestExtractFeatures:
    """Tests for TextPreprocessor.extract_features()"""

    def test_text_length(self, preprocessor):
        text = "Hello world"
        features = preprocessor.extract_features(text)
        assert features['text_length'] == len(text)

    def test_word_count(self, preprocessor):
        text = "This has five words here"
        features = preprocessor.extract_features(text)
        assert features['word_count'] == 5

    def test_exclamation_count(self, preprocessor):
        text = "Wow! Amazing! Great!"
        features = preprocessor.extract_features(text)
        assert features['exclamation_count'] == 3

    def test_uppercase_ratio(self, preprocessor):
        text = "HELLO world"  # 5 uppercase out of 10 letters (excluding space)
        features = preprocessor.extract_features(text)
        assert features['uppercase_ratio'] > 0

    def test_empty_input(self, preprocessor):
        features = preprocessor.extract_features("")
        assert features['text_length'] == 0
        assert features['word_count'] == 0

    def test_none_input(self, preprocessor):
        features = preprocessor.extract_features(None)
        assert features['text_length'] == 0
