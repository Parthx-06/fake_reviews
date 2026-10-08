"""Tests for the ML model."""

import os
import pytest
from app.model import FakeReviewModel


@pytest.fixture
def model():
    return FakeReviewModel()


class TestModelLoading:
    """Tests for model loading behavior."""

    def test_model_init(self, model):
        assert model.is_loaded is False
        assert model.model is None
        assert model.vectorizer is None

    def test_load_missing_model(self, model):
        model.model_path = 'nonexistent/path/model.pkl'
        result = model.load_model()
        assert result is False
        assert model.is_loaded is False

    def test_predict_without_model(self, model):
        result = model.predict("This is a test review")
        assert 'error' in result
        assert result['prediction'] is None


class TestModelPrediction:
    """Tests for model predictions (requires trained model)."""

    @pytest.fixture
    def trained_model(self):
        """Load model if available, skip if not."""
        m = FakeReviewModel()
        if not m.load_model():
            pytest.skip("Model not trained. Run 'python -m training.train' first.")
        return m

    def test_predict_returns_dict(self, trained_model):
        result = trained_model.predict("This is a great product, I love it!")
        assert isinstance(result, dict)
        assert 'prediction' in result
        assert 'confidence' in result

    def test_prediction_label(self, trained_model):
        result = trained_model.predict("Good quality laptop for the price.")
        assert result['prediction'] in ['FAKE', 'GENUINE']

    def test_confidence_range(self, trained_model):
        result = trained_model.predict("Decent product, works as expected.")
        assert 0 <= result['confidence'] <= 1

    def test_processing_time(self, trained_model):
        result = trained_model.predict("Testing the processing time.")
        assert result['processing_time_ms'] >= 0

    def test_features_included(self, trained_model):
        result = trained_model.predict("This review has some features to extract!")
        assert 'features' in result
        assert 'word_count' in result['features']

    def test_obvious_fake_review(self, trained_model):
        fake = "OMG BEST PRODUCT EVER!!! BUY IT NOW!!! AMAZING!!! INCREDIBLE!!! MUST HAVE!!! LIFE CHANGING!!!"
        result = trained_model.predict(fake)
        # We expect high confidence for obvious cases
        assert result['confidence'] > 0.5

    def test_obvious_genuine_review(self, trained_model):
        genuine = (
            "I've been using this laptop for about three months now. "
            "The battery life is decent, lasting around 6-7 hours with normal use. "
            "The keyboard feels comfortable for long typing sessions. "
            "One downside is the screen could be brighter for outdoor use."
        )
        result = trained_model.predict(genuine)
        assert result['confidence'] > 0.5
