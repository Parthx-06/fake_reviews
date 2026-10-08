"""Tests for the Flask API endpoints."""

import json
import pytest
from app.main import app


@pytest.fixture
def client():
    """Create a test client."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


class TestHealthEndpoint:
    """Tests for GET /health"""

    def test_health_check(self, client):
        response = client.get('/health')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['status'] == 'healthy'
        assert 'model_loaded' in data
        assert data['version'] == '1.0.0'


class TestPredictEndpoint:
    """Tests for POST /predict"""

    def test_missing_body(self, client):
        response = client.post('/predict', content_type='application/json')
        assert response.status_code == 400

    def test_missing_review_field(self, client):
        response = client.post('/predict',
            data=json.dumps({'text': 'hello'}),
            content_type='application/json'
        )
        assert response.status_code == 400

    def test_short_review(self, client):
        response = client.post('/predict',
            data=json.dumps({'review': 'short'}),
            content_type='application/json'
        )
        assert response.status_code == 400

    def test_valid_prediction(self, client):
        response = client.post('/predict',
            data=json.dumps({
                'review': 'This is a decent product with good quality and fair pricing.'
            }),
            content_type='application/json'
        )
        # Either 200 (model loaded) or 503 (model not loaded)
        assert response.status_code in [200, 503]
        data = json.loads(response.data)
        if response.status_code == 200:
            assert 'prediction' in data
            assert 'confidence' in data


class TestBatchPredictEndpoint:
    """Tests for POST /batch_predict"""

    def test_missing_reviews(self, client):
        response = client.post('/batch_predict',
            data=json.dumps({}),
            content_type='application/json'
        )
        assert response.status_code == 400

    def test_empty_reviews(self, client):
        response = client.post('/batch_predict',
            data=json.dumps({'reviews': []}),
            content_type='application/json'
        )
        assert response.status_code == 400

    def test_too_many_reviews(self, client):
        reviews = ['test review text here'] * 51
        response = client.post('/batch_predict',
            data=json.dumps({'reviews': reviews}),
            content_type='application/json'
        )
        assert response.status_code == 400


class TestIndexPage:
    """Tests for GET /"""

    def test_index_loads(self, client):
        response = client.get('/')
        assert response.status_code == 200
        assert b'Fake Review Detector' in response.data
