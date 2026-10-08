"""
Flask Application — Fake Review Detection API.
Serves the ML model via REST endpoints and a web UI.
"""

import os
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from app.model import FakeReviewModel

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Initialize model
model = FakeReviewModel()

# Try loading model on startup
model_loaded = model.load_model()
if not model_loaded:
    print("=" * 60)
    print("  MODEL NOT FOUND!")
    print("  Run: python -m training.train")
    print("  The app will start but predictions won't work until")
    print("  the model is trained.")
    print("=" * 60)


@app.route('/')
def index():
    """Serve the web UI."""
    return render_template('index.html')


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint for monitoring and load balancers."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model.is_loaded,
        'version': '1.0.0'
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict whether a review is fake or genuine.

    Request Body:
        {"review": "This product is amazing!"}

    Returns:
        JSON with prediction, confidence, and processing details.
    """
    try:
        data = request.get_json(silent=True)

        if not data or 'review' not in data:
            return jsonify({
                'error': 'Missing "review" field in request body.',
                'usage': 'POST /predict with {"review": "your review text here"}'
            }), 400

        review_text = data['review'].strip()

        if len(review_text) < 10:
            return jsonify({
                'error': 'Review text too short. Provide at least 10 characters.'
            }), 400

        result = model.predict(review_text)

        if 'error' in result and result['prediction'] is None:
            return jsonify(result), 503

        return jsonify(result)

    except Exception as e:
        return jsonify({
            'error': f'Prediction failed: {str(e)}'
        }), 500


@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """
    Predict multiple reviews at once.

    Request Body:
        {"reviews": ["review 1", "review 2", ...]}

    Returns:
        JSON array of predictions.
    """
    try:
        data = request.get_json(silent=True)

        if not data or 'reviews' not in data:
            return jsonify({
                'error': 'Missing "reviews" field in request body.'
            }), 400

        reviews = data['reviews']

        if not isinstance(reviews, list) or len(reviews) == 0:
            return jsonify({
                'error': '"reviews" must be a non-empty list.'
            }), 400

        if len(reviews) > 50:
            return jsonify({
                'error': 'Maximum 50 reviews per batch request.'
            }), 400

        results = [model.predict(review) for review in reviews]

        return jsonify({
            'results': results,
            'total': len(results)
        })

    except Exception as e:
        return jsonify({
            'error': f'Batch prediction failed: {str(e)}'
        }), 500


if __name__ == '__main__':
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', '0') == '1'

    print(f"\n[*] Starting Fake Review Detection API on {host}:{port}")
    print(f"[*] Model loaded: {model.is_loaded}")
    print(f"[*] Open http://localhost:{port} in your browser\n")

    app.run(host=host, port=port, debug=debug)
