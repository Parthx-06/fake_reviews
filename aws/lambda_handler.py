"""
AWS Lambda Handler for Fake Review Detection.
Serves predictions via API Gateway.
"""

import os
import json
import sys

# Add project root to path for Lambda layers
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '/opt/python')

import joblib
import boto3
from app.preprocessing import TextPreprocessor
import numpy as np


# Global model cache (persists across warm Lambda invocations)
MODEL_CACHE = {
    'model': None,
    'vectorizer': None,
    'preprocessor': TextPreprocessor()
}


def load_model_from_s3():
    """Download and load model from S3 (cached across invocations)."""
    if MODEL_CACHE['model'] is not None:
        return True

    bucket = os.environ.get('S3_BUCKET_NAME')
    model_key = os.environ.get('MODEL_S3_KEY', 'models/fake_review_model.pkl')

    if not bucket:
        print("[ERROR] S3_BUCKET_NAME environment variable not set")
        return False

    try:
        s3 = boto3.client('s3')
        local_path = '/tmp/fake_review_model.pkl'
        s3.download_file(bucket, model_key, local_path)

        artifacts = joblib.load(local_path)
        MODEL_CACHE['model'] = artifacts['model']
        MODEL_CACHE['vectorizer'] = artifacts['vectorizer']
        print(f"[INFO] Model loaded from s3://{bucket}/{model_key}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to load model from S3: {e}")
        return False


def lambda_handler(event, context):
    """
    AWS Lambda entry point.

    Supports API Gateway proxy integration.

    Args:
        event: API Gateway event.
        context: Lambda context.

    Returns:
        dict: API Gateway response with prediction.
    """
    # CORS headers
    headers = {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'POST, GET, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type'
    }

    # Handle CORS preflight
    if event.get('httpMethod') == 'OPTIONS':
        return {
            'statusCode': 200,
            'headers': headers,
            'body': ''
        }

    # Health check
    if event.get('httpMethod') == 'GET' and event.get('path', '').endswith('/health'):
        return {
            'statusCode': 200,
            'headers': headers,
            'body': json.dumps({
                'status': 'healthy',
                'model_loaded': MODEL_CACHE['model'] is not None,
                'version': '1.0.0'
            })
        }

    # Prediction
    if event.get('httpMethod') == 'POST':
        # Load model if not cached
        if not load_model_from_s3():
            return {
                'statusCode': 503,
                'headers': headers,
                'body': json.dumps({'error': 'Model not available'})
            }

        try:
            body = json.loads(event.get('body', '{}'))
            review_text = body.get('review', '').strip()

            if len(review_text) < 10:
                return {
                    'statusCode': 400,
                    'headers': headers,
                    'body': json.dumps({'error': 'Review text too short (min 10 chars)'})
                }

            # Preprocess and predict
            preprocessor = MODEL_CACHE['preprocessor']
            cleaned = preprocessor.clean_text(review_text)
            vector = MODEL_CACHE['vectorizer'].transform([cleaned])
            prediction = MODEL_CACHE['model'].predict(vector)[0]
            probabilities = MODEL_CACHE['model'].predict_proba(vector)[0]
            confidence = float(np.max(probabilities))

            label_map = {1: 'FAKE', 0: 'GENUINE'}
            result = {
                'prediction': label_map.get(prediction, str(prediction)),
                'confidence': round(confidence, 4),
                'features': preprocessor.extract_features(review_text)
            }

            return {
                'statusCode': 200,
                'headers': headers,
                'body': json.dumps(result)
            }

        except json.JSONDecodeError:
            return {
                'statusCode': 400,
                'headers': headers,
                'body': json.dumps({'error': 'Invalid JSON body'})
            }
        except Exception as e:
            return {
                'statusCode': 500,
                'headers': headers,
                'body': json.dumps({'error': f'Prediction failed: {str(e)}'})
            }

    # Method not allowed
    return {
        'statusCode': 405,
        'headers': headers,
        'body': json.dumps({'error': 'Method not allowed'})
    }
