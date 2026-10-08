"""
ML Model Manager for Fake Review Detection.
Handles model loading, prediction, and interaction with AWS S3 for model storage.
"""

import os
import time
import joblib
import boto3
import numpy as np
from botocore.exceptions import ClientError, NoCredentialsError
from app.preprocessing import TextPreprocessor


class FakeReviewModel:
    """Manages the fake review detection ML model."""

    def __init__(self, model_path=None):
        """
        Initialize the model manager.

        Args:
            model_path (str): Path to the trained model file (.pkl).
        """
        self.model_path = model_path or os.getenv('MODEL_PATH', 'models/fake_review_model.pkl')
        self.model = None
        self.vectorizer = None
        self.preprocessor = TextPreprocessor()
        self.is_loaded = False

    def load_model(self):
        """Load the trained model and vectorizer from disk."""
        try:
            if os.path.exists(self.model_path):
                artifacts = joblib.load(self.model_path)
                self.model = artifacts['model']
                self.vectorizer = artifacts['vectorizer']
                self.is_loaded = True
                print(f"[INFO] Model loaded from {self.model_path}")
                return True
            else:
                print(f"[WARN] Model file not found at {self.model_path}")
                print("[WARN] Run 'python -m training.train' to train the model first.")
                return False
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            return False

    def load_from_s3(self, bucket_name=None, s3_key='models/fake_review_model.pkl'):
        """
        Download and load model from AWS S3.

        Args:
            bucket_name (str): S3 bucket name.
            s3_key (str): S3 object key for the model file.
        """
        bucket = bucket_name or os.getenv('S3_BUCKET_NAME')
        if not bucket:
            print("[ERROR] S3_BUCKET_NAME not set.")
            return False

        try:
            s3 = boto3.client('s3')
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            s3.download_file(bucket, s3_key, self.model_path)
            print(f"[INFO] Model downloaded from s3://{bucket}/{s3_key}")
            return self.load_model()
        except NoCredentialsError:
            print("[ERROR] AWS credentials not configured.")
            return False
        except ClientError as e:
            print(f"[ERROR] S3 download failed: {e}")
            return False

    def upload_to_s3(self, bucket_name=None, s3_key='models/fake_review_model.pkl'):
        """
        Upload trained model to AWS S3.

        Args:
            bucket_name (str): S3 bucket name.
            s3_key (str): S3 object key.
        """
        bucket = bucket_name or os.getenv('S3_BUCKET_NAME')
        if not bucket:
            print("[ERROR] S3_BUCKET_NAME not set.")
            return False

        try:
            s3 = boto3.client('s3')
            s3.upload_file(self.model_path, bucket, s3_key)
            print(f"[INFO] Model uploaded to s3://{bucket}/{s3_key}")
            return True
        except NoCredentialsError:
            print("[ERROR] AWS credentials not configured.")
            return False
        except ClientError as e:
            print(f"[ERROR] S3 upload failed: {e}")
            return False

    def predict(self, review_text):
        """
        Predict whether a review is fake or genuine.

        Args:
            review_text (str): The review text to analyze.

        Returns:
            dict: Prediction result with label, confidence, and processing time.
        """
        if not self.is_loaded:
            return {
                'error': 'Model not loaded. Train the model first with: python -m training.train',
                'prediction': None,
                'confidence': None
            }

        start_time = time.time()

        # Preprocess the text
        cleaned_text = self.preprocessor.clean_text(review_text)

        # Vectorize
        text_vector = self.vectorizer.transform([cleaned_text])

        # Predict
        prediction = self.model.predict(text_vector)[0]
        probabilities = self.model.predict_proba(text_vector)[0]

        # Get confidence for the predicted class
        confidence = float(np.max(probabilities))

        processing_time = (time.time() - start_time) * 1000  # ms

        # Map labels
        label_map = {'OR': 'FAKE', 'CG': 'GENUINE', 1: 'FAKE', 0: 'GENUINE'}
        prediction_label = label_map.get(prediction, str(prediction))

        return {
            'prediction': prediction_label,
            'confidence': round(confidence, 4),
            'processing_time_ms': round(processing_time, 2),
            'cleaned_text': cleaned_text,
            'features': self.preprocessor.extract_features(review_text)
        }
