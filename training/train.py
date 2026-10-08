"""
Model Training Script for Fake Review Detection.
Trains a TF-IDF + Logistic Regression pipeline on review data.
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score
)

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.preprocessing import TextPreprocessor


def load_data(data_path='data/sample_reviews.csv'):
    """
    Load and validate the review dataset.

    Args:
        data_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    if not os.path.exists(data_path):
        print(f"[ERROR] Dataset not found at: {data_path}")
        print("[INFO]  Place your dataset in data/ or use the sample data.")
        print("[INFO]  Download from: https://www.kaggle.com/datasets/lievgarcia/amazon-reviews")
        sys.exit(1)

    df = pd.read_csv(data_path)
    print(f"[INFO] Loaded dataset: {len(df)} rows")
    print(f"[INFO] Columns: {list(df.columns)}")

    return df


def preprocess_data(df):
    """
    Preprocess the dataset for training.

    Args:
        df (pd.DataFrame): Raw dataset.

    Returns:
        tuple: (X_texts, y_labels)
    """
    preprocessor = TextPreprocessor()

    # Detect column names (handle different dataset formats)
    text_col = None
    label_col = None

    for col in df.columns:
        col_lower = col.lower().strip()
        if col_lower in ['text_', 'text', 'review', 'review_text', 'reviewtext', 'content']:
            text_col = col
        if col_lower in ['label', 'labels', 'fake', 'is_fake', 'class', 'target', 'sentiment']:
            label_col = col

    if text_col is None:
        # Fall back to first text-like column
        text_col = df.columns[0]
        print(f"[WARN] Using '{text_col}' as text column (auto-detected)")

    if label_col is None:
        label_col = df.columns[-1]
        print(f"[WARN] Using '{label_col}' as label column (auto-detected)")

    print(f"[INFO] Text column: '{text_col}'")
    print(f"[INFO] Label column: '{label_col}'")
    print(f"[INFO] Label distribution:\n{df[label_col].value_counts()}")

    # Drop rows with missing values
    df = df.dropna(subset=[text_col, label_col])

    # Preprocess text
    print("[INFO] Preprocessing text...")
    df['cleaned_text'] = df[text_col].apply(preprocessor.clean_text)

    # Encode labels: map to binary (0 = genuine, 1 = fake)
    unique_labels = df[label_col].unique()
    print(f"[INFO] Unique labels: {unique_labels}")

    # Handle different label formats
    label_mapping = {}
    for label in unique_labels:
        label_str = str(label).lower().strip()
        if label_str in ['or', 'fake', '1', 'deceptive', 'spam', 'yes', 'true']:
            label_mapping[label] = 1  # Fake
        else:
            label_mapping[label] = 0  # Genuine

    df['binary_label'] = df[label_col].map(label_mapping)
    print(f"[INFO] Label mapping: {label_mapping}")

    return df['cleaned_text'].tolist(), np.array(df['binary_label'].tolist(), dtype=int)


def train_model(X_texts, y_labels, model_type='logistic_regression'):
    """
    Train the fake review detection model.

    Args:
        X_texts (array): Preprocessed text data.
        y_labels (array): Binary labels.
        model_type (str): 'logistic_regression' or 'random_forest'.

    Returns:
        tuple: (model, vectorizer, metrics_dict)
    """
    print(f"\n{'='*60}")
    print(f"  TRAINING: {model_type}")
    print(f"{'='*60}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_texts, y_labels, test_size=0.2, random_state=42, stratify=y_labels
    )
    print(f"[INFO] Train: {len(X_train)} | Test: {len(X_test)}")

    # TF-IDF Vectorization
    print("[INFO] Fitting TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Train model
    if model_type == 'logistic_regression':
        model = LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight='balanced',
            solver='liblinear',
            random_state=42
        )
    elif model_type == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=50,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    print("[INFO] Training model...")
    model.fit(X_train_tfidf, y_train)

    # Evaluate
    y_pred = model.predict(X_test_tfidf)
    y_pred_proba = model.predict_proba(X_test_tfidf)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_pred_proba)

    # Cross-validation
    print("[INFO] Running cross-validation...")
    cv_scores = cross_val_score(model, X_train_tfidf, y_train, cv=5, scoring='accuracy')

    print(f"\n--- Results ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC-ROC:  {auc_score:.4f}")
    print(f"CV Mean:  {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred, target_names=['Genuine', 'Fake'])}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

    metrics = {
        'model_type': model_type,
        'accuracy': round(float(accuracy), 4),
        'auc_roc': round(float(auc_score), 4),
        'cv_mean': round(float(cv_scores.mean()), 4),
        'cv_std': round(float(cv_scores.std()), 4),
        'train_size': len(X_train),
        'test_size': len(X_test),
        'n_features': X_train_tfidf.shape[1]
    }

    return model, vectorizer, metrics


def save_model(model, vectorizer, metrics, output_dir='models'):
    """
    Save trained model, vectorizer, and metrics to disk.

    Args:
        model: Trained sklearn model.
        vectorizer: Fitted TF-IDF vectorizer.
        metrics (dict): Training metrics.
        output_dir (str): Output directory.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save model + vectorizer as a single artifact
    model_path = os.path.join(output_dir, 'fake_review_model.pkl')
    joblib.dump({
        'model': model,
        'vectorizer': vectorizer,
        'metrics': metrics
    }, model_path)
    print(f"\n[INFO] Model saved to: {model_path}")

    # Save metrics as JSON
    metrics_path = os.path.join(output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"[INFO] Metrics saved to: {metrics_path}")

    return model_path


def main():
    """Main training pipeline."""
    print("\n" + "=" * 60)
    print("  [*] FAKE REVIEW DETECTION -- MODEL TRAINING")
    print("=" * 60)

    # Load data
    data_path = os.getenv('TRAINING_DATA', 'data/sample_reviews.csv')
    df = load_data(data_path)

    # Preprocess
    X_texts, y_labels = preprocess_data(df)

    # Train
    model, vectorizer, metrics = train_model(X_texts, y_labels, model_type='logistic_regression')

    # Save
    model_path = save_model(model, vectorizer, metrics)

    print(f"\n{'='*60}")
    print(f"  [+] TRAINING COMPLETE")
    print(f"  Model: {model_path}")
    print(f"  Accuracy: {metrics['accuracy']*100:.1f}%")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
