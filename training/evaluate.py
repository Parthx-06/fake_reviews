"""
Model Evaluation Script.
Generates detailed evaluation metrics and visualizations.
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.preprocessing import TextPreprocessor


def evaluate_model(model_path='models/fake_review_model.pkl', data_path=None):
    """
    Evaluate a trained model on the dataset.

    Args:
        model_path (str): Path to the saved model.
        data_path (str): Path to evaluation data.
    """
    if data_path is None:
        if os.path.exists('data/fake_reviews_dataset.csv'):
            data_path = 'data/fake_reviews_dataset.csv'
        else:
            data_path = 'data/sample_reviews.csv'
    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found: {model_path}")
        print("[INFO]  Run 'python -m training.train' first.")
        sys.exit(1)

    # Load model
    artifacts = joblib.load(model_path)
    model = artifacts['model']
    vectorizer = artifacts['vectorizer']
    print(f"[INFO] Model loaded from {model_path}")

    # Load and preprocess data
    df = pd.read_csv(data_path)
    preprocessor = TextPreprocessor()

    # Auto-detect columns
    text_col = None
    label_col = None
    for col in df.columns:
        col_lower = col.lower().strip()
        if col_lower in ['text_', 'text', 'review', 'review_text', 'content']:
            text_col = col
        if col_lower in ['label', 'labels', 'fake', 'is_fake', 'class', 'target']:
            label_col = col

    if text_col is None:
        text_col = df.columns[0]
    if label_col is None:
        label_col = df.columns[-1]

    df = df.dropna(subset=[text_col, label_col])
    df['cleaned'] = df[text_col].apply(preprocessor.clean_text)

    # Encode labels
    unique_labels = df[label_col].unique()
    label_mapping = {}
    for label in unique_labels:
        label_str = str(label).lower().strip()
        if label_str in ['or', 'fake', '1', 'deceptive', 'spam', 'yes', 'true']:
            label_mapping[label] = 1
        else:
            label_mapping[label] = 0

    df['binary_label'] = df[label_col].map(label_mapping)

    X = vectorizer.transform(df['cleaned'].tolist())
    y_true = np.array(df['binary_label'].tolist(), dtype=int)

    # Predictions
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    # Metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    auc = roc_auc_score(y_true, y_proba)
    cm = confusion_matrix(y_true, y_pred)

    report = {
        'accuracy': round(float(accuracy), 4),
        'precision': round(float(precision), 4),
        'recall': round(float(recall), 4),
        'f1_score': round(float(f1), 4),
        'auc_roc': round(float(auc), 4),
        'confusion_matrix': cm.tolist(),
        'total_samples': len(y_true),
        'genuine_count': int(np.sum(y_true == 0)),
        'fake_count': int(np.sum(y_true == 1))
    }

    print("\n" + "=" * 50)
    print("  MODEL EVALUATION REPORT")
    print("=" * 50)
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  AUC-ROC:   {auc:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"  TN={cm[0][0]}  FP={cm[0][1]}")
    print(f"  FN={cm[1][0]}  TP={cm[1][1]}")
    print("=" * 50)

    print(f"\nFull Classification Report:\n")
    print(classification_report(y_true, y_pred, target_names=['Genuine', 'Fake']))

    # Save report
    os.makedirs('models', exist_ok=True)
    with open('models/evaluation_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    print("[INFO] Report saved to models/evaluation_report.json")

    return report


if __name__ == '__main__':
    evaluate_model()
