
"""Spam vs Ham Email Classifier using TF-IDF and Logistic Regression.

This script trains a TF-IDF + Logistic Regression model on spam.csv
and saves:
  - Trained model: models/spam_tfidf_logreg.joblib
  - PNG charts in docs/: label_distribution.png, top_spam_terms.png, confusion_matrix_tfidf.png
"""

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def train_model(data_path: Path, models_dir: Path, docs_dir: Path) -> None:
    # Load dataset
    df = pd.read_csv(data_path)
    X = df["Message"]
    y = df["Category"]

    # Train/test split (stratified to keep spam/ham ratio similar)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Build TF-IDF + Logistic Regression pipeline
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(stop_words="english", max_df=0.95, min_df=2)),
            ("logreg", LogisticRegression(max_iter=1000)),
        ]
    )

    # Train
    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.4f}")

    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

    # Confusion matrix
    labels = ["ham", "spam"]
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    print("Confusion matrix (rows=true, cols=predicted):")
    print(cm)

    # ======================
    # Visualization 1: Label distribution
    # ======================
    docs_dir.mkdir(exist_ok=True)
    models_dir.mkdir(exist_ok=True)

    label_counts = y.value_counts()
    plt.figure()
    plt.bar(label_counts.index.astype(str), label_counts.values)
    plt.xlabel("Email Category")
    plt.ylabel("Count")
    plt.title("Spam vs Ham Label Distribution")
    plt.tight_layout()
    plt.savefig(docs_dir / "label_distribution.png")
    plt.close()

    # ======================
    # Visualization 2: Top spam TF-IDF terms
    # ======================
    tfidf: TfidfVectorizer = pipeline.named_steps["tfidf"]
    logreg: LogisticRegression = pipeline.named_steps["logreg"]

    feature_names = np.array(tfidf.get_feature_names_out())
    # For binary classification, coef_[0] corresponds to the positive class
    spam_coefs = logreg.coef_[0]

    top_n = 20
    top_indices = np.argsort(spam_coefs)[-top_n:]
    top_terms = feature_names[top_indices]
    top_values = spam_coefs[top_indices]

    plt.figure(figsize=(8, 6))
    positions = np.arange(len(top_terms))
    plt.barh(positions, top_values)
    plt.yticks(positions, top_terms)
    plt.xlabel("Coefficient Weight")
    plt.title("Top TF-IDF Terms for Spam Class")
    plt.tight_layout()
    plt.savefig(docs_dir / "top_spam_terms.png")
    plt.close()

    # ======================
    # Visualization 3: Confusion matrix
    # ======================
    plt.figure()
    im = plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix (TF-IDF + Logistic Regression)")
    plt.colorbar(im)
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.tight_layout()
    plt.savefig(docs_dir / "confusion_matrix_tfidf.png")
    plt.close()

    # ======================
    # Save model
    # ======================
    model_path = models_dir / "spam_tfidf_logreg.joblib"
    joblib.dump(pipeline, model_path)
    print(f"Model saved to: {model_path}")

    # Accuracy explanation (for your report):
    # Accuracy is the fraction of correctly classified emails on the test set.
    # It is computed as: (number of correct predictions) / (total number of test emails).
    # A high accuracy (close to 1.0) means the model is good at distinguishing spam from ham
    # on unseen data, but you should always check the confusion matrix and precision/recall
    # to be sure it is not biased toward one class.


def predict_single_email(model_path: Path, text: str) -> str:
    """Load a saved model and predict whether the given email text is spam or ham."""
    pipeline: Pipeline = joblib.load(model_path)
    prediction = pipeline.predict([text])[0]
    return prediction


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a TF-IDF spam/ham classifier.")
    parser.add_argument(
        "--data",
        type=str,
        default="spam.csv",
        help="Path to spam.csv dataset (with Category and Message columns).",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="Directory to save trained model.",
    )
    parser.add_argument(
        "--docs-dir",
        type=str,
        default="docs",
        help="Directory to save documentation images (PNG).",
    )

    args = parser.parse_args()

    data_path = Path(args.data)
    models_dir = Path(args.models_dir)
    docs_dir = Path(args.docs_dir)

    train_model(data_path, models_dir, docs_dir)


if __name__ == "__main__":
    main()
