import os
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from configs.logging_config import logger


class TFIDFBaselineClassifier:
    """TF-IDF + Logistic Regression baseline classifier.

    On first use (or if the persisted model file is missing/corrupted) this
    trains automatically on the project's demo corpus (see
    datasets/demo_data.py, ~46 labeled articles) so the classifier is always
    usable without any manual setup step, and persists the result so
    subsequent runs load instantly instead of retraining.
    """

    MODEL_PATH = "models/nlp/tfidf_model.pkl"

    def __init__(self, max_features: int = 5000, c_param: float = 1.0):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            stop_words="english",
        )
        self.model = LogisticRegression(C=c_param, max_iter=1000, random_state=42)
        self.is_fitted = False

        if os.path.exists(self.MODEL_PATH):
            try:
                self.load(self.MODEL_PATH)
            except Exception as e:
                logger.warning(f"Could not load TF-IDF model ({e}); retraining automatically.")
                self._train_demo_model()
        else:
            self._train_demo_model()

    def _train_demo_model(self):
        """Auto-bootstraps a real trained model on the shared demo corpus
        (datasets/demo_data.py) so the classifier never falls back to
        keyword matching or hardcoded scores."""
        from datasets.demo_data import DEMO_ARTICLES

        texts = [a["text"] for a in DEMO_ARTICLES]
        labels = [a["label"] for a in DEMO_ARTICLES]
        self.fit(texts, labels)
        self.save(self.MODEL_PATH)

    def fit(self, texts: list, labels: list):
        """Train TF-IDF + Logistic Regression."""
        X = self.vectorizer.fit_transform(texts)
        self.model.fit(X, labels)
        self.is_fitted = True
        logger.info(f"TF-IDF model trained on {len(texts)} samples.")

    def predict_proba(self, texts: list):
        """Return probabilities."""
        if not self.is_fitted:
            logger.warning("TF-IDF model not fitted. Training demo model...")
            self._train_demo_model()
        X = self.vectorizer.transform(texts)
        return self.model.predict_proba(X)

    def predict(self, texts: list):
        """Return predicted labels."""
        probs = self.predict_proba(texts)
        return np.argmax(probs, axis=1)

    def save(self, filepath: str = None):
        """Save model."""
        filepath = filepath or self.MODEL_PATH
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({"vectorizer": self.vectorizer, "model": self.model}, filepath)
        logger.info(f"Saved TF-IDF model -> {filepath}")

    def load(self, filepath: str = None):
        """Load model."""
        filepath = filepath or self.MODEL_PATH
        data = joblib.load(filepath)
        self.vectorizer = data["vectorizer"]
        self.model = data["model"]
        self.is_fitted = True
        logger.info(f"Loaded TF-IDF model <- {filepath}")
