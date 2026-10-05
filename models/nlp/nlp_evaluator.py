import numpy as np
import pandas as pd
from models.nlp.tfidf_model import TFIDFBaselineClassifier
from models.nlp.bert_model import BERTFakeNewsClassifier
from utils.metrics import compute_classification_metrics
from configs.logging_config import logger

class NLPEvaluator:
    """Evaluates and compares TF-IDF + Logistic Regression baseline vs BERT contextual model."""

    def __init__(self):
        self.tfidf_model = TFIDFBaselineClassifier()
        self.bert_model = BERTFakeNewsClassifier()

    def evaluate_all(self, texts: list, y_true: list) -> dict:
        """Runs evaluation for both TF-IDF baseline and BERT model."""
        y_true = np.array(y_true)

        tfidf_probs = self.tfidf_model.predict_proba(texts)[:, 1]
        tfidf_preds = (tfidf_probs >= 0.5).astype(int)

        bert_probs = self.bert_model.predict_proba(texts)[:, 1]
        bert_preds = (bert_probs >= 0.5).astype(int)

        tfidf_metrics = compute_classification_metrics(y_true, tfidf_preds, tfidf_probs)
        tfidf_metrics["confidence_avg"] = float(np.mean(np.abs(tfidf_probs - 0.5) * 2.0))

        bert_metrics = compute_classification_metrics(y_true, bert_preds, bert_probs)
        bert_metrics["confidence_avg"] = float(np.mean(np.abs(bert_probs - 0.5) * 2.0))

        logger.info("Completed text model comparative evaluation.")
        return {
            "tfidf_logistic_regression": tfidf_metrics,
            "bert_transformer": bert_metrics
        }
