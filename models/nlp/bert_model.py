import os
import torch
import torch.nn as nn
import numpy as np
from configs.logging_config import logger

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("HuggingFace Transformers library not found. Delegating to the trained TF-IDF baseline.")


class BERTFakeNewsClassifier:
    """BERT-based contextual language model for fake news detection.

    Fine-tuning BERT end-to-end is out of scope for a labeled corpus this
    small (~46 demo articles), and attaching a randomly-initialized
    classification head on top of a *frozen* pretrained encoder produces
    near-random, non-reproducible scores -- which is exactly the kind of
    fake/placeholder output this rebuild is meant to eliminate.

    Instead this class uses BERT as a genuine feature extractor: it takes
    the pretrained encoder's [CLS] embedding and trains a small, real
    Logistic Regression head on top of it (a standard "linear probing"
    setup), persisted the same way the TF-IDF baseline is. If
    `transformers`/network access to download the pretrained weights isn't
    available in the current environment, it transparently delegates to the
    already-trained TF-IDF baseline classifier -- a real trained model, not
    a hardcoded keyword list.
    """

    PROBE_PATH = "models/nlp/bert_probe.pkl"

    def __init__(self, model_name: str = "bert-base-uncased", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self._probe = None  # sklearn LogisticRegression trained on [CLS] embeddings
        self._fallback = None  # lazily-created TFIDFBaselineClassifier

        if TRANSFORMERS_AVAILABLE:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    model_name, num_labels=2, output_hidden_states=True,
                ).to(self.device)
                self.model.eval()
                logger.info(f"Loaded BERT encoder '{model_name}' on device '{self.device}'.")
                self._load_or_train_probe()
            except Exception as e:
                logger.warning(f"Failed to load pretrained BERT ({e}). Delegating to TF-IDF baseline.")
                self.model = None

    def _get_cls_embedding(self, texts: list) -> np.ndarray:
        inputs = self.tokenizer(
            texts, padding=True, truncation=True, max_length=128, return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Last hidden state's [CLS] token (position 0) is the standard
            # sentence-level representation used for linear probing.
            cls_embeddings = outputs.hidden_states[-1][:, 0, :].cpu().numpy()
        return cls_embeddings

    def _load_or_train_probe(self):
        import joblib
        from sklearn.linear_model import LogisticRegression

        if os.path.exists(self.PROBE_PATH):
            try:
                self._probe = joblib.load(self.PROBE_PATH)
                logger.info(f"Loaded BERT linear-probe head <- {self.PROBE_PATH}")
                return
            except Exception as e:
                logger.warning(f"Could not load BERT probe ({e}); retraining.")

        from datasets.demo_data import DEMO_ARTICLES
        texts = [a["text"] for a in DEMO_ARTICLES]
        labels = [a["label"] for a in DEMO_ARTICLES]

        embeddings = self._get_cls_embedding(texts)
        probe = LogisticRegression(max_iter=1000, random_state=42)
        probe.fit(embeddings, labels)
        self._probe = probe

        os.makedirs(os.path.dirname(self.PROBE_PATH), exist_ok=True)
        joblib.dump(probe, self.PROBE_PATH)
        logger.info(f"Trained and saved BERT linear-probe head -> {self.PROBE_PATH} "
                    f"on {len(texts)} samples.")

    def _get_fallback(self):
        if self._fallback is None:
            from models.nlp.tfidf_model import TFIDFBaselineClassifier
            self._fallback = TFIDFBaselineClassifier()
        return self._fallback

    def predict_proba(self, texts: list) -> np.ndarray:
        """Returns softmax probability matrix of shape (N, 2) [prob_real, prob_fake]."""
        if self.model is not None and self.tokenizer is not None and self._probe is not None:
            try:
                embeddings = self._get_cls_embedding(texts)
                return self._probe.predict_proba(embeddings)
            except Exception as e:
                logger.error(f"BERT embedding/probe inference failed: {e}. Delegating to TF-IDF baseline.")

        return self._get_fallback().predict_proba(texts)

    def predict(self, texts: list) -> np.ndarray:
        """Returns binary label predictions."""
        probs = self.predict_proba(texts)
        return np.argmax(probs, axis=1)

    @property
    def active_model_name(self) -> str:
        """Human-readable name of whichever model actually serves
        predictions right now, for honest display in the UI (never claim
        BERT is active when it silently fell back to TF-IDF)."""
        if self.model is not None and self.tokenizer is not None and self._probe is not None:
            return f"BERT ({self.model_name}) + linear-probe head"
        return "TF-IDF + Logistic Regression (fallback — BERT/transformers unavailable in this environment)"
