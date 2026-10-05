import numpy as np
from configs.logging_config import logger
from models.fusion.calibration import ConfidenceCalibrator

class WeightedMultimodalFusionEngine:
    """Combines unimodal probabilities (Text, Image, Graph) into a unified final prediction."""

    def __init__(self, w_text: float = 0.40, w_image: float = 0.35, w_graph: float = 0.25):
        self.w_text = w_text
        self.w_image = w_image
        self.w_graph = w_graph
        self._normalize_weights()
        self.calibrator = ConfidenceCalibrator()

    def fit_calibrator(self, raw_scores: list, labels: list) -> dict:
        """Fits the confidence calibrator. See calibration_bootstrap.py for
        the caller that assembles (raw_score, label) pairs from real
        per-modality inference over the demo corpus."""
        return self.calibrator.fit(raw_scores, labels)

    def _normalize_weights(self):
        """Ensures weights sum up to 1.0."""
        total = self.w_text + self.w_image + self.w_graph
        if total > 0:
            self.w_text /= total
            self.w_image /= total
            self.w_graph /= total

    def fuse(self, p_text: float, p_image: float, p_graph: float) -> dict:
        """
        Calculates weighted multimodal score and returns full explanation dict.
        Scores should be probability of FAKE (0.0 = Real, 1.0 = Fake).
        """
        # Weighted linear combination
        final_score = (
            self.w_text * p_text +
            self.w_image * p_image +
            self.w_graph * p_graph
        )

        # Both the label and the confidence are derived from the SAME
        # calibrated probability, so they can never disagree (e.g. label
        # says FAKE while confidence secretly implies REAL is more likely
        # -- possible if the label used the raw threshold while confidence
        # used a differently-centered calibrated probability).
        calibrated_p_fake = self.calibrator.calibrate(final_score)
        final_label = "FAKE" if calibrated_p_fake >= 0.50 else "REAL"
        confidence = float((calibrated_p_fake if final_label == "FAKE" else (1.0 - calibrated_p_fake)) * 100.0)

        # Modality relative contributions
        contributions = {
            "text_contribution": float(round((self.w_text * p_text) / (final_score + 1e-8) * 100.0, 2)),
            "image_contribution": float(round((self.w_image * p_image) / (final_score + 1e-8) * 100.0, 2)),
            "graph_contribution": float(round((self.w_graph * p_graph) / (final_score + 1e-8) * 100.0, 2)),
        }

        logger.info(f"Multimodal Fusion: Score={final_score:.4f}, Label={final_label}, Confidence={confidence:.2f}%")

        return {
            "final_score": float(round(final_score, 4)),
            "final_label": final_label,
            "confidence_percent": float(round(confidence, 2)),
            "unimodal_scores": {
                "text_score": float(round(p_text, 4)),
                "image_score": float(round(p_image, 4)),
                "graph_score": float(round(p_graph, 4))
            },
            "weights": {
                "w_text": float(round(self.w_text, 2)),
                "w_image": float(round(self.w_image, 2)),
                "w_graph": float(round(self.w_graph, 2))
            },
            "contributions": contributions
        }

    def evaluate_all_combinations(self, test_samples: list) -> dict:
        """
        Runs ablation experiments comparing all modality subsets:
        - Text Only
        - Image Only
        - Graph Only
        - Text + Image
        - Text + Graph
        - Image + Graph
        - All 3 Modalities (Trimodal)
        """
        results = {}
        combos = {
            "Text Only": (1.0, 0.0, 0.0),
            "Image Only": (0.0, 1.0, 0.0),
            "Graph Only": (0.0, 0.0, 1.0),
            "Text + Image": (0.55, 0.45, 0.0),
            "Text + Graph": (0.60, 0.0, 0.40),
            "Image + Graph": (0.0, 0.60, 0.40),
            "All Three Modalities": (self.w_text, self.w_image, self.w_graph)
        }

        for combo_name, (wt, wi, wg) in combos.items():
            scores = []
            for sample in test_samples:
                pt, pi, pg = sample["p_text"], sample["p_image"], sample["p_graph"]
                y_true = sample["y_true"]
                
                score = wt * pt + wi * pi + wg * pg
                pred = 1 if score >= 0.5 else 0
                scores.append(1 if pred == y_true else 0)

            acc = float(np.mean(scores)) if scores else None
            results[combo_name] = {
                "accuracy": round(acc, 4) if acc is not None else None,
                "n_samples": len(scores),
                "weights": {"text": wt, "image": wi, "graph": wg}
            }

        return results
