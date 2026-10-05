"""
Fits the confidence calibrator (see calibration.py) using real per-modality
model inference over the project's labeled demo corpus, so confidence
values are backed by an actual statistical fit rather than a fixed formula.

Called once from both entry points (dashboard/state.py's cached
`get_models()` and backend/main.py's startup), so it only ever runs once
per process and its result is persisted to disk for subsequent runs.
"""
from __future__ import annotations

from configs.logging_config import logger


def ensure_calibrator_fitted(models: dict, sample_size: int = 30) -> None:
    fusion = models["fusion"]
    if fusion.calibrator.is_fitted:
        return

    from datasets.dataset_loader import DatasetLoader

    loader = DatasetLoader()
    df = loader.load_dataset("demo")
    df = df.sample(n=min(sample_size, len(df)), random_state=42).reset_index(drop=True)

    raw_scores, labels = [], []
    for _, row in df.iterrows():
        text = row["text"]
        image_path = row.get("image_path")

        p_text = float(models["bert"].predict_proba([text])[0, 1])
        p_image = float(models["resnet"].predict_proba(image_path)[1]) if image_path else 0.5
        graph = models["builder"].build_propagation_graph_for_article(text, risk_score=p_text, num_nodes=20)
        p_graph, _ = models["gat"].predict_graph_probability(graph)

        raw_score = fusion.w_text * p_text + fusion.w_image * p_image + fusion.w_graph * p_graph
        raw_scores.append(raw_score)
        labels.append(int(row["label"]))

    result = fusion.fit_calibrator(raw_scores, labels)
    logger.info(f"Confidence calibrator bootstrap: {result}")
