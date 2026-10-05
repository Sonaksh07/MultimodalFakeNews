"""
Central session/state manager.

This is the single source of truth for "what article/image is currently
being analyzed" across the whole dashboard. Every page reads from
`get_current_analysis()` instead of loading its own sample/demo data, so
navigating between pages never resets or reverts to placeholder content.

Models are loaded exactly once per Streamlit session via `st.cache_resource`
(not once per button click).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from utils.timezone import now_ist, now_ist_str
from typing import Optional

import streamlit as st

from models.nlp.bert_model import BERTFakeNewsClassifier
from models.nlp.tfidf_model import TFIDFBaselineClassifier
from models.cv.resnet_model import ResNet50ImageClassifier
from models.cv.ela_detector import ELADetector
from models.cv.phash_detector import PerceptualHashDetector
from models.cv.gradcam import GradCAMVisualizer
from models.gnn.graph_builder import SocialGraphBuilder
from models.gnn.graph_metrics import GraphAnalyticsEngine
from models.gnn.gat_model import GATPropagationPipeline
from models.fusion.weighted_fusion import WeightedMultimodalFusionEngine
from models.xai.explainer import ExplainableAIEngine

_SESSION_KEY = "current_analysis"


# --------------------------------------------------------------------------
# Model registry -- loaded ONCE per Streamlit session (st.cache_resource),
# never re-instantiated on every page render or button click.
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading models (first run only)...")
def get_models():
    """Returns a dict of every model/engine instance, constructed exactly
    once per Streamlit server session and reused by every page/click."""
    bert = BERTFakeNewsClassifier()
    tfidf = TFIDFBaselineClassifier()
    resnet = ResNet50ImageClassifier()
    ela = ELADetector()
    phash = PerceptualHashDetector()
    builder = SocialGraphBuilder()
    metrics_engine = GraphAnalyticsEngine()
    gat = GATPropagationPipeline()
    fusion = WeightedMultimodalFusionEngine()
    xai = ExplainableAIEngine()
    gradcam = GradCAMVisualizer(resnet)

    models_dict = {
        "bert": bert, "tfidf": tfidf, "resnet": resnet, "ela": ela, "phash": phash,
        "builder": builder, "metrics_engine": metrics_engine, "gat": gat,
        "fusion": fusion, "xai": xai, "gradcam": gradcam,
    }

    from models.fusion.calibration_bootstrap import ensure_calibrator_fitted
    ensure_calibrator_fitted(models_dict)

    return models_dict


@dataclass
class CurrentAnalysis:
    """Everything downstream pages need about the article/image currently
    being analyzed, computed exactly once by `run_full_analysis()`."""

    title: str
    text: str
    has_text: bool
    image_path: Optional[str]
    has_image: bool

    p_text: Optional[float]
    p_text_tfidf: Optional[float]
    p_image: Optional[float]
    p_graph: float
    fusion_result: dict

    graph: object  # networkx.DiGraph
    graph_metrics: dict
    node_importance: dict

    ela_score: Optional[float]
    ela_is_manipulated: Optional[bool]
    ela_image: object  # PIL.Image
    ela_caveat: Optional[str]
    phash_value: Optional[str]
    gradcam_overlay: object  # np.ndarray
    forensic_metadata: Optional[dict]
    forensic_histogram: Optional[dict]
    forensic_noise: Optional[dict]
    forensic_edges: Optional[dict]
    forensic_narrative: Optional[str]

    shap_explanation: Optional[dict]
    graph_node_importance_df: object  # pandas.DataFrame

    text_model_name: Optional[str]
    image_model_name: Optional[str]
    graph_model_name: str
    text_time_seconds: Optional[float]
    image_time_seconds: Optional[float]
    graph_time_seconds: float

    prediction_db_id: Optional[int] = None
    processing_time_seconds: float = 0.0
    analyzed_at: str = field(default_factory=now_ist_str)


def run_full_analysis(title: str, text: str, image_path: Optional[str]) -> CurrentAnalysis:
    """Runs the ENTIRE multimodal pipeline exactly once and stores the
    result in st.session_state, where every other page reads it from.
    This is the only place inference happens for a given user action --
    no page re-runs the models on the same input.

    Supports three modes: text-only, image-only, or text+image. At least
    one of `text`/`image_path` must be provided (enforced by the caller).
    """
    import time
    _start = time.perf_counter()

    m = get_models()
    text = (text or "").strip()
    has_text = len(text) > 0
    has_image = image_path is not None and os.path.exists(image_path)

    # --- Text ---
    if has_text:
        _t0 = time.perf_counter()
        p_text = float(m["bert"].predict_proba([text])[0, 1])
        text_model_name = m["bert"].active_model_name
        # Always also compute the raw TF-IDF baseline score (even when BERT
        # is active) so Model Comparison can show a genuine, same-article
        # BERT-vs-TF-IDF comparison rather than comparing BERT to itself
        # when BERT happens to be delegating to TF-IDF internally.
        p_text_tfidf = float(m["tfidf"].predict_proba([text])[0, 1])
        shap_explanation = m["xai"].explain_text_shap(text, m["tfidf"])
        text_time_seconds = round(time.perf_counter() - _t0, 4)
    else:
        p_text, p_text_tfidf, text_model_name, shap_explanation = None, None, None, None
        text_time_seconds = None

    # --- Image ---
    if has_image:
        _t0 = time.perf_counter()
        p_image = float(m["resnet"].predict_proba(image_path)[1])
        image_model_name = m["resnet"].active_model_name
        ela_image, ela_score, ela_is_manipulated, ela_caveat = m["ela"].compute_ela(image_path)
        phash_value = m["phash"].compute_phash(image_path)
        gradcam_overlay = m["gradcam"].generate_heatmap(image_path)

        from models.cv.forensics_engine import (
            extract_metadata, histogram_analysis, noise_inconsistency_analysis,
            edge_inconsistency_analysis, generate_forensic_narrative,
        )
        forensic_metadata = extract_metadata(image_path)
        forensic_histogram = histogram_analysis(image_path)
        forensic_noise = noise_inconsistency_analysis(image_path)
        forensic_edges = edge_inconsistency_analysis(image_path)
        forensic_narrative = generate_forensic_narrative(
            forensic_metadata, forensic_histogram, forensic_noise, forensic_edges,
        )
        image_time_seconds = round(time.perf_counter() - _t0, 4)
    else:
        p_image, image_model_name = None, None
        ela_image, ela_score, ela_is_manipulated, ela_caveat = None, None, None, None
        phash_value, gradcam_overlay = None, None
        forensic_metadata, forensic_histogram, forensic_noise, forensic_edges, forensic_narrative = None, None, None, None, None
        image_time_seconds = None

    # --- Graph (derived from whatever content IS available for this
    # analysis; falls back to the image's perceptual hash as the seed if
    # there is no text, so image-only submissions still get a graph that's
    # specific to that image rather than a generic default) ---
    _t0 = time.perf_counter()
    graph_seed_text = text if has_text else (phash_value or image_path or title)
    risk_for_graph = p_text if p_text is not None else (p_image if p_image is not None else 0.5)
    graph = m["builder"].build_propagation_graph_for_article(graph_seed_text, risk_score=risk_for_graph)
    graph_metrics = m["metrics_engine"].compute_metrics(graph)
    p_graph, node_importance = m["gat"].predict_graph_probability(graph)
    graph_model_name = m["gat"].active_model_name
    graph_time_seconds = round(time.perf_counter() - _t0, 4)

    # --- Fusion (missing modalities default to neutral 0.5 inside fuse(),
    # this is intentional/documented, not a fake result) ---
    fusion_result = m["fusion"].fuse(
        p_text=p_text if p_text is not None else 0.5,
        p_image=p_image if p_image is not None else 0.5,
        p_graph=p_graph,
    )

    # --- Explainability (graph) ---
    graph_node_importance_df = m["xai"].explain_graph_node_importance(graph)

    analysis = CurrentAnalysis(
        title=title, text=text, has_text=has_text, image_path=image_path, has_image=has_image,
        p_text=p_text, p_text_tfidf=p_text_tfidf, p_image=p_image, p_graph=p_graph, fusion_result=fusion_result,
        graph=graph, graph_metrics=graph_metrics, node_importance=node_importance,
        ela_score=ela_score, ela_is_manipulated=ela_is_manipulated, ela_image=ela_image, ela_caveat=ela_caveat,
        phash_value=phash_value, gradcam_overlay=gradcam_overlay,
        forensic_metadata=forensic_metadata, forensic_histogram=forensic_histogram,
        forensic_noise=forensic_noise, forensic_edges=forensic_edges, forensic_narrative=forensic_narrative,
        shap_explanation=shap_explanation, graph_node_importance_df=graph_node_importance_df,
        text_model_name=text_model_name, image_model_name=image_model_name,
        graph_model_name=graph_model_name,
        text_time_seconds=text_time_seconds, image_time_seconds=image_time_seconds,
        graph_time_seconds=graph_time_seconds,
        processing_time_seconds=round(time.perf_counter() - _start, 3),
    )

    st.session_state[_SESSION_KEY] = analysis
    _persist_to_db(analysis)
    return analysis


def _persist_to_db(analysis: CurrentAnalysis) -> None:
    """Saves the prediction to the database so it shows up in Reports /
    audit history."""
    try:
        from database.db import SessionLocal
        from database.models import PredictionRecord

        db = SessionLocal()
        record = PredictionRecord(
            title=analysis.title,
            text_content=analysis.text,
            image_path=analysis.image_path,
            image_filename=os.path.basename(analysis.image_path) if analysis.image_path else None,
            text_score=analysis.p_text if analysis.p_text is not None else 0.5,
            image_score=analysis.p_image if analysis.p_image is not None else 0.5,
            graph_score=analysis.p_graph,
            final_score=analysis.fusion_result["final_score"],
            final_label=analysis.fusion_result["final_label"],
            confidence=analysis.fusion_result["confidence_percent"],
            text_weight=analysis.fusion_result["weights"]["w_text"],
            image_weight=analysis.fusion_result["weights"]["w_image"],
            graph_weight=analysis.fusion_result["weights"]["w_graph"],
            processing_time_seconds=analysis.processing_time_seconds,
            report_status="not_generated",
            text_model_version=analysis.text_model_name,
            image_model_version=analysis.image_model_name,
            graph_model_version=analysis.graph_model_name,
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        analysis.prediction_db_id = record.id
        db.close()
    except Exception as e:  # noqa: BLE001
        from configs.logging_config import logger
        logger.error(f"Failed to persist prediction to database: {e}")


def get_current_analysis() -> Optional[CurrentAnalysis]:
    """Returns the current session's analysis, or None if the user hasn't
    run Live Detection yet this session."""
    return st.session_state.get(_SESSION_KEY)


def has_current_analysis() -> bool:
    return _SESSION_KEY in st.session_state


def clear_analysis() -> None:
    st.session_state.pop(_SESSION_KEY, None)
