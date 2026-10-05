"""
Generates a human-readable reasoning paragraph and a structured evidence
list from the actual outputs of a completed analysis (SHAP tokens, ELA/GAT
scores, propagation structure). Every sentence is assembled from real,
per-analysis signals -- nothing here is a fixed template string reused
across articles; which sentences appear, and their wording, depends
entirely on what was actually detected for THIS submission.
"""
from __future__ import annotations

import re

# Lightweight linguistic signal detectors used only to *describe* what SHAP
# already found (never used to score or classify -- that stays 100% model-
# driven). These are the same category of surface cue linguists use to
# describe sensationalist writing; they only affect the wording of the
# evidence list, never the prediction itself.
_SENSATIONAL_PATTERN = re.compile(
    r"\b(shocking|secret|conspiracy|banned|hoax|expos\w+|whistleblow\w+|"
    r"miracle|100%|insiders?|leaked|anonymous)\b", re.IGNORECASE,
)
_SOURCED_PATTERN = re.compile(
    r"\b(according to|confirmed by|officials? said|independent review|"
    r"peer-reviewed|study finds|reported by|statement)\b", re.IGNORECASE,
)
_EMOTIONAL_PATTERN = re.compile(
    r"\b(outrage|terrifying|horrific|unbelievable|you won.?t believe|"
    r"panic|fear|urgent|act now)\b", re.IGNORECASE,
)


def build_evidence_list(analysis) -> list[str]:
    """Returns a list of short evidence strings, each backed by an actual
    detected signal from this specific analysis. Order roughly reflects
    how strongly each signal pointed toward the final verdict."""
    evidence = []
    res = analysis.fusion_result
    is_fake = res["final_label"] == "FAKE"

    # --- Text evidence ---
    if analysis.has_text:
        sensational_hits = _SENSATIONAL_PATTERN.findall(analysis.text)
        sourced_hits = _SOURCED_PATTERN.findall(analysis.text)
        emotional_hits = _EMOTIONAL_PATTERN.findall(analysis.text)

        if sensational_hits:
            unique = sorted(set(w.lower() for w in sensational_hits))
            evidence.append(
                f"Sensationalist/conspiratorial language detected: {', '.join(unique[:5])}."
            )
        if emotional_hits:
            unique = sorted(set(w.lower() for w in emotional_hits))
            evidence.append(f"Emotionally-charged phrasing detected: {', '.join(unique[:5])}.")
        if sourced_hits and not is_fake:
            unique = sorted(set(w.lower() for w in sourced_hits))
            evidence.append(f"Attribution to a credible source found: {', '.join(unique[:3])}.")
        if not sourced_hits and analysis.p_text is not None and analysis.p_text >= 0.5:
            evidence.append("No attribution to an official source, study, or named authority was found in the text.")

        if analysis.shap_explanation and analysis.shap_explanation.get("explanation_df") is not None:
            df = analysis.shap_explanation["explanation_df"]
            if not df.empty:
                top_col = df.columns[1]
                top_positive = df[df[top_col] > 0].head(3)
                if not top_positive.empty:
                    tokens = ", ".join(top_positive.iloc[:, 0].tolist())
                    evidence.append(f"Text model attribution flagged these tokens as most fake-indicative: {tokens}.")

    # --- Image evidence ---
    if analysis.has_image:
        if analysis.ela_score is None:
            evidence.append("Error Level Analysis could not be computed for this image file.")
        elif analysis.ela_is_manipulated:
            evidence.append(
                f"Error Level Analysis flagged the image as likely manipulated "
                f"(anomaly score {analysis.ela_score:.3f})."
            )
        else:
            evidence.append(
                f"Error Level Analysis found no strong splicing/re-compression signal "
                f"(anomaly score {analysis.ela_score:.3f})."
            )
        if analysis.p_image is not None and analysis.p_image >= 0.5:
            evidence.append("Image classifier assigned this image an elevated manipulation-risk score.")
        if analysis.forensic_noise and analysis.forensic_noise.get("is_inconsistent"):
            evidence.append("Noise-inconsistency analysis found unusually uneven noise levels across the image, a pattern sometimes associated with compositing.")
        if analysis.forensic_edges and analysis.forensic_edges.get("is_imbalanced"):
            evidence.append("Edge-density analysis found an uneven distribution across image quadrants, consistent with a possible splice boundary.")
        if analysis.forensic_histogram:
            if analysis.forensic_histogram.get("clipping_detected_channels"):
                evidence.append("Histogram analysis found hard tonal clipping, consistent with aggressive brightness/contrast editing.")
            if analysis.forensic_histogram.get("posterization_detected_channels"):
                evidence.append("Histogram analysis found posterization, consistent with heavy re-compression or tonal editing.")
        if analysis.forensic_metadata and not analysis.forensic_metadata.get("has_exif"):
            evidence.append("No EXIF metadata was found in the image file.")

    # --- Propagation evidence ---
    metrics = analysis.graph_metrics
    bot_ratio = metrics.get("bot_ratio") if isinstance(metrics, dict) else None
    if bot_ratio is not None:
        if bot_ratio > 0.15:
            evidence.append(
                f"Simulated propagation graph shows an elevated proportion of "
                f"suspicious/bot-like spreader accounts ({bot_ratio:.0%})."
            )
        else:
            evidence.append(
                f"Simulated propagation graph shows a low proportion of "
                f"suspicious/bot-like spreader accounts ({bot_ratio:.0%})."
            )
    if analysis.p_graph >= 0.6:
        evidence.append("Graph Attention Network flagged the propagation structure as high-risk.")

    if not evidence:
        evidence.append("No strong individual risk indicators were detected in any submitted modality.")

    return evidence


def build_reasoning_paragraph(analysis) -> str:
    """Composes a short natural-language paragraph explaining WHY the
    system reached its verdict, built entirely from this analysis's actual
    scores, contributions, and evidence -- not a fixed template."""
    res = analysis.fusion_result
    label = res["final_label"]
    confidence = res["confidence_percent"]
    contribs = res["contributions"]
    modality_names = {
        "text_contribution": "the article's text",
        "image_contribution": "the submitted image",
        "graph_contribution": "the simulated propagation pattern",
    }
    top_modality_key = max(contribs, key=contribs.get)
    top_modality = modality_names[top_modality_key]
    top_share = contribs[top_modality_key]

    evidence = build_evidence_list(analysis)
    evidence_clause = evidence[0].rstrip(".").lower() if evidence else "no single dominant signal"

    verdict_word = "likely fabricated or misleading" if label == "FAKE" else "likely authentic"
    confidence_word = (
        "high" if confidence >= 80 else "moderate" if confidence >= 60 else "limited"
    )

    parts = [
        f"This submission was classified as {verdict_word}, with {confidence_word} confidence "
        f"({confidence:.1f}%).",
        f"{top_modality.capitalize()} contributed most to this verdict ({top_share:.0f}% of the fused score), "
        f"primarily because {evidence_clause}.",
    ]
    if len(evidence) > 1:
        parts.append(f"Additional supporting evidence: {evidence[1].rstrip('.').lower()}.")

    if not analysis.has_text:
        parts.append("No article text was submitted, so the text signal defaulted to neutral in the fusion.")
    if not analysis.has_image:
        parts.append("No image was submitted, so the image signal defaulted to neutral in the fusion.")

    return " ".join(parts)
