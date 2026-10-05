import streamlit as st

from dashboard.state import get_current_analysis
from dashboard.components.ui_styles import render_verdict_badge, render_model_tag, render_empty_state


def render():
    st.title("Image Analysis")
    st.caption("Forensic diagnostics — perceptual hash, Error Level Analysis, ResNet-50, and GradCAM — "
               "for the image submitted in Live Detection.")

    analysis = get_current_analysis()

    if not analysis or not analysis.has_image:
        render_empty_state(
            "&#128247;", "No uploaded image available.",
            "Please upload an image in Live Detection.",
        )
        return

    col1, col2 = st.columns(2)
    with col1:
        st.image(analysis.image_path, caption=analysis.title, width="stretch")

    with col2:
        st.markdown(render_verdict_badge("FAKE" if analysis.p_image >= 0.5 else "REAL"), unsafe_allow_html=True)
        st.metric("Image risk score", f"{analysis.p_image:.4f}")
        st.markdown(render_model_tag(analysis.image_model_name), unsafe_allow_html=True)

        st.markdown("#### Forensic signals")
        st.write(f"**Perceptual hash (pHash):** `{analysis.phash_value}`")
        if analysis.ela_score is not None:
            st.write(f"**ELA anomaly score:** `{analysis.ela_score:.4f}`")
            st.write(f"**Splicing/tampering flag:** {'&#9888; Likely manipulated' if analysis.ela_is_manipulated else 'No strong splicing signal'}")
            if analysis.ela_caveat:
                st.caption(f"Note: {analysis.ela_caveat}")
        else:
            st.warning("Error Level Analysis could not be computed for this image "
                       "(unsupported/corrupt file) — no ELA verdict is shown rather than a fabricated one.")

    st.markdown("---")
    st.subheader("Visual localization")
    t1, t2, t3 = st.tabs(["Error Level Analysis (ELA)", "GradCAM (ResNet-50 attention)", "Extended forensics"])

    with t1:
        if analysis.ela_image is not None:
            st.image(analysis.ela_image, caption="Compression-difference map — bright regions indicate inconsistent re-compression, a common splicing artifact.", width="stretch")
        else:
            st.warning("ELA map unavailable for this image.")

    with t2:
        st.image(analysis.gradcam_overlay, caption="Regions the classifier weighted most heavily for this verdict.", width="stretch")

    with t3:
        st.markdown("**Forensic summary**")
        if analysis.forensic_narrative:
            st.write(analysis.forensic_narrative)
        st.markdown("---")

        meta = analysis.forensic_metadata or {}
        hist = analysis.forensic_histogram or {}
        noise = analysis.forensic_noise or {}
        edges = analysis.forensic_edges or {}

        fc1, fc2 = st.columns(2)
        with fc1:
            st.markdown("**Metadata**")
            st.write(f"Format: `{meta.get('format', 'N/A')}` · Dimensions: `{meta.get('dimensions', 'N/A')}`")
            st.write(f"EXIF present: {'Yes' if meta.get('has_exif') else 'No'}")
            if meta.get("software"):
                st.write(f"Editing software tag: `{meta['software']}`")
            if meta.get("camera_make") or meta.get("camera_model"):
                st.write(f"Camera: {meta.get('camera_make', '')} {meta.get('camera_model', '')}")

            st.markdown("**Noise inconsistency**")
            st.write(f"Coefficient of variation across blocks: `{noise.get('coefficient_of_variation', 0):.3f}`")
            st.write("Flagged as inconsistent" if noise.get("is_inconsistent") else "No strong inconsistency detected")

        with fc2:
            st.markdown("**Histogram analysis**")
            if hist.get("clipping_detected_channels"):
                st.write(f"Clipping detected in: {', '.join(hist['clipping_detected_channels'])}")
            else:
                st.write("No hard tonal clipping detected")
            if hist.get("posterization_detected_channels"):
                st.write(f"Posterization detected in: {', '.join(hist['posterization_detected_channels'])}")
            else:
                st.write("No posterization detected")

            st.markdown("**Edge density balance**")
            st.write(f"Quadrant imbalance ratio: `{edges.get('imbalance_ratio', 1.0):.2f}x`")
            st.write("Flagged as imbalanced" if edges.get("is_imbalanced") else "Edge density evenly distributed")
