import streamlit as st
import os

from dashboard.state import run_full_analysis, get_current_analysis
from dashboard.components.ui_styles import render_verdict_badge, render_model_tag, render_empty_state


def render():
    st.title("Live Multimodal Detection")
    st.caption(
        "Submit an article's text, an image, or both. Every other page — Text Analysis, Image Analysis, "
        "Propagation, Model Comparison, Explainability, Reports — will reflect this exact submission until "
        "you run a new analysis."
    )

    col_input, col_img = st.columns([3, 2])

    with col_input:
        st.markdown("<div class='kicker'>Article</div>", unsafe_allow_html=True)
        title = st.text_input("Title (optional)", value="", placeholder="e.g. Headline of the article")
        text = st.text_area(
            "Body text", value="", height=180,
            placeholder="Paste the full article text here...",
        )

    with col_img:
        st.markdown("<div class='kicker'>Image (optional)</div>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload an accompanying image", type=["jpg", "jpeg", "png", "bmp", "webp"])
        image_path = None

        if uploaded_file is not None:
            os.makedirs("uploads", exist_ok=True)
            image_path = f"uploads/{uploaded_file.name}"
            with open(image_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.image(image_path, caption="Uploaded image", width="stretch")
        else:
            render_empty_state("&#128247;", "No image uploaded.", "Text-only analysis will be performed.")

    st.markdown("")
    can_analyze = bool(text.strip()) or (uploaded_file is not None)

    if not can_analyze:
        st.info("Provide article text, an image, or both before analyzing.")

    if st.button("Analyze", type="primary", disabled=not can_analyze, width="stretch"):
        with st.spinner("Running text, image, and graph propagation models..."):
            run_full_analysis(
                title=title.strip() or "Untitled submission",
                text=text.strip(),
                image_path=image_path,
            )
        st.success("Analysis complete — this submission is now active across every page.")
        st.rerun()

    st.markdown("---")

    analysis = get_current_analysis()
    if not analysis:
        render_empty_state("&#9889;", "No analysis yet.", "Fill in the form above and click Analyze.")
        return

    res = analysis.fusion_result
    st.subheader(f"Result — {analysis.title}")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(render_verdict_badge(res["final_label"]), unsafe_allow_html=True)
    with c2:
        st.metric("Fused risk score", f"{res['final_score']:.4f}")
    with c3:
        st.metric("Confidence", f"{res['confidence_percent']:.2f}%")

    st.markdown("#### Modality breakdown")
    u1, u2, u3 = st.columns(3)
    with u1:
        if analysis.has_text:
            st.metric("Text", f"{analysis.p_text:.4f}", f"weight {res['weights']['w_text']:.2f}")
            st.markdown(render_model_tag(analysis.text_model_name), unsafe_allow_html=True)
        else:
            st.metric("Text", "not submitted")
    with u2:
        if analysis.has_image:
            st.metric("Image", f"{analysis.p_image:.4f}", f"weight {res['weights']['w_image']:.2f}")
            st.markdown(render_model_tag(analysis.image_model_name), unsafe_allow_html=True)
        else:
            st.metric("Image", "not submitted")
    with u3:
        st.metric("Propagation", f"{analysis.p_graph:.4f}", f"weight {res['weights']['w_graph']:.2f}")
        st.markdown(render_model_tag(analysis.graph_model_name), unsafe_allow_html=True)

    st.markdown("#### Why this verdict?")
    from models.reasoning.evidence_engine import build_reasoning_paragraph
    st.write(build_reasoning_paragraph(analysis))

    st.caption(
        "Text and image scores default to a neutral 0.5 when that modality wasn't submitted — "
        "this is a documented default, not a fabricated result. "
        "See **Text Analysis**, **Image Analysis**, **Propagation**, **Model Comparison**, "
        "**Explainability**, and **Reports** for a deeper breakdown of this submission."
    )
