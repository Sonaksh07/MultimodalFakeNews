import streamlit as st


def render():
    st.title("Multimodal Fake News & Image Tampering Detection")
    st.caption(
        "Big Data–based detection and propagation analysis of fake news and manipulated images on social networks"
    )

    st.markdown("""
    <div class="app-card">
    <div class="kicker">Overview</div>

    <p>
    This system evaluates a submitted news article and/or image through three
    independent analytical pipelines before combining their outputs into a
    single multimodal decision.
    </p>

    <ul>
        <li><b>Text Analysis</b> — BERT classifier with a TF-IDF + Logistic Regression fallback.</li>
        <li><b>Image Analysis</b> — ResNet-50, Error Level Analysis (ELA), and Perceptual Hashing (pHash).</li>
        <li><b>Propagation Analysis</b> — NetworkX graph analytics and a Graph Attention Network (GAT).</li>
    </ul>

    </div>
    """, unsafe_allow_html=True)

    st.markdown("### System Architecture")

    left, right = st.columns(2, gap="large")

    with left:
        st.markdown("""
        <div style="
        padding:18px;
        border:1px solid #2f3542;
        border-radius:10px;
        background:#161b22;
        min-height:210px;
        ">

        <h4>Artificial Intelligence Models</h4>

        <b>Text Processing</b><br>
        • BERT Classifier<br>
        • TF-IDF + Logistic Regression<br><br>

        <b>Image Processing</b><br>
        • ResNet-50<br>
        • Error Level Analysis (ELA)<br>
        • Perceptual Hashing (pHash)

        </div>
        """, unsafe_allow_html=True)

    with right:
        st.markdown("""
        <div style="
        padding:18px;
        border:1px solid #2f3542;
        border-radius:10px;
        background:#161b22;
        min-height:210px;
        ">

        <h4>Graph & Big Data Layer</h4>

        <b>Graph Analytics</b><br>
        • Graph Attention Network (GAT)<br>
        • NetworkX<br>
        • Louvain Community Detection<br><br>

        <b>Data Infrastructure</b><br>
        • Apache Spark<br>
        • PySpark<br>
        • SQLite Database

        </div>
        """, unsafe_allow_html=True)

    st.markdown("")

    st.markdown("### Core Capabilities")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class="app-card" style="text-align:center;">
        <b>Text Analysis</b><br>
        Classification, linguistic analysis, and misinformation detection.
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="app-card" style="text-align:center;">
        <b>Image Analysis</b><br>
        Tampering detection using deep learning and forensic techniques.
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="app-card" style="text-align:center;">
        <b>Explainability</b><br>
        SHAP, Grad-CAM, graph attribution, and multimodal reasoning.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.subheader("Workflow")

    st.markdown("""
    1. Submit article text, an image, or both through **Live Detection**.

    2. Execute multimodal analysis to obtain independent predictions from text, image, and graph models.

    3. Review detailed outputs in **Text Analysis**, **Image Analysis**, and **Propagation Analysis**.

    4. Compare model outputs using the **Model Comparison** module.

    5. Interpret predictions through **Explainability** using SHAP, Grad-CAM, graph influence, and fusion attribution.

    6. Generate a comprehensive diagnostic report from the **Reports** section.
    """)

    st.info(
        "Every analysis performed in Live Detection is automatically synchronized across all dashboard pages. "
        "The dashboard operates on the current submission and does not rely on hardcoded demonstration data."
    )