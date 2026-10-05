import streamlit as st

def render():
    st.title("About")
    st.caption("Project synopsis and technical architecture")

    st.markdown("""
    <div class="app-card">
    <div class="kicker">Project title</div>
    <h3>Big Data Based Detection and Propagation Analysis of Fake News and Manipulated Images on Social Networks</h3>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="app-card">
        <div class="kicker">Authors</div>
        <p><b>Sonaksh Jain</b> (Reg No: 230953364)<br/>
        <b>Hansika Abhiraj</b> (Reg No: 230911128)</p>
        <p style="color:var(--text-secondary);font-size:0.85rem;">
        Department of Computer Science &amp; Engineering, Manipal Institute of Technology, Manipal, Karnataka, India<br/>
        Under the supervision of Dr. P.B. Shanthi
        </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="app-card">
        <div class="kicker">Architecture stack</div>
        <ul style="margin-top:6px;">
        <li><b>Text</b> — TF-IDF + Logistic Regression baseline; BERT linear-probe head (frozen embeddings), with a documented TF-IDF fallback if transformers/network access is unavailable.</li>
        <li><b>Image</b> — Perceptual Hashing (pHash), Error Level Analysis (ELA), ResNet-50 backbone.</li>
        <li><b>Graph</b> — NetworkX centralities, Louvain communities, PyTorch Geometric Graph Attention Network (GAT).</li>
        <li><b>Big data</b> — Apache PySpark for distributed preprocessing, with a pandas fallback.</li>
        <li><b>Fusion</b> — Weighted linear combination with per-modality contribution tracking.</li>
        <li><b>Explainability</b> — SHAP (text), GradCAM (image), PageRank/centrality (graph).</li>
        <li><b>Backend</b> — FastAPI + SQLite, ReportLab PDF export.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="app-card">
    <div class="kicker">Design notes</div>
    <p style="font-size:0.9rem;">
    This project has no live connection to a real social-media platform, so the propagation graph
    is a simulated cascade calibrated to each submission's content and risk score, not observed
    real-world spread. BERT and ResNet-50 use frozen pretrained weights (linear-probe / seeded head
    respectively) rather than full fine-tuning, since no labeled fine-tuning dataset ships with the
    project — both are documented in-code with a path to real fine-tuning via the <code>training/</code>
    scripts. Every score shown anywhere in this dashboard is computed live from the current submission;
    nothing is precomputed or hardcoded.
    </p>
    </div>
    """, unsafe_allow_html=True)
