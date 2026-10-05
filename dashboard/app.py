import streamlit as st
import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from database.db import init_db
from dashboard.components.ui_styles import apply_custom_glassmorphism
from dashboard.views import (
    p01_home, p02_live_detection, p03_dataset_explorer,
    p04_text_analysis, p05_image_analysis, p06_propagation_analysis,
    p07_model_comparison, p08_explainability, p09_reports, p10_about
)

st.set_page_config(
    page_title="Multimodal Fake News Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

init_db()
apply_custom_glassmorphism()

# Sidebar Navigation
st.sidebar.markdown("### Multimodal Misinformation Detection")
st.sidebar.caption("Big Data & Graph Neural Network Detection Engine")

menu_options = [
    "Home",
    "Live Detection",
    "Dataset Explorer",
    "Text Analysis",
    "Image Analysis",
    "Propagation Analysis",
    "Model Comparison",
    "Explainability",
    "Reports",
    "About",
]

choice = st.sidebar.radio("Navigation", menu_options, label_visibility="collapsed")

st.sidebar.markdown("---")

from dashboard.state import get_current_analysis, clear_analysis  # noqa: E402

_analysis = get_current_analysis()
if _analysis:
    st.sidebar.markdown("**Active analysis**")
    st.sidebar.write(_analysis.title[:48] + ("…" if len(_analysis.title) > 48 else ""))
    st.sidebar.caption(f"Verdict: {_analysis.fusion_result['final_label']} "
                        f"({_analysis.fusion_result['final_score']:.3f}) — shown on every page.")
    if st.sidebar.button("Clear current analysis"):
        clear_analysis()
        st.rerun()
else:
    st.sidebar.caption("No article analyzed yet. Go to **Live Detection** to begin.")

# Page Routing
_pages = {
    "Home": p01_home.render,
    "Live Detection": p02_live_detection.render,
    "Dataset Explorer": p03_dataset_explorer.render,
    "Text Analysis": p04_text_analysis.render,
    "Image Analysis": p05_image_analysis.render,
    "Propagation Analysis": p06_propagation_analysis.render,
    "Model Comparison": p07_model_comparison.render,
    "Explainability": p08_explainability.render,
    "Reports": p09_reports.render,
    "About": p10_about.render,
}
_pages[choice]()
