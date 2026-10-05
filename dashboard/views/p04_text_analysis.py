import streamlit as st

from dashboard.state import get_current_analysis, get_models
from dashboard.components.ui_styles import render_verdict_badge, render_model_tag, render_empty_state


def render():
    st.title("Text Analysis")
    st.caption("Text-model verdict and token-level attribution for the article submitted in Live Detection.")

    analysis = get_current_analysis()

    if not analysis or not analysis.has_text:
        render_empty_state(
            "&#128221;", "No article text has been analyzed yet.",
            "Submit article text in Live Detection to see its text-model diagnostics here.",
        )
    else:
        st.markdown(f"### {analysis.title}")
        with st.expander("Full article text", expanded=False):
            st.write(analysis.text)

        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown(render_verdict_badge("FAKE" if analysis.p_text >= 0.5 else "REAL"), unsafe_allow_html=True)
            st.metric("Text risk score", f"{analysis.p_text:.4f}")
            st.markdown(render_model_tag(analysis.text_model_name), unsafe_allow_html=True)

        with c2:
            st.markdown("**Token attribution for this article**")
            df = analysis.shap_explanation.get("explanation_df") if analysis.shap_explanation else None
            if df is not None and not df.empty:
                st.dataframe(df, width="stretch")
            else:
                st.caption("No attributable tokens were found for this text "
                           "(e.g. it consisted entirely of stopwords or out-of-vocabulary terms).")

    st.markdown("---")
    with st.expander("Reference: TF-IDF vs BERT benchmark on the project's demo corpus", expanded=False):
        st.caption(
            "This section benchmarks the two text models against a labeled demo corpus for reference "
            "only — it is not part of the live analysis above and does not affect the current submission's verdict."
        )
        from datasets.dataset_loader import DatasetLoader
        from models.nlp.nlp_evaluator import NLPEvaluator
        from utils.visualization import plot_confusion_matrix, plot_roc_curve

        loader = DatasetLoader()
        df = loader.load_dataset("demo")
        evaluator = NLPEvaluator()
        metrics = evaluator.evaluate_all(df["text"].tolist(), df["label"].tolist())
        tfidf_m, bert_m = metrics["tfidf_logistic_regression"], metrics["bert_transformer"]

        comp_table = [
            {"Model": "TF-IDF + Logistic Regression", "Accuracy": f"{tfidf_m['accuracy']:.4f}", "Precision": f"{tfidf_m['precision']:.4f}", "Recall": f"{tfidf_m['recall']:.4f}", "F1": f"{tfidf_m['f1_score']:.4f}", "ROC-AUC": f"{tfidf_m['auc']:.4f}"},
            {"Model": "BERT / linear-probe", "Accuracy": f"{bert_m['accuracy']:.4f}", "Precision": f"{bert_m['precision']:.4f}", "Recall": f"{bert_m['recall']:.4f}", "F1": f"{bert_m['f1_score']:.4f}", "ROC-AUC": f"{bert_m['auc']:.4f}"},
        ]
        st.dataframe(comp_table, width="stretch")

        bc1, bc2 = st.columns(2)
        with bc1:
            st.plotly_chart(plot_roc_curve(metrics), width="stretch")
        with bc2:
            st.plotly_chart(plot_confusion_matrix(bert_m["confusion_matrix"], title="BERT Confusion Matrix"), width="stretch")

        st.caption(f"Benchmarked on {len(df)} labeled demo articles (see datasets/demo_data.py).")
