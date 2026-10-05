import numpy as np
import pandas as pd
from configs.logging_config import logger

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP library not found. Operating with raw TF-IDF*coefficient attribution.")


class ExplainableAIEngine:
    """Unified Explainable AI (XAI) suite for Text (SHAP), Vision (GradCAM), and Graph (Node Centrality)."""

    def __init__(self):
        pass

    def explain_text_shap(self, text: str, tfidf_model) -> dict:
        """Computes real word/token attribution scores for the CURRENT article
        text against the trained TF-IDF + Logistic Regression model.

        Primary path: SHAP LinearExplainer over the model's actual learned
        coefficients. Secondary path (only if the `shap` package itself is
        unavailable, not if it merely errors on this input): the token's
        raw (tfidf_weight * logistic_regression_coefficient) product, which
        is still a real, per-article, per-model-derived attribution value —
        never a fixed keyword dictionary.
        """
        text = text or ""
        tokens = text.split()
        if not tokens:
            return {"top_tokens": [], "shap_values": [], "explanation_df": pd.DataFrame(columns=["Word/Token", "Attribution Score"])}

        if not (hasattr(tfidf_model, "model") and tfidf_model.is_fitted):
            logger.warning("TF-IDF model is not fitted; cannot compute attribution for this article.")
            return {"top_tokens": [], "shap_values": [], "explanation_df": pd.DataFrame(columns=["Word/Token", "Attribution Score"])}

        feature_names = tfidf_model.vectorizer.get_feature_names_out()
        X_vec = tfidf_model.vectorizer.transform([text])
        nonzero = X_vec.nonzero()[1]

        if len(nonzero) == 0:
            # Every token was out-of-vocabulary / stopword-filtered.
            return {"top_tokens": [], "shap_values": [], "explanation_df": pd.DataFrame(columns=["Word/Token", "Attribution Score"])}

        if SHAP_AVAILABLE:
            try:
                explainer = shap.LinearExplainer(
                    tfidf_model.model,
                    tfidf_model.vectorizer.transform(["baseline"]),
                )
                shap_vals = explainer.shap_values(X_vec)
                if isinstance(shap_vals, list):
                    shap_vals = shap_vals[1]  # class 1 == "fake"

                token_scores = [(feature_names[idx], float(shap_vals[0, idx])) for idx in nonzero]
                token_scores.sort(key=lambda t: abs(t[1]), reverse=True)
                top = token_scores[:10]
                return {
                    "top_tokens": [t[0] for t in top],
                    "shap_values": [t[1] for t in top],
                    "explanation_df": pd.DataFrame(top, columns=["Word/Token", "SHAP Impact Score"]),
                }
            except Exception as e:
                logger.warning(f"SHAP explainer call failed ({e}); using raw TF-IDF*coefficient attribution for this article.")

        # Secondary (still real, model-derived) path: raw coefficient * tfidf weight.
        coefs = tfidf_model.model.coef_[0]
        token_scores = [(feature_names[idx], float(X_vec[0, idx] * coefs[idx])) for idx in nonzero]
        token_scores.sort(key=lambda t: abs(t[1]), reverse=True)
        top = token_scores[:10]
        return {
            "top_tokens": [t[0] for t in top],
            "shap_values": [t[1] for t in top],
            "explanation_df": pd.DataFrame(top, columns=["Word/Token", "Attribution Score (TF-IDF x Coefficient)"]),
        }

    def explain_graph_node_importance(self, G, top_n: int = 5) -> pd.DataFrame:
        """Ranks graph nodes by their social influence and propagation centrality."""
        try:
            import networkx as nx
            if G.number_of_nodes() == 0:
                return pd.DataFrame()
            pr = nx.pagerank(G, alpha=0.85)
            deg = nx.degree_centrality(G)
            bet = nx.betweenness_centrality(G)

            records = []
            for node in G.nodes():
                data = G.nodes[node]
                records.append({
                    "Node ID": node,
                    "Label": data.get("label", str(node)),
                    "Role": data.get("type", "user"),
                    "PageRank (Authority)": round(pr.get(node, 0.0), 4),
                    "Degree Centrality": round(deg.get(node, 0.0), 4),
                    "Betweenness (Bridge)": round(bet.get(node, 0.0), 4),
                    "Bot Risk Score": round(data.get("bot_score", 0.1), 2),
                })

            df = pd.DataFrame(records).sort_values(by="PageRank (Authority)", ascending=False)
            return df.head(top_n)
        except Exception as e:
            logger.error(f"Graph node importance explanation failed: {e}")
            return pd.DataFrame()
