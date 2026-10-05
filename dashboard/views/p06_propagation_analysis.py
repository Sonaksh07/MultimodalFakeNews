import streamlit as st

from utils.visualization import plot_propagation_network
from dashboard.state import get_current_analysis
from dashboard.components.ui_styles import render_model_tag, render_empty_state


def render():
    st.title("Propagation Analysis")
    st.caption("Simulated social-spread network, network topology metrics, and Graph Attention Network "
               "propagation risk for the current submission.")

    analysis = get_current_analysis()
    if not analysis:
        render_empty_state(
            "&#127760;", "No analysis yet.",
            "Run an analysis in Live Detection to generate this submission's propagation graph.",
        )
        return

    G = analysis.graph
    metrics_res = analysis.graph_metrics
    p_graph = analysis.p_graph

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Nodes", G.number_of_nodes())
    m2.metric("Cascade edges", G.number_of_edges())
    m3.metric("Communities", len(set(metrics_res["communities"].values())))
    m4.metric("GAT propagation risk", f"{p_graph:.4f}")
    st.markdown(render_model_tag(analysis.graph_model_name), unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Social spread network")
    fig_graph = plot_propagation_network(G, metrics_res["summary_df"])
    st.plotly_chart(fig_graph, width="stretch")

    st.markdown("---")
    st.subheader("Top influential / high bot-risk nodes")
    st.dataframe(metrics_res["summary_df"].head(10), width="stretch")

    st.caption(
        "This project has no live connection to real social-platform interaction data, so the "
        "propagation cascade is a simulated model, not observed real-world spread. Its structure "
        "(bot ratio, spread fan-out) is deterministically derived from this submission's content and "
        "risk score — see `SocialGraphBuilder.build_propagation_graph_for_article()`."
    )
