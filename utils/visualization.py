import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
import numpy as np
import pandas as pd
from PIL import Image
import io

def plot_confusion_matrix(cm: list, title: str = "Confusion Matrix"):
    """Generates Plotly Heatmap for Confusion Matrix."""
    labels = ["REAL", "FAKE"]
    fig = px.imshow(
        cm,
        x=labels,
        y=labels,
        color_continuous_scale="Blues",
        text_auto=True,
        title=title,
        labels=dict(x="Predicted Label", y="Actual Label", color="Count")
    )
    fig.update_layout(template="plotly_dark", margin=dict(l=40, r=40, t=60, b=40))
    return fig

def plot_roc_curve(metrics_dict: dict):
    """Generates Plotly ROC Curve comparing models."""
    fig = go.Figure()

    for model_name, metrics in metrics_dict.items():
        fpr = metrics.get("fpr", [0, 1])
        tpr = metrics.get("tpr", [0, 1])
        auc_val = metrics.get("auc", 0.5)

        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f"{model_name} (AUC = {auc_val:.3f})",
            line=dict(width=2.5)
        ))

    # Baseline diagonal
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Baseline',
        line=dict(dash='dash', color='gray')
    ))

    fig.update_layout(
        title="ROC Curve - Model Comparison",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        template="plotly_dark",
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig

def plot_propagation_network(G: nx.DiGraph, summary_df: pd.DataFrame = None):
    """Generates dynamic Plotly 2D Network Graph for social propagation analysis."""
    pos = nx.spring_layout(G, seed=42)

    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.8, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    node_x, node_y, node_text, node_color, node_size = [], [], [], [], []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

        b_score = G.nodes[node].get("bot_score", 0.1)
        u_type = G.nodes[node].get("type", "user")
        label = G.nodes[node].get("label", str(node))

        color_map = {"article": "#e74c3c", "suspicious_bot": "#e67e22", "influencer": "#f1c40f", "user": "#3498db"}
        node_color.append(color_map.get(u_type, "#9b59b6"))
        node_size.append(25 if u_type == "article" else (18 if u_type == "influencer" else 10))

        node_text.append(f"Node: {label}<br>Type: {u_type}<br>Bot Risk: {b_score:.2f}")

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            color=node_color,
            size=node_size,
            line=dict(width=1, color='#fff')
        )
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title="Social Media Retweet & Reply Propagation Graph",
        showlegend=False,
        hovermode='closest',
        template="plotly_dark",
        margin=dict(b=20, l=20, r=20, t=50),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    return fig
