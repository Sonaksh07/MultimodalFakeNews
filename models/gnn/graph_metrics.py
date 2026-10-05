import networkx as nx
import pandas as pd
from configs.logging_config import logger

try:
    import community as community_louvain
    LOUVAIN_AVAILABLE = True
except ImportError:
    LOUVAIN_AVAILABLE = False
    logger.warning("python-louvain not installed. Falling back to NetworkX greedy modularity communities.")

class GraphAnalyticsEngine:
    """Computes social network topology metrics: PageRank, Centralities, and Louvain Communities."""

    def __init__(self):
        pass

    def compute_metrics(self, G: nx.DiGraph) -> dict:
        """Computes NetworkX graph metrics."""
        # 1. PageRank
        try:
            pagerank = nx.pagerank(G, alpha=0.85)
        except Exception:
            pagerank = {n: 1.0 / G.number_of_nodes() for n in G.nodes()}

        # 2. Degree Centrality
        degree_centrality = nx.degree_centrality(G)

        # 3. Betweenness Centrality
        try:
            betweenness_centrality = nx.betweenness_centrality(G)
        except Exception:
            betweenness_centrality = {n: 0.0 for n in G.nodes()}

        # 4. Community Detection (Louvain algorithm)
        G_undirected = G.to_undirected()
        if LOUVAIN_AVAILABLE:
            try:
                communities = community_louvain.best_partition(G_undirected)
            except Exception:
                communities = {n: 0 for n in G.nodes()}
        else:
            try:
                comm_sets = list(nx.community.greedy_modularity_communities(G_undirected))
                communities = {}
                for idx, c_set in enumerate(comm_sets):
                    for n in c_set:
                        communities[n] = idx
            except Exception:
                communities = {n: 0 for n in G.nodes()}

        # Combine into DataFrame summary
        summary_data = []
        for node in G.nodes():
            node_data = G.nodes[node]
            summary_data.append({
                "node_id": node,
                "label": node_data.get("label", str(node)),
                "type": node_data.get("type", "user"),
                "pagerank": pagerank.get(node, 0.0),
                "degree_centrality": degree_centrality.get(node, 0.0),
                "betweenness_centrality": betweenness_centrality.get(node, 0.0),
                "community": communities.get(node, 0),
                "bot_score": node_data.get("bot_score", 0.1)
            })

        df_summary = pd.DataFrame(summary_data).sort_values(by="pagerank", ascending=False)

        bot_count = int((df_summary["type"] == "suspicious_bot").sum())
        bot_ratio = bot_count / len(df_summary) if len(df_summary) else 0.0

        try:
            root = next((n for n, d in G.nodes(data=True) if d.get("type") == "article"), None)
            if root is not None and G.number_of_nodes() > 1:
                lengths = nx.single_source_shortest_path_length(G, root)
                diffusion_depth = max(lengths.values()) if lengths else 0
            else:
                diffusion_depth = 0
        except Exception:
            diffusion_depth = 0

        logger.info(f"Computed network metrics for {G.number_of_nodes()} nodes across {len(set(communities.values()))} communities.")
        return {
            "pagerank": pagerank,
            "degree_centrality": degree_centrality,
            "betweenness_centrality": betweenness_centrality,
            "communities": communities,
            "summary_df": df_summary,
            "bot_count": bot_count,
            "bot_ratio": bot_ratio,
            "diffusion_depth": diffusion_depth,
        }
