import hashlib
import networkx as nx
import numpy as np
import pandas as pd
from configs.logging_config import logger


class SocialGraphBuilder:
    """Constructs multi-relational social propagation graphs from user
    interactions (retweets, shares, replies).

    NOTE ON REALISM: this project has no live connection to a real social
    media platform's interaction data (no Twitter/X firehose, no
    FakeNewsNet propagation dump bundled with the repo), so there is no
    real per-article spread history to visualize. Rather than pretend
    otherwise, `build_propagation_graph_for_article()` below generates a
    *simulated* propagation cascade whose structure (bot ratio, spread
    fan-out, edge density) is deterministically derived from the current
    article's text and its text-model risk score -- so different articles
    genuinely produce different, reproducible graphs correlated with their
    content, instead of one frozen synthetic graph that never changes.
    """

    def __init__(self):
        pass

    def build_propagation_graph_for_article(
        self, article_text: str, risk_score: float = 0.5, num_nodes: int = 45,
    ) -> nx.DiGraph:
        """Builds a propagation graph whose seed and spread parameters are
        derived from the given article's text + predicted risk score, so
        the SAME article always produces the SAME graph, and DIFFERENT
        articles produce visibly different graphs.

        Higher `risk_score` (i.e. more likely fake) biases the simulation
        toward higher bot involvement and faster, wider fan-out -- modeling
        the well-documented empirical finding that misinformation tends to
        spread faster and via more automated accounts than verified news
        (Vosoughi et al., 2018).
        """
        text_hash = hashlib.sha256((article_text or "").encode("utf-8")).hexdigest()
        seed = int(text_hash[:8], 16) % (2**31)
        rng = np.random.default_rng(seed)

        risk_score = float(np.clip(risk_score, 0.0, 1.0))
        bot_bias = 0.15 + 0.5 * risk_score          # higher risk -> more bots
        fanout_bias = 3 + int(round(4 * risk_score))  # higher risk -> wider spread

        G = nx.DiGraph()
        G.add_node(0, label="RootArticle", type="article", bot_score=0.0)

        for i in range(1, num_nodes):
            bot_score = float(rng.beta(0.5 + bot_bias, 0.5)) if rng.random() < (0.15 + 0.25 * risk_score) else float(rng.uniform(0.01, 0.3))
            user_type = "suspicious_bot" if bot_score > 0.65 else ("influencer" if i < 6 else "user")
            G.add_node(i, label=f"User_{i}", type=user_type, bot_score=bot_score)

        num_initial_spreaders = min(4 + fanout_bias, num_nodes - 1)
        for i in rng.choice(range(1, num_nodes), size=num_initial_spreaders, replace=False):
            G.add_edge(0, int(i), relation="retweet", weight=float(rng.uniform(0.8, 1.0)))

        for i in range(1, num_nodes):
            possible_targets = [t for t in range(1, num_nodes) if t != i]
            num_edges = rng.integers(1, max(2, fanout_bias - 1))
            targets = rng.choice(possible_targets, size=min(num_edges, len(possible_targets)), replace=False)
            for t in targets:
                rel = rng.choice(["retweet", "reply", "share", "mention"], p=[0.5, 0.25, 0.15, 0.1])
                G.add_edge(i, int(t), relation=str(rel), weight=float(rng.uniform(0.1, 0.9)))

        logger.info(
            f"Built article-conditioned propagation graph (seed={seed}, risk={risk_score:.2f}): "
            f"{G.number_of_nodes()} nodes, {G.number_of_edges()} edges."
        )
        return G

    def build_synthetic_propagation_graph(self, num_nodes: int = 45, seed: int = 42) -> nx.DiGraph:
        """Legacy fixed-seed synthetic graph, kept for pages that want a
        generic illustrative network not tied to a specific article (e.g.
        exploratory graph-analytics demos)."""
        return self.build_propagation_graph_for_article(
            article_text=f"__legacy_fixed_seed_{seed}__", risk_score=0.5, num_nodes=num_nodes,
        )

    def build_graph_from_edges(self, edges_df: pd.DataFrame) -> nx.DiGraph:
        """Constructs Directed Graph from a pandas dataframe of edges (source, target, relation)."""
        G = nx.DiGraph()
        for _, row in edges_df.iterrows():
            G.add_edge(row["source"], row["target"], relation=row.get("relation", "retweet"))
        logger.info(f"Loaded graph from edges dataframe: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
        return G
