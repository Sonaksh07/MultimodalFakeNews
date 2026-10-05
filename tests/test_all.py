import pytest
import numpy as np
import pandas as pd
from datasets.demo_data import get_demo_df
from datasets.dataset_loader import DatasetLoader
from models.nlp.tfidf_model import TFIDFBaselineClassifier
from models.nlp.bert_model import BERTFakeNewsClassifier
from models.cv.phash_detector import PerceptualHashDetector
from models.cv.ela_detector import ELADetector
from models.gnn.graph_builder import SocialGraphBuilder
from models.gnn.gat_model import GATPropagationPipeline
from models.fusion.weighted_fusion import WeightedMultimodalFusionEngine

def test_demo_dataset():
    df = get_demo_df()
    assert len(df) > 0
    assert "text" in df.columns
    assert "label" in df.columns

def test_dataset_loader():
    loader = DatasetLoader()
    df = loader.load_dataset("demo")
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0

def test_tfidf_classifier():
    model = TFIDFBaselineClassifier()
    texts = ["Government approves clean solar grid", "Alien pyramid found on moon"]
    probs = model.predict_proba(texts)
    assert probs.shape == (2, 2)
    assert np.allclose(np.sum(probs, axis=1), 1.0)

def test_bert_classifier():
    model = BERTFakeNewsClassifier()
    texts = ["Government approves clean solar grid", "Alien pyramid found on moon"]
    probs = model.predict_proba(texts)
    assert probs.shape == (2, 2)
    assert np.allclose(np.sum(probs, axis=1), 1.0)

def test_phash_detector():
    detector = PerceptualHashDetector()
    df = get_demo_df()
    img_path = df["image_path"].iloc[0]
    p_hash = detector.compute_phash(img_path)
    assert isinstance(p_hash, str)
    assert len(p_hash) > 0

def test_ela_detector():
    detector = ELADetector()
    df = get_demo_df()
    img_path = df["image_path"].iloc[0]
    _, score, is_manip, caveat = detector.compute_ela(img_path)
    assert 0.0 <= score <= 1.0
    assert isinstance(is_manip, bool)

def test_graph_builder_and_gat():
    builder = SocialGraphBuilder()
    G = builder.build_synthetic_propagation_graph(num_nodes=15)
    assert G.number_of_nodes() == 15
    assert G.number_of_edges() > 0

    gat = GATPropagationPipeline()
    p_graph, node_imp = gat.predict_graph_probability(G)
    assert 0.0 <= p_graph <= 1.0
    assert isinstance(node_imp, dict)

def test_weighted_fusion():
    fusion = WeightedMultimodalFusionEngine()
    res = fusion.fuse(p_text=0.8, p_image=0.7, p_graph=0.6)
    assert "final_score" in res
    assert res["final_label"] in ["REAL", "FAKE"]
    assert 0.0 <= res["confidence_percent"] <= 100.0
