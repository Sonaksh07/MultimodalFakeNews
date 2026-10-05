import os
import shutil
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy.orm import Session

from database.db import init_db, get_db
from database.models import PredictionRecord, UploadedImage, GeneratedReport
from datasets.dataset_loader import DatasetLoader
from models.nlp.bert_model import BERTFakeNewsClassifier
from models.nlp.tfidf_model import TFIDFBaselineClassifier
from models.nlp.nlp_evaluator import NLPEvaluator
from models.cv.phash_detector import PerceptualHashDetector
from models.cv.ela_detector import ELADetector
from models.cv.resnet_model import ResNet50ImageClassifier
from models.gnn.graph_builder import SocialGraphBuilder
from models.gnn.graph_metrics import GraphAnalyticsEngine
from models.gnn.gat_model import GATPropagationPipeline
from models.fusion.weighted_fusion import WeightedMultimodalFusionEngine
from models.xai.explainer import ExplainableAIEngine
from backend.news_api import NewsAPIClient
from utils.pdf_report import PDFReportGenerator
from configs.logging_config import logger

# Initialize database
init_db()

app = FastAPI(
    title="Big Data Multimodal Fake News Detection System API",
    description="Production-grade RESTful API for text, image forgery, and graph social propagation analysis.",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared Pipeline Engines
bert_model = BERTFakeNewsClassifier()
tfidf_model = TFIDFBaselineClassifier()
nlp_evaluator = NLPEvaluator()
phash_detector = PerceptualHashDetector()
ela_detector = ELADetector()
resnet_model = ResNet50ImageClassifier()
graph_builder = SocialGraphBuilder()
graph_metrics_engine = GraphAnalyticsEngine()
gat_pipeline = GATPropagationPipeline()
fusion_engine = WeightedMultimodalFusionEngine()
xai_engine = ExplainableAIEngine()
news_client = NewsAPIClient()
pdf_generator = PDFReportGenerator()
dataset_loader = DatasetLoader()

from models.fusion.calibration_bootstrap import ensure_calibrator_fitted
ensure_calibrator_fitted({
    "bert": bert_model, "resnet": resnet_model, "builder": graph_builder,
    "gat": gat_pipeline, "fusion": fusion_engine,
})

# Pydantic Schemas
class TextPredictionRequest(BaseModel):
    text: str
    title: Optional[str] = ""

class MultimodalPredictionRequest(BaseModel):
    text: str
    image_path: Optional[str] = ""
    title: Optional[str] = ""

@app.get("/")
def root():
    return {
        "status": "online",
        "system": "Multimodal Fake News & Image Tampering Detection Engine",
        "docs_url": "/docs"
    }

@app.post("/api/v1/predict/text")
def predict_text(req: TextPredictionRequest, db: Session = Depends(get_db)):
    """Predicts fake news probability for a text passage."""
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text content cannot be empty.")

    p_bert = float(bert_model.predict_proba([req.text])[0, 1])
    p_tfidf = float(tfidf_model.predict_proba([req.text])[0, 1])

    # Fusion for text only
    fusion = fusion_engine.fuse(p_text=p_bert, p_image=0.5, p_graph=0.5)

    rec = PredictionRecord(
        title=req.title or "API Text Request",
        text_content=req.text,
        text_score=p_bert,
        image_score=0.5,
        graph_score=0.5,
        final_score=fusion["final_score"],
        final_label=fusion["final_label"],
        confidence=fusion["confidence_percent"]
    )
    db.add(rec)
    db.commit()

    return {
        "text_bert_score": p_bert,
        "text_tfidf_score": p_tfidf,
        "fusion_result": fusion
    }

    # 2. MultiModal
@app.post("/api/v1/predict/multimodal")
def predict_multimodal(
    text: str = Form(""),
    title: Optional[str] = Form(""),
    file: Optional[UploadFile] = File(None),
    db: Session = Depends(get_db)
):
    """Executes multimodal prediction. Accepts text only, image only, or both."""

    text = (text or "").strip()
    has_text = len(text) > 0
    if not has_text and file is None:
        raise HTTPException(status_code=400, detail="Provide article text, an image, or both.")

    # ---------------- TEXT ----------------
    if has_text:
        p_text = float(bert_model.predict_proba([text])[0, 1])
    else:
        p_text = 0.5  # neutral default -- no text submitted, not a fabricated score

    # ---------------- IMAGE ----------------
    image_path = None
    p_image = 0.5
    phash_val = None

    if file is not None:

        os.makedirs("uploads", exist_ok=True)

        image_path = os.path.join("uploads", file.filename)

        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        p_image = float(resnet_model.predict_proba(image_path)[1])

        phash_val = phash_detector.compute_phash(image_path)
        _, ela_score, is_manip, _ela_caveat = ela_detector.compute_ela(image_path)
        if ela_score is None:
            ela_score, is_manip = 0.0, False

        uploaded = UploadedImage(
            filename=file.filename,
            file_path=image_path,
            phash=phash_val,
            ela_score=ela_score,
            is_manipulated=is_manip
        )

        db.add(uploaded)

    # ---------------- GRAPH ----------------
    # Falls back to the image's perceptual hash as the graph seed when no
    # text was submitted, so image-only requests still get a graph specific
    # to that image rather than a generic default.
    graph_seed_text = text if has_text else (phash_val or (file.filename if file else title or "untitled"))
    G = graph_builder.build_propagation_graph_for_article(graph_seed_text, risk_score=(p_text if has_text else p_image))
    p_graph, node_imp = gat_pipeline.predict_graph_probability(G)

    # ---------------- FUSION ----------------
    fusion_result = fusion_engine.fuse(
        p_text=p_text,
        p_image=p_image,
        p_graph=p_graph
    )

    # ---------------- SAVE ----------------
    rec = PredictionRecord(
        title=title or "Multimodal Request",
        text_content=text,
        image_path=image_path,
        text_score=p_text,
        image_score=p_image,
        graph_score=p_graph,
        final_score=fusion_result["final_score"],
        final_label=fusion_result["final_label"],
        confidence=fusion_result["confidence_percent"]
    )

    db.add(rec)
    db.commit()
    db.refresh(rec)

    return {
        "prediction_id": rec.id,
        "fusion_result": fusion_result,
        "xai_text_shap": xai_engine.explain_text_shap(text, tfidf_model)["top_tokens"] if has_text else [],
    }

@app.get("/api/v1/dataset")
def get_dataset_info(name: str = "demo"):
    """Returns dataset overview statistics."""
    df = dataset_loader.load_dataset(name)
    return {
        "dataset_name": name,
        "total_records": len(df),
        "columns": list(df.columns),
        "sample_records": df.head(5).to_dict(orient="records")
    }

@app.get("/api/v1/analytics/comparison")
def get_model_comparison(sample_size: int = 20):
    """Returns comparative benchmarking of NLP (TF-IDF vs BERT) and Modality Fusion,
    computed from REAL model inference over a sample of the demo corpus."""
    df_demo = dataset_loader.load_dataset("demo")
    nlp_comp = nlp_evaluator.evaluate_all(df_demo["text"].tolist(), df_demo["label"].tolist())

    # Modality Combos -- real inference, not label-derived synthetic scores.
    df_sample = df_demo.sample(n=min(sample_size, len(df_demo)), random_state=42)
    samples = []
    for _, row in df_sample.iterrows():
        text = row["text"]
        image_path = row.get("image_path")
        p_text = float(bert_model.predict_proba([text])[0, 1])
        p_image = float(resnet_model.predict_proba(image_path)[1]) if image_path else 0.5
        G = graph_builder.build_propagation_graph_for_article(text, risk_score=p_text, num_nodes=20)
        p_graph, _ = gat_pipeline.predict_graph_probability(G)
        samples.append({"p_text": p_text, "p_image": p_image, "p_graph": p_graph, "y_true": row["label"]})
    fusion_comp = fusion_engine.evaluate_all_combinations(samples)

    return {
        "nlp_models_comparison": nlp_comp,
        "modality_fusion_comparison": fusion_comp,
        "ablation_sample_size": len(samples),
    }

@app.get("/api/v1/news")
def fetch_live_news(query: str = "technology"):
    """Fetches top live news articles."""
    articles = news_client.fetch_top_headlines(query=query)
    return {"articles": articles}

@app.post("/api/v1/reports/generate")
def generate_report(prediction_id: int, db: Session = Depends(get_db)):
    """Generates PDF report for a given prediction ID."""
    rec = db.query(PredictionRecord).filter(PredictionRecord.id == prediction_id).first()
    if not rec:
        raise HTTPException(status_code=404, detail="Prediction record not found.")

    pred_dict = {
        "final_label": rec.final_label,
        "final_score": rec.final_score,
        "confidence_percent": rec.confidence,
        "unimodal_scores": {
            "text_score": rec.text_score,
            "image_score": rec.image_score,
            "graph_score": rec.graph_score
        },
        "weights": {
            "w_text": rec.text_weight,
            "w_image": rec.image_weight,
            "w_graph": rec.graph_weight
        },
        "contributions": {
            "text_contribution": round((rec.text_weight * rec.text_score) / (rec.final_score + 1e-8) * 100, 1),
            "image_contribution": round((rec.image_weight * rec.image_score) / (rec.final_score + 1e-8) * 100, 1),
            "graph_contribution": round((rec.graph_weight * rec.graph_score) / (rec.final_score + 1e-8) * 100, 1),
        },
        "text_sample": rec.text_content
    }

    pdf_file = pdf_generator.generate_pdf_report(pred_dict, f"Report_Prediction_{rec.id}.pdf")
    
    rep = GeneratedReport(
        report_name=f"Report_Prediction_{rec.id}.pdf",
        pdf_path=pdf_file,
        prediction_id=rec.id
    )
    db.add(rep)
    db.commit()

    return {"status": "success", "pdf_path": pdf_file}
