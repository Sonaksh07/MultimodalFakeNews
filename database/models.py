from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from utils.timezone import now_ist_naive

Base = declarative_base()

class PredictionRecord(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String(255), nullable=True)
    text_content = Column(Text, nullable=True)
    image_path = Column(String(512), nullable=True)
    image_filename = Column(String(255), nullable=True)

    # Unimodal prediction scores (0.0 = Real, 1.0 = Fake)
    text_score = Column(Float, nullable=False, default=0.5)
    image_score = Column(Float, nullable=False, default=0.5)
    graph_score = Column(Float, nullable=False, default=0.5)
    
    # Multimodal fused output
    final_score = Column(Float, nullable=False, default=0.5)
    final_label = Column(String(50), nullable=False, default="UNCERTAIN")  # "REAL" or "FAKE"
    confidence = Column(Float, nullable=False, default=0.0)
    
    # Modality contributions
    text_weight = Column(Float, default=0.40)
    image_weight = Column(Float, default=0.35)
    graph_weight = Column(Float, default=0.25)

    # Operational metadata (Phase 18)
    processing_time_seconds = Column(Float, nullable=True)
    report_status = Column(String(32), default="not_generated")  # not_generated | generated
    text_model_version = Column(String(128), nullable=True)
    image_model_version = Column(String(128), nullable=True)
    graph_model_version = Column(String(128), nullable=True)

    created_at = Column(DateTime, default=now_ist_naive)

class UploadedImage(Base):
    __tablename__ = "uploaded_images"

    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(512), nullable=False)
    phash = Column(String(64), nullable=True)
    ela_score = Column(Float, nullable=True)
    is_manipulated = Column(Boolean, default=False)
    is_reused = Column(Boolean, default=False)
    uploaded_at = Column(DateTime, default=now_ist_naive)

class GeneratedReport(Base):
    __tablename__ = "reports"

    id = Column(Integer, primary_key=True, autoincrement=True)
    report_name = Column(String(255), nullable=False)
    pdf_path = Column(String(512), nullable=False)
    prediction_id = Column(Integer, nullable=True)
    generated_at = Column(DateTime, default=now_ist_naive)
