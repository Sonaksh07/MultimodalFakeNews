import streamlit as st
from database.db import SessionLocal
from database.models import PredictionRecord
from utils.pdf_report import PDFReportGenerator
from dashboard.state import get_current_analysis
from dashboard.components.ui_styles import render_empty_state


def _build_report_dict_from_analysis(analysis) -> dict:
    from models.reasoning.evidence_engine import build_evidence_list, build_reasoning_paragraph

    res = analysis.fusion_result
    image_summary = None
    if analysis.has_image:
        image_summary = {
            "is_manipulated": analysis.ela_is_manipulated,
            "ela_score": analysis.ela_score,
            "ela_available": analysis.ela_score is not None,
            "phash": analysis.phash_value,
            "forensic_narrative": analysis.forensic_narrative,
        }
    return {
        "final_label": res["final_label"],
        "final_score": res["final_score"],
        "confidence_percent": res["confidence_percent"],
        "unimodal_scores": res["unimodal_scores"],
        "weights": res["weights"],
        "contributions": res["contributions"],
        "text_sample": analysis.text if analysis.has_text else "(no article text submitted)",
        "top_shap_tokens": (analysis.shap_explanation.get("top_tokens", []) if analysis.shap_explanation else []),
        "image_summary": image_summary,
        "propagation_summary": {
            "num_nodes": analysis.graph.number_of_nodes(),
            "num_edges": analysis.graph.number_of_edges(),
            "num_communities": len(set(analysis.graph_metrics["communities"].values())),
            "p_graph": analysis.p_graph,
            "bot_ratio": analysis.graph_metrics.get("bot_ratio", 0.0),
            "diffusion_depth": analysis.graph_metrics.get("diffusion_depth", 0),
        },
        "evidence_list": build_evidence_list(analysis),
        "reasoning": build_reasoning_paragraph(analysis),
        "processing_time_seconds": analysis.processing_time_seconds,
        "analyzed_at": analysis.analyzed_at,
    }


def _build_report_dict_from_record(rec) -> dict:
    return {
        "final_label": rec.final_label,
        "final_score": rec.final_score,
        "confidence_percent": rec.confidence,
        "unimodal_scores": {"text_score": rec.text_score, "image_score": rec.image_score, "graph_score": rec.graph_score},
        "weights": {"w_text": rec.text_weight, "w_image": rec.image_weight, "w_graph": rec.graph_weight},
        "contributions": {
            "text_contribution": round((rec.text_weight * rec.text_score) / (rec.final_score + 1e-8) * 100, 1),
            "image_contribution": round((rec.image_weight * rec.image_score) / (rec.final_score + 1e-8) * 100, 1),
            "graph_contribution": round((rec.graph_weight * rec.graph_score) / (rec.final_score + 1e-8) * 100, 1),
        },
        "text_sample": rec.text_content or "(no article text submitted)",
    }


def render():
    st.title("Reports")
    st.caption("Generate a downloadable diagnostic PDF report for the current or a past analysis.")

    analysis = get_current_analysis()
    if analysis:
        st.markdown("#### Current analysis")
        st.markdown(f"**{analysis.title}** — {analysis.fusion_result['final_label']} "
                    f"({analysis.fusion_result['final_score']:.4f})")
        if st.button("Generate PDF for current analysis", type="primary"):
            generator = PDFReportGenerator()
            pred_dict = _build_report_dict_from_analysis(analysis)
            pdf_path = generator.generate_pdf_report(pred_dict, f"Report_current_{analysis.prediction_db_id or 'session'}.pdf")

            if analysis.prediction_db_id:
                db = SessionLocal()
                rec = db.query(PredictionRecord).filter_by(id=analysis.prediction_db_id).first()
                if rec:
                    rec.report_status = "generated"
                    db.commit()
                db.close()

            with open(pdf_path, "rb") as f:
                st.download_button(
                    label="Download PDF report", data=f,
                    file_name="FakeNews_Report_Current.pdf", mime="application/pdf",
                )
        st.markdown("---")
    else:
        render_empty_state("&#128196;", "No analysis yet.", "Run an analysis in Live Detection first.")
        st.markdown("---")

    db = SessionLocal()
    records = db.query(PredictionRecord).order_by(PredictionRecord.created_at.desc()).all()
    db.close()

    if records:
        st.subheader("Prediction audit history")
        rec_data = [{
            "ID": r.id, "Title": r.title, "Label": r.final_label,
            "Risk Score": f"{r.final_score:.4f}", "Confidence": f"{r.confidence:.2f}%",
            "Report": r.report_status, "Timestamp": r.created_at,
        } for r in records]
        st.dataframe(rec_data, width="stretch")

        selected_id = st.number_input(
            "Prediction ID to generate a PDF report for",
            min_value=1, max_value=max(r.id for r in records), value=records[0].id,
        )

        if st.button("Generate PDF for selected ID", type="secondary"):
            rec = next((r for r in records if r.id == selected_id), None)
            if rec:
                generator = PDFReportGenerator()
                pdf_path = generator.generate_pdf_report(_build_report_dict_from_record(rec), f"Report_{rec.id}.pdf")

                db2 = SessionLocal()
                db_rec = db2.query(PredictionRecord).filter_by(id=rec.id).first()
                if db_rec:
                    db_rec.report_status = "generated"
                    db2.commit()
                db2.close()

                with open(pdf_path, "rb") as f:
                    st.download_button(
                        label="Download PDF report", data=f,
                        file_name=f"FakeNews_Report_ID_{rec.id}.pdf", mime="application/pdf",
                    )
    else:
        st.caption("No prediction history recorded yet.")
