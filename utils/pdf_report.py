import os
from pathlib import Path
from datetime import datetime
from utils.timezone import now_ist, now_ist_str
from configs.logging_config import logger

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    logger.warning("ReportLab library not installed. PDF report generation will use HTML fallback.")

class PDFReportGenerator:
    """Generates formal research-grade PDF reports for multimodal fake news analysis."""

    def __init__(self, output_dir: str = "reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_pdf_report(self, prediction_dict: dict, report_filename: str = None) -> str:
        """
        Builds downloadable PDF report detailing prediction, confidence, unimodal contributions, and XAI findings.
        Returns file path string.
        """
        if not report_filename:
            timestamp = now_ist().strftime("%Y%m%d_%H%M%S")
            report_filename = f"FakeNews_Report_{timestamp}.pdf"

        pdf_path = self.output_dir / report_filename

        if REPORTLAB_AVAILABLE:
            try:
                doc = SimpleDocTemplate(
                    str(pdf_path),
                    pagesize=letter,
                    rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36
                )
                styles = getSampleStyleSheet()

                title_style = ParagraphStyle(
                    'DocTitle',
                    parent=styles['Heading1'],
                    fontSize=20,
                    leading=24,
                    textColor=colors.HexColor("#1A252C"),
                    alignment=1
                )
                subtitle_style = ParagraphStyle(
                    'DocSubTitle',
                    parent=styles['Normal'],
                    fontSize=11,
                    leading=14,
                    textColor=colors.HexColor("#7F8C8D"),
                    alignment=1
                )
                heading_style = ParagraphStyle(
                    'DocHeading',
                    parent=styles['Heading2'],
                    fontSize=14,
                    leading=18,
                    textColor=colors.HexColor("#2980B9"),
                    spaceBefore=12,
                    spaceAfter=6
                )

                elements = []
                _section_num = [0]
                def _next_section(title):
                    _section_num[0] += 1
                    return f"{_section_num[0]}. {title}"

                # Header
                elements.append(Paragraph("Multimodal Fake News & Image Tampering Diagnostic Report", title_style))
                elements.append(Paragraph(f"Generated on {now_ist_str()} IST | Project Forensic Audit", subtitle_style))
                analyzed_at = prediction_dict.get("analyzed_at")
                if analyzed_at:
                    elements.append(Paragraph(f"Analysis performed at {analyzed_at} IST", subtitle_style))
                elements.append(Spacer(1, 15))

                # Section 1: Executive Summary
                elements.append(Paragraph(_next_section("Executive Summary & Verdict"), heading_style))

                label = prediction_dict.get("final_label", "UNCERTAIN")
                score = prediction_dict.get("final_score", 0.5)
                conf = prediction_dict.get("confidence_percent", 0.0)
                proc_time = prediction_dict.get("processing_time_seconds")

                label_color = "#E74C3C" if label == "FAKE" else "#2ECC71"

                summary_table_data = [
                    ["Final Classification Verdict", f"<font color='{label_color}'><b>{label}</b></font>"],
                    ["Overall Multimodal Risk Score", f"{score:.4f} (0.0=Real, 1.0=Fake)"],
                    ["Prediction Confidence", f"{conf:.2f}%"],
                ]
                if proc_time is not None:
                    summary_table_data.append(["Processing Time", f"{proc_time:.2f} seconds"])

                table1 = Table(
                    [[Paragraph(cell, styles['Normal']) for cell in row] for row in summary_table_data],
                    colWidths=[200, 300]
                )
                table1.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F8F9F9")),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#BDC3C7")),
                    ('PADDING', (0, 0), (-1, -1), 6),
                ]))
                elements.append(table1)
                elements.append(Spacer(1, 10))

                reasoning = prediction_dict.get("reasoning")
                if reasoning:
                    elements.append(Paragraph(reasoning, styles['Normal']))
                elements.append(Spacer(1, 15))

                # Section 2: Evidence Summary
                evidence_list = prediction_dict.get("evidence_list", [])
                if evidence_list:
                    elements.append(Paragraph(_next_section("Evidence Summary"), heading_style))
                    for item in evidence_list:
                        elements.append(Paragraph(f"&bull; {item}", styles['Normal']))
                    elements.append(Spacer(1, 15))

                # Section 3: Unimodal Decomposition
                elements.append(Paragraph(_next_section("Model Results & Fusion Weights"), heading_style))

                unimodal = prediction_dict.get("unimodal_scores", {})
                weights = prediction_dict.get("weights", {})
                contribs = prediction_dict.get("contributions", {})

                decomp_data = [
                    ["Modality", "Model Engine", "Risk Score", "Weight", "Relative Impact"],
                    ["Text Module", "TF-IDF / BERT Transformer", f"{unimodal.get('text_score', 0):.4f}", f"{weights.get('w_text', 0.4):.2f}", f"{contribs.get('text_contribution', 0):.1f}%"],
                    ["Image Module", "ResNet50 / ELA / pHash", f"{unimodal.get('image_score', 0):.4f}", f"{weights.get('w_image', 0.35):.2f}", f"{contribs.get('image_contribution', 0):.1f}%"],
                    ["Graph Module", "PyG Graph Attention (GAT)", f"{unimodal.get('graph_score', 0):.4f}", f"{weights.get('w_graph', 0.25):.2f}", f"{contribs.get('graph_contribution', 0):.1f}%"]
                ]

                table2 = Table(
                    [[Paragraph(cell, styles['Normal']) for cell in row] for row in decomp_data],
                    colWidths=[90, 150, 80, 70, 90]
                )
                table2.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#34495E")),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#BDC3C7")),
                    ('PADDING', (0, 0), (-1, -1), 5),
                ]))
                elements.append(table2)
                elements.append(Spacer(1, 15))

                # Section 4: Input Details
                elements.append(Paragraph(_next_section("Analyzed Article Content"), heading_style))
                text_sample = prediction_dict.get("text_sample", "N/A")
                if len(text_sample) > 400:
                    text_sample = text_sample[:400] + "..."
                elements.append(Paragraph(f"<i>\"{text_sample}\"</i>", styles['Italic']))
                elements.append(Spacer(1, 15))

                # Section 5: Text Explainability
                top_tokens = prediction_dict.get("top_shap_tokens", [])
                if top_tokens:
                    elements.append(Paragraph(_next_section("Text Explanation — Top Contributing Tokens"), heading_style))
                    token_str = ", ".join(top_tokens[:8])
                    elements.append(Paragraph(
                        f"SHAP/coefficient attribution identified the following tokens as most influential "
                        f"in the text-model verdict: <b>{token_str}</b>.", styles['Normal'],
                    ))
                    elements.append(Spacer(1, 15))

                # Section 6: Image Explanation
                img_summary = prediction_dict.get("image_summary")
                if img_summary:
                    elements.append(Paragraph(_next_section("Image Explanation"), heading_style))
                    if img_summary.get("ela_available", True):
                        manip_text = "flagged as likely manipulated" if img_summary.get("is_manipulated") else "showed no strong splicing signal"
                        elements.append(Paragraph(
                            f"Error Level Analysis {manip_text} (anomaly score "
                            f"{img_summary.get('ela_score', 0):.4f}). Perceptual hash: "
                            f"<font face='Courier'>{img_summary.get('phash', 'N/A')}</font>. "
                            f"A GradCAM attention heatmap localizing the regions most influential to the "
                            f"ResNet-50 verdict was generated for this image (see dashboard for the visual overlay).",
                            styles['Normal'],
                        ))
                    else:
                        elements.append(Paragraph(
                            "Error Level Analysis could not be computed for this image file; no ELA "
                            "verdict is reported rather than a fabricated one. Perceptual hash: "
                            f"<font face='Courier'>{img_summary.get('phash', 'N/A')}</font>.",
                            styles['Normal'],
                        ))
                    elements.append(Spacer(1, 15))
                    if img_summary.get("forensic_narrative"):
                        elements.append(Paragraph(img_summary["forensic_narrative"], styles['Normal']))
                        elements.append(Spacer(1, 15))

                # Section 7: Propagation Summary
                prop_summary = prediction_dict.get("propagation_summary")
                if prop_summary:
                    elements.append(Paragraph(_next_section("Propagation Analysis Summary"), heading_style))
                    elements.append(Paragraph(
                        f"Simulated propagation graph: {prop_summary.get('num_nodes', 0)} nodes, "
                        f"{prop_summary.get('num_edges', 0)} cascade edges, "
                        f"{prop_summary.get('num_communities', 0)} detected communities, "
                        f"diffusion depth {prop_summary.get('diffusion_depth', 0)}, "
                        f"{prop_summary.get('bot_ratio', 0):.0%} suspicious/bot-like accounts. "
                        f"Graph Attention Network propagation risk score: "
                        f"{prop_summary.get('p_graph', 0):.4f}.", styles['Normal'],
                    ))
                    elements.append(Spacer(1, 15))

                # Section 8: Fusion Explanation
                elements.append(Paragraph(_next_section("Fusion Analysis"), heading_style))
                elements.append(Paragraph(
                    "The final verdict is a weighted linear combination of the unimodal risk scores "
                    "(see Section 3), calibrated via Platt scaling fit on the project's labeled demo "
                    "corpus. Any modality not submitted for this analysis defaults to a neutral 0.5 score "
                    "rather than being fabricated. The modality with the largest 'Relative Impact' in the "
                    "table above contributed most to the final classification.", styles['Normal'],
                ))
                elements.append(Spacer(1, 15))

                # Section 9: Final Decision & Recommendations
                elements.append(Paragraph(_next_section("Final Decision & Recommendations"), heading_style))
                if label == "FAKE":
                    recommendation = (
                        "This content shows characteristics consistent with misinformation. "
                        "Before sharing or acting on it: verify the claim against at least one "
                        "established fact-checking organization or primary source, check whether "
                        "the accompanying image appears elsewhere in a different context (reverse "
                        "image search), and treat unattributed claims and emotionally-charged "
                        "language as a caution signal."
                    )
                else:
                    recommendation = (
                        "This content does not show strong indicators of fabrication based on the "
                        "signals available to this system. As with any automated assessment, "
                        "cross-verification against primary sources is still recommended before "
                        "using this content for high-stakes decisions."
                    )
                elements.append(Paragraph(recommendation, styles['Normal']))

                doc.build(elements)
                logger.info(f"ReportLab PDF report created successfully at {pdf_path}")
                return str(pdf_path)

            except Exception as e:
                logger.error(f"Failed to generate PDF with ReportLab: {e}")

        # Fallback text report creation if ReportLab is uninstalled
        with open(pdf_path, "w", encoding="utf-8") as f:
            f.write(f"MULTIMODAL FAKE NEWS REPORT\n")
            f.write(f"Generated: {now_ist_str()} IST\n")
            f.write(f"Verdict: {prediction_dict.get('final_label')}\n")
            f.write(f"Score: {prediction_dict.get('final_score')}\n")
        return str(pdf_path)
