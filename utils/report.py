from fpdf import FPDF
import tempfile
import os


def generate_report(sections: dict) -> str:
    """Génère un rapport HTML simple."""
    html = "<h1>Rapport d'Analyse ZeroWasteData</h1>"
    for title, content in sections.items():
        html += f"<h2>{title}</h2>"
        if isinstance(content, dict):
            html += f"<p>{content['text']}</p>"
        else:
            html += f"<p>{content}</p>"
    return html


def generate_pdf_report(sections: dict) -> bytes:
    """Génère un rapport PDF."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Rapport d'Analyse ZeroWasteData", ln=True, align="C")
    pdf.ln(10)

    pdf.set_font("Arial", "", 12)

    for title, content in sections.items():
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, title, ln=True)
        pdf.set_font("Arial", "", 12)

        text = ""
        plot = None

        if isinstance(content, dict):
            text = content.get("text", "")
            plot = content.get("plot")
        else:
            text = str(content)

        # Write text (handle encoding)
        pdf.multi_cell(0, 10, text.encode("latin-1", "replace").decode("latin-1"))
        pdf.ln(5)

        # Add plot if available
        if plot:
            try:
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    plot.save(tmp.name, width=6, height=4, dpi=150)
                    pdf.image(tmp.name, w=170)
                    pdf.ln(5)
                os.unlink(tmp.name)
            except Exception as e:
                pdf.cell(0, 10, f"[Erreur lors de l'ajout du graphique : {e}]", ln=True)

    return pdf.output(dest="S").encode("latin-1")
