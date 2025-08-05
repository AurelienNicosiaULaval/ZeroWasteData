"""Génération de rapports en Markdown puis conversion en HTML."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import markdown
from jinja2 import Template
from fpdf import FPDF


DEFAULT_TEMPLATE = """\
# Rapport d'analyses

{% for title, content in sections.items() %}
## {{ title }}
{{ content }}

{% endfor %}
"""


def generate_report(sections: Dict[str, str], template: str | None = None) -> str:
    """Génère un rapport HTML à partir d'un dictionnaire de sections.

    Parameters
    ----------
    sections : dict
        Mapping `titre -> contenu Markdown`.
    template : str, optional
        Template Markdown à utiliser. Par défaut, un template minimal est utilisé.

    Returns
    -------
    str
        Rapport converti en HTML.
    """
    template_str = template or DEFAULT_TEMPLATE
    md_text = Template(template_str).render(sections=sections)
    return markdown.markdown(md_text, extensions=["tables"])


def generate_pdf_report(sections: Dict[str, str]) -> bytes:
    """Génère un rapport PDF à partir d'un dictionnaire de sections.

    Parameters
    ----------
    sections : dict
        Mapping `titre -> contenu Markdown`.

    Returns
    -------
    bytes
        Fichier PDF généré.
    """
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Rapport d'analyses", ln=True)
    for title, content in sections.items():
        pdf.ln(5)
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, title, ln=True)
        pdf.set_font("Arial", size=12)
        for line in content.splitlines():
            pdf.multi_cell(0, 8, line)
    return pdf.output(dest="S").encode("latin-1")
