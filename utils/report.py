"""Génération de rapports en Markdown puis conversion en HTML."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import markdown
from jinja2 import Template


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
