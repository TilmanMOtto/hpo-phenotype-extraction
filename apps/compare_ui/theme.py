"""Chrome, re-exported -- plus the two vocabularies this app adds and nobody else has.

Nothing here is new design. ``apps.ui_common.theme`` is the source of truth for the palette, the
components and the plot layout, and five UIs that look and read the same are one UI a reader has to
learn once. The palette is also a validated CVD-separated set, so it is re-exported and never
extended by hand -- a new hue would be a new meaning nobody defined.

What this app does add is two mappings from *its* vocabulary to roles that already exist:

:data:`OUTCOME_TONE`
    ``tp`` / ``fp`` / ``fn``. These are roles, not colours, which counts more here than anywhere
    else in the repo: a reader scanning five gutters is reading colour before text, and ``good``
    has to stay "this method got it" in dark mode and for someone who cannot separate red from
    green by hue. Every chip therefore carries a glyph as well -- the tone is a second channel, not
    the only one.

:data:`FAMILY_TONE`
    ``external`` published work against ``ours``. *not* good/bad: this is provenance,
    and colouring our own methods as "good" would be the app taking a side in the comparison it
    exists to let the reader make.
"""

from __future__ import annotations

from apps.ui_common.theme import (  # noqa: F401 - re-exported as this app's chrome API
    DARK, GRAPH_CONFIG, LIGHT, empty, note, palette, panel, plotly_layout, stat, stat_row,
)

#: Role per outcome. ``fn`` is ``warning`` and not ``critical``: a miss and a spurious prediction
#: are different failures, and a screen that paints them the same colour cannot be scanned for
#: either one.
OUTCOME_TONE = {
    "tp": "good",
    "fp": "critical",
    "fn": "warning",
}

#: The glyph beside every chip, so the outcome survives a monochrome screenshot and a reader who
#: does not separate the hues. These are the same three marks the layout sketch used.
OUTCOME_GLYPH = {
    "tp": "✓",     # check
    "fp": "✗",     # ballot X
    "fn": "–",     # en dash: the method said nothing, rather than said something wrong
}

OUTCOME_HELP = {
    "tp": "in the curated ground truth and predicted by this method",
    "fp": "predicted by this method and not in the curated ground truth",
    "fn": "in the curated ground truth and not predicted by this method",
}

#: Provenance, not quality. See the module docstring.
FAMILY_TONE = {"external": "neutral", "ours": "accent"}

FAMILY_HELP = {
    "external": "a published method, reproduced here",
    "ours": "a method of this thesis",
}

#: What the five reasoning payloads are called, and the one sentence each panel opens with. Keyed
#: by the ``kind`` an adapter stamps on its reason, so a payload whose renderer is missing still
#: says what it was.
KIND_TITLE = {
    "tagger": "Span tagger",
    "linker": "Retrieve, then link",
    "rag": "Retrieve, then verify",
    "jury": "Eight jurors, one vote",
    "tree": "Ontology walk",
}
