r"""The thesis's colour system. Every figure script takes its colours from here and nowhere else.

Colour carries meaning, and a meaning keeps its colour from figure to figure. Consistency is
strictest where a reader compares across figures -- the methods, and the two decisions of
chapter 4 -- and loosest in chapter 3's setup schematics, which only have to be clear on their own.

1. **A method owns one hue** wherever methods are compared (chapter 6 set the assignment), and a
   method's own data series in its chapter use it too:

       PhenoBERT     mid grey        the reference baseline
       RAG-HPO       blue            8B light, 70B dark
       AutoPCR       amber           8B light, 70B dark
       TreePhenoRAG  reddish purple  chapter 4
       PhenoJury     bluish green    chapter 5

2. **Roles inside the pipelines** (chapters 4 and 5), fixed wherever they appear:

       green       accepted / predicted / annotated -- the positive outcome
       vermillion  pruned / rejected / not annotated / wrong -- the negative outcome
       purple      expansion (TreePhenoRAG's tree walk. Acceptance is green)
       blue        verifier evidence: segment scores, the ontology index
       amber       what changed, the synthetic-sentence index, the report sentence the jurors read

3. **Error breakdowns share one vermillion ramp**, so the FP and FN bars of chapters 4 and 5
   read against each other.

4. **Black and grey are structure, not colours**: text, axes, reference lines, context. No data
   category is drawn in black. Grey marks only a baseline or context.

Everything is Okabe-Ito, a tint or shade of it, or (for one ordered axis) viridis, so the palette
stays colour-blind safe. ``make_all.py`` fails the build on any hex literal in a figure script
outside this module.
"""
from __future__ import annotations


def _mix(color: str, other: str, t: float) -> str:
    """``color`` moved a fraction ``t`` of the way towards ``other``, in sRGB."""
    a = [int(color[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(other[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02X}" for x, y in zip(a, b))


def tint(color: str, t: float) -> str:
    """Towards white: ``t=0`` is the colour itself, ``t=1`` is white. Schematic fills use ~0.8."""
    return _mix(color, "#FFFFFF", t)


def shade(color: str, t: float) -> str:
    """Towards black: the text colour on a tint of the same hue uses ~0.45."""
    return _mix(color, "#000000", t)


def text_on(fill: str) -> str:
    """White or ink, whichever reads on ``fill`` (WCAG relative luminance, threshold 0.4)."""
    def lin(c):
        c /= 255
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = (lin(int(fill[i:i + 2], 16)) for i in (1, 3, 5))
    return "#FFFFFF" if 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.4 else INK


# ── Okabe-Ito, by colour name ────────────────────────────────────────────────────────────────
BLUE = "#0072B2"
SKY = "#56B4E9"
AMBER = "#E69F00"
GREEN = "#009E73"
VERMILLION = "#D55E00"
PURPLE = "#CC79A7"          # Okabe-Ito's "reddish purple"
YELLOW = "#F0E442"

# ── Neutrals (rule 4): structure only ────────────────────────────────────────────────────────
INK = "#222222"        # text, reference lines
DARK = "#555555"       # secondary text, connectors, box edges
MID = "#7F7F7F"        # The baseline grey: PhenoBERT, V0, the condition not selected. Muted notes
LIGHT = "#BFBFBF"      # rules, CI bands, inactive outlines, context curves
FILL = "#EFEFEF"       # schematic box fill, empty-bar track
WHITE = "#FFFFFF"

BASELINE = MID

# ── Methods (rule 1) ─────────────────────────────────────────────────────────────────────────
PHENOBERT = MID
RAGHPO, RAGHPO_LIGHT = BLUE, SKY
AUTOPCR, AUTOPCR_LIGHT = AMBER, "#F0C476"
TREE = PURPLE
JURY = GREEN

JURY_LIGHT = "#66C2A5"

# ── Pipeline roles (rule 2) ──────────────────────────────────────────────────────────────────
POSITIVE = GREEN       # accepted, predicted, annotated
NEGATIVE = VERMILLION  # pruned, rejected, not annotated
EXPANSION = PURPLE
ACCEPTANCE = GREEN
EVIDENCE = BLUE        # verifier segment scores, the ontology index
CHANGED = AMBER        # what a variant changes. Also the synthetic-sentence index and the report sentence

# An ordered axis with three steps (PhenoJury's vote scope, coarse -> fine): three stops of
# viridis, a green and a blue with the teal between them, so the steps are far enough apart to
# read as three lines and still read as one progression.
ORDERED_3 = ("#5EC962", "#21918C", "#3B528B")

# ── Error (rule 3) ───────────────────────────────────────────────────────────────────────────
ERROR = VERMILLION
ERROR_DARK = shade(ERROR, 0.40)     # text on an error tint. The far end of the ramp


def error_ramp(n: int) -> list[str]:
    """``n`` steps from a light vermillion tint to a dark shade, for ordered error categories.

    Light is the mildest (nearest the right answer, earliest in the pipeline), dark the most
    severe. Hatches still distinguish neighbours. The ramp only orders them.
    """
    if n == 1:
        return [ERROR]
    stops = [tint(ERROR, 0.72), tint(ERROR, 0.40), ERROR, shade(ERROR, 0.25), shade(ERROR, 0.50)]
    if n <= len(stops):
        # Spread n picks across the five stops, always keeping both ends.
        return [stops[round(i * (len(stops) - 1) / (n - 1))] for i in range(n)]
    raise ValueError(f"error_ramp supports at most {len(stops)} categories, got {n}")


# ── Schematic box styles (rule 6): fill, edge, text ──────────────────────────────────────────

def box(hue: str, *, fill: float = 0.80, text: float = 0.45) -> dict:
    """A schematic box in ``hue``: a light tint, an edge in the hue, text in a dark shade of it."""
    return dict(fc=tint(hue, fill), ec=hue, tc=shade(hue, text))


PLAIN_BOX = dict(fc=WHITE, ec=DARK, tc=INK)
NEUTRAL_BOX = dict(fc=FILL, ec=MID, tc=INK)
MUTED_BOX = dict(fc=WHITE, ec=LIGHT, tc=MID)

# Lightness steps of TreePhenoRAG's hue, light -> dark, for its ordered variants (the ladder rungs).
TREE_RAMP = (tint(TREE, 0.45), TREE, shade(TREE, 0.35))
