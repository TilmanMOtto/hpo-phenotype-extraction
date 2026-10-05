r"""Read the thesis the way LaTeX does: which sections compile, what they ``\input``, what they typeset.

Standard library only, so ``make_all.py`` can audit the thesis without importing a plotting stack.
It fails the build on a float typeset in a section under a label that a generated file also
carries: a pasted copy of a generated table, which is how thirteen tables drifted from their CSVs
before 2026-09-28 (a changed value, missing columns, stale bold). It also lists generated floats the
thesis never inputs. (The appendix table of generated sources it once fed, tab_app_sources, was
removed from the thesis on 2026-09-28.)

Only the sections ``neurips_2025.tex`` actually ``\input``s count. ``results.tex`` and the
commented-out ``05_PhenoJury`` are on disk but not in the PDF, so a float there is nobody's problem.
"""
from __future__ import annotations

import os
import re
from hpo_extraction.paths import thesis_dir  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
THESIS = str(thesis_dir())
MAIN_TEX = os.path.join(THESIS, "neurips_2025.tex")
TEX_DIR = os.path.join(THESIS, "thesis_figures_latex")

# An unescaped % to the end of the line. `95\,\%` is text, not a comment. Treating it as one
# silently truncates every caption with a confidence level in it.
_COMMENT = re.compile(r"(?<!\\)%[^\n]*")
_SECTION_INPUT = re.compile(r"\\input\{sections/([^}]+)\}")
_GENERATED_INPUT = re.compile(r"\\input\{thesis_figures_latex/([^}]+)\}")
_LABEL = re.compile(r"\\label\{([^}]+)\}")
_FLOAT = re.compile(r"\\begin\{(table|figure)\*?\}.*?\\end\{\1\*?\}", re.S)


def strip_comments(text: str) -> str:
    """*text* with LaTeX comments removed."""
    return _COMMENT.sub("", text)


def _read(path: str) -> str:
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def section_files() -> list[str]:
    """The section files the main document compiles, in order, as paths."""
    names = _SECTION_INPUT.findall(strip_comments(_read(MAIN_TEX)))
    return [os.path.join(THESIS, "sections", n if n.endswith(".tex") else n + ".tex")
            for n in names]


def generated_inputs() -> list[tuple[str, str]]:
    r"""``(section file name, stem)`` for every ``\input{thesis_figures_latex/<stem>}``, in order.

    Inputs in the main file's preamble (the ``num_*`` macro files) are not floats and are not
    returned. They are the only generated inputs outside ``sections/``.
    """
    found = []
    for path in section_files():
        for stem in _GENERATED_INPUT.findall(strip_comments(_read(path))):
            found.append((os.path.basename(path), stem[:-4] if stem.endswith(".tex") else stem))
    return found


def labels_of(stem: str) -> list[str]:
    """The labels a generated file defines, primary first. Empty if the file does not exist."""
    path = os.path.join(TEX_DIR, stem + ".tex")
    if not os.path.isfile(path):
        return []
    return _LABEL.findall(strip_comments(_read(path)))


def generated_labels() -> dict[str, str]:
    """Every label defined by a generated ``tab_``/``fig_`` file on disk, mapped to its stem."""
    out: dict[str, str] = {}
    if not os.path.isdir(TEX_DIR):
        return out
    for entry in sorted(os.listdir(TEX_DIR)):
        stem, ext = os.path.splitext(entry)
        if ext == ".tex" and stem.startswith(("tab_", "fig_")):
            for label in labels_of(stem):
                out.setdefault(label, stem)
    return out


def inline_floats() -> list[tuple[str, str, list[str]]]:
    """``(section file name, kind, labels)`` for every float typeset directly in a section."""
    found = []
    for path in section_files():
        for m in _FLOAT.finditer(strip_comments(_read(path))):
            found.append((os.path.basename(path), m.group(1), _LABEL.findall(m.group(0))))
    return found


def pasted_copies() -> list[str]:
    """A float typeset in a section under a label a generated file also defines.

    Either it is a hand copy of the generated table -- which drifts the moment the CSV changes --
    or it is ``\\input`` elsewhere too and LaTeX sees the label twice. Both are wrong.
    """
    generated = generated_labels()
    problems = []
    for section, kind, labels in inline_floats():
        # One line per float, not per label: an aliased table carries two.
        hit = next((label for label in labels if label in generated), None)
        if hit:
            problems.append(f"{section}: {kind} \\label{{{hit}}} is typeset by hand, but "
                            f"{generated[hit]}.tex generates it -- \\input that instead")
    return problems
