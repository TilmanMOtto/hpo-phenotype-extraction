"""Ground an SLM's free-text findings to HPO identifiers with the ontology, not with PhenoBERT.

This is the earlier replacement for :func:`hpo_extraction.phenojury.ensemble_eval.run_phenobert_per_model`, and it is a
**drop-in**: :func:`link_records` returns the same ``{patient_id: {sentence_number: {hpo_id:
count}}}`` shape, so ``_write_detections``, ``derive_patient_hpos``, ``vote_k_sets``,
``plurality_sets``, ``_write_rule`` and all of the result-table library's scoring consume it unchanged. That is the
whole design: the linker is the only thing an earlier exploratory run varies, so the comparison against the Free Listing generation run's
vote table is an ablation of one component rather than of a pipeline.

**Why this is worth measuring.** ``findings/exp13_phenobert_input_format.md`` put PhenoBERT's
recovery of the intended id at **89.2 %** from a canonical HPO term name against **58.5 %** from the
report's own wording, the largest effect in the earlier runs. The 2026-08-26 earlier prep then found **228 of
489 GSC+ misses recoverable by a lexical label/synonym index alone**, with no model call. If the
SLM is already writing something close to a term name, a dictionary should beat a CNN at reading it.

**Why it is not** :attr:`HPOTree.p_phrase2HPO`. That map exists and looks like it would do. Its key
is a *sorted bag of processed words*, so *Hepatic steatosis* and *Steatosis hepatic* collide, and it
is populated last-write-wins over the whole ontology, so a synonym shared by two terms silently
drops one. :mod:`hpo_extraction.retrieval.surface_index` is the collision-aware index built for this job:
18 354 labels with zero collisions, 40 335 label strings with one.

**Three escalating tiers, and the boundary between them is the finding.**

``exact``
    :func:`surface_index.resolve_surface`, the emitted string *is* a label or synonym, modulo case
    and whitespace. Costs nothing and cannot be wrong in a way a threshold would fix.

``lemma``
    :meth:`OntologyIndex.lexical`, the same dictionary under the ontology's own normalisation
    (punctuation stripped, tokens lemmatised and sorted). Still three dict lookups, still no
    similarity. This tier exists because measurement demanded it: on the Free Listing generation run's GSC+ cache the
    largest class of ``exact`` failure is **plurals and trailing glosses**, not semantics, *Basal cell carcinomas*, *Odontogenic keratocysts*, *Multiple malformations.*, and a linker
    that cannot absorb a plural is measuring string equality, not linking.

``search``
    :meth:`OntologyIndex.candidates`, abbreviation expansion, partial containment, polarity-guarded
    fuzzy, and (when a definition index is attached) semantic neighbours. Only consulted when
    ``exact`` returned nothing, and only *accepted* when the winning candidate's source is in
    ``accept_sources``.

``accept_sources`` defaults to the two lossless sources for a reason. ``partial`` and ``fuzzy``
fire happily on prose, and a linker that accepts them by default would trade the precision the Free Listing generation run
actually has (µP 0.813 at vote_k2) for recall nobody asked for. Widening it is a swept axis, never a
default, and :func:`crosses_polarity` is what stops the widened version from linking
*macrocytosis* to *microcytosis*, which scores **0.917** against its own antonym and only **0.867**
against the intended match. No threshold separates those. Only the morpheme guard does.

**Judging is injected, never imported.** ``link_records(judge=...)`` takes a callable so an earlier exploratory run
stays pure CPU with no torch import anywhere in its path, and an earlier exploratory run supplies a
:class:`core.constrained_select.ConstrainedSelector` without this module growing a GPU dependency.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from hpo_extraction.phenojury.ensemble_eval import wrote_something
from hpo_extraction.retrieval.ontology_index import Candidate, OntologyIndex, crosses_polarity
from hpo_extraction.phenojury.prompt_diagnostics import _output_lines, _shape_of, term_candidates
from hpo_extraction.phenojury.prompts import SHAPE_PLAIN
from hpo_extraction.retrieval.surface_index import resolve_surface

logger = logging.getLogger(__name__)

#: Routes a hypothesis line can take, in escalation order. ``none`` is a real outcome, not an error:
#: a line the ontology does not recognise is the thing an earlier exploratory run's repair hop exists to attack, and
#: counting those is how the loop's value gets measured.
ROUTES = ("exact", "lemma", "search", "judge", "none")

#: How far the escalation is allowed to go. Each is a superset of the one before, and the jump from
#: ``lexical`` to ``search`` is where the cost is: the first two tiers are dict lookups (~0.5 s for
#: a whole model's cohort), ``search`` adds ``partial``/``fuzzy`` at roughly 1 s per unresolved
#: line (~15 min per model per cohort). They are named, not inferred from ``index is None``
#: so a config can ask for the cheap tier while still holding an index.
MODE_EXACT = "exact"                       # surface_index only: case + whitespace
MODE_LEXICAL = "lexical"                   # + normalise_phrase / lemma_key. Still O(1).
MODE_SEARCH = "exact_plus_search"          # + partial / fuzzy / definition. Expensive.
LINK_MODES = (MODE_EXACT, MODE_LEXICAL, MODE_SEARCH)

#: Candidate sources accepted without adjudication. ``exact``/``synonym`` are lossless lookups;
#: everything after them in ``ontology_index.SOURCES`` is a similarity judgement that belongs to
#: The judge (an earlier exploratory run) or to an explicit widening of this list.
DEFAULT_ACCEPT_SOURCES = ("exact", "synonym")


@dataclass(frozen=True)
class LinkResult:
    """One hypothesis line's resolution.

    ``hpo_ids`` is a list because the fixed release contains one ambiguous label string
    (``ASD``, both *Autistic behavior* and *Atrial septal defect*). Returning a bare string would
    be right 40 334 times out of 40 335 and silently wrong on the last one.
    """

    text: str
    hpo_ids: list[str] = field(default_factory=list)
    route: str = "none"
    candidates: list[Candidate] = field(default_factory=list)
    n_model_calls: int = 0

    @property
    def resolved(self) -> bool:
        """True when the phrase was linked to at least one HPO term."""
        return bool(self.hpo_ids)

    @property
    def n_candidates(self) -> int:
        """Number of candidate terms that were offered for the phrase."""
        return len(self.candidates)


def link_line(
    text: str,
    *,
    surface2hpo: dict,
    index: OntologyIndex | None = None,
    sentence: str = "",
    k: int = 8,
    accept_sources: tuple[str, ...] = DEFAULT_ACCEPT_SOURCES,
    judge=None,
    mode: str = MODE_SEARCH,
) -> LinkResult:
    """Resolve one emitted line to HPO identifiers, or return an unresolved :class:`LinkResult`.

    The escalation is strictly ordered and each step is skippable by configuration, so an earlier exploratory run
    (``index=None``) and an earlier exploratory run (``judge=...``) are the same function under different arguments,
    not two code paths that could drift.

    *judge*, when given, is called as ``judge(text, sentence, candidates) -> str | None`` and is
    only reached when ``search`` produced candidates that ``accept_sources`` refuses. Returning
    ``None`` is an abstention and leaves the line unresolved, which is the correct answer, and the
    reason the earlier runs' I5 ("unresolvable spans escalate, never guess") survives into this pipeline.
    """
    cleaned = " ".join((text or "").split())
    if not cleaned:
        return LinkResult(text=text or "")

    hits = resolve_surface(cleaned, surface2hpo)
    if hits:
        return LinkResult(text=cleaned, hpo_ids=list(hits), route="exact")

    if index is None or mode == MODE_EXACT:
        return LinkResult(text=cleaned)

    # Tier 2, the dictionary with the ontology's own normalisation. Three dict lookups, no
    # similarity, and it is what separates a linker from a string comparison: `surface_key`
    # collapses case and whitespace only, so *Basal cell carcinomas* misses *Basal cell carcinoma*
    # and *Odontogenic keratocysts* misses *Odontogenic keratocysts of the jaw*. Measured on
    # The Free Listing generation run's GSC+ cache, plurals and trailing glosses, not semantics, are the single largest
    # class of exact-lookup failure.
    #
    # The key here is a *sorted bag* (`normalise_phrase`), which is the property
    # `surface_index`'s docstring warns about: *Hepatic steatosis* and *Steatosis hepatic* collide.
    # That is acceptable as a tier reached only after the order-preserving lookup has already
    # failed, and the route is recorded so its contribution stays separable in the trace.
    # Singular readings first: they are checked against the *surface* index, so they keep word
    # order and the zero-collision guarantee that the bag-of-words tier below gives up.
    for variant in singular_variants(cleaned):
        hits = resolve_surface(variant, surface2hpo)
        if hits:
            return LinkResult(text=cleaned, hpo_ids=list(hits), route="lemma")

    lexical = {
        hpo_id: sources for hpo_id, sources in index.lexical(cleaned).items()
        if any(source in accept_sources for source in sources)
    }
    if lexical:
        return LinkResult(text=cleaned, hpo_ids=sorted(lexical), route="lemma")

    if mode == MODE_LEXICAL:
        return LinkResult(text=cleaned)

    candidates = index.candidates(cleaned, sentence=sentence, k=k)
    # The polarity guard is re-applied *here* because the index applies it per-source, not per-
    # result: `partial` and `fuzzy` refuse a polarity-crossing match internally, but `abbreviation`
    # and `definition` do not, and those are the sources a widened `accept_sources` or an
    # attached definition index would let through. Lowercased on both sides because the check is a
    # bare substring test and both call sites inside `ontology_index` feed it normalised keys;
    # passing a release label verbatim would silently never match `("macro", "micro")`.
    lowered = cleaned.casefold()
    candidates = [c for c in candidates if not crosses_polarity(lowered, c.label.casefold())]
    if not candidates:
        return LinkResult(text=cleaned)

    top = candidates[0]
    if any(source in accept_sources for source in top.sources):
        return LinkResult(
            text=cleaned, hpo_ids=[top.hpo_id], route="search", candidates=candidates
        )

    if judge is not None:
        chosen = judge(cleaned, sentence, candidates)
        return LinkResult(
            text=cleaned,
            hpo_ids=[chosen] if chosen else [],
            route="judge",
            candidates=candidates,
            n_model_calls=1,
        )

    return LinkResult(text=cleaned, candidates=candidates)


#: Endings that look plural but are not, so a naive trailing-``s`` strip must skip them:
#: ``-ss`` (*illness*), ``-us`` (*situs*), ``-is`` (*Ptosis* → *Ptosi*).
#:
#: **not** ``-as``: that would exclude *carcinomas*, *neurofibromas*, *meningiomas*,
#: *adenomas*, the ``-oma`` plurals, which are the single most common plural in this vocabulary
#: and led the list of annotated terms this linker was losing. The exclusions above are cheap insurance
#: on words that are never English plurals. A blanket ``-as`` rule would have cost the fix its
#: entire point. The real safety net is elsewhere anyway: every variant must still match a real
#: label string, so a wrong singularisation (*pancreas* → *pancrea*) names nothing and
#: fails harmlessly.
_NOT_PLURAL_ENDINGS = ("ss", "us", "is")


def singular_variants(text: str) -> list[str]:
    """Deterministic singular readings of *text*, most conservative first.

    **Why this exists, and why it is not lemmatisation.** ``lemma_key`` looks like the right tool
    and is inert here: ``WordItem.lemma_dict`` is a fixed table that does not carry medical plurals,
    so ``lemma_key("Neurofibromas")`` returns ``"neurofibromas"`` unchanged. Measured on the Free Listing generation run's
    GSC+ cache, adding that tier recovered **1–22 lines per model**, nothing, while the annotated terms
    the linker was losing to PhenoBERT were led by the plurals it could not absorb:
    *Neurofibroma* (35), *Basal cell carcinoma* (18), *Meningioma* (15).

    HPO labels are singular, so singularising the *query* and re-checking it against the
    order-preserving, collision-free surface index is the narrow fix. This is not fuzzy matching:
    each variant is a specific string that must still match a real label string. A variant
    that names nothing simply fails.

    Two variants are produced, every token singularised, and the last token only, because a label
    may be plural-headed but singular-modified (*Dysplastic hip joints* vs *Basal cell carcinomas*).
    """
    def _singular(token: str) -> str:
        low = token.casefold()
        if len(low) > 4 and low.endswith("ies"):
            return token[:-3] + "y"
        if len(low) > 4 and low.endswith(("ses", "xes", "zes", "ches", "shes")):
            return token[:-2]
        if len(low) > 3 and low.endswith("s") and not low.endswith(_NOT_PLURAL_ENDINGS):
            return token[:-1]
        return token

    tokens = text.split()
    if not tokens:
        return []
    out = []
    for candidate in (
        " ".join(_singular(t) for t in tokens),
        " ".join(tokens[:-1] + [_singular(tokens[-1])]),
    ):
        if candidate != text and candidate not in out:
            out.append(candidate)
    return out


def split_conjuncts(text: str) -> list[str]:
    """A prose line's comma/semicolon-separated parts, or ``[]`` when there is nothing to split.

    **This is what makes the comparison against PhenoBERT an ablation, not a handicap.**
    PhenoBERT extracts candidate *phrases* from anywhere in its input. A line-oriented linker's
    unit is the line. Those are the same thing only when the model wrote one finding per line, and
    measured on the Free Listing generation run's own GSC+ cache they often are not, OpenBioLLM and Apertus answer
    ``p0_baseline`` with a single prose sentence listing every finding at once:

        "Brachydactyly, aplastic or hypoplastic nails, symphalangism, craniosynostosis, ..."

    Whole-line lookup scores that at zero, which would be reported as a linker failure and
    is really a layout difference. Splitting restores the like-for-like unit while staying entirely
    deterministic: every part is still resolved by exact lookup, so nothing fuzzy enters here.

    Only ``,`` and ``;`` split. Splitting on " and " is not done, it would cut
    *aplastic or hypoplastic nails* and *palmar and plantar pits* into fragments that name nothing,
    trading one bias for another.
    """
    if not any(sep in text for sep in ",;"):
        return []
    parts = [p.strip(" \t.;,") for p in re.split(r"[,;]", text)]
    return [p for p in parts if p]


def link_record(
    record: dict,
    *,
    surface2hpo: dict,
    index: OntologyIndex | None = None,
    shape: str = SHAPE_PLAIN,
    k: int = 8,
    accept_sources: tuple[str, ...] = DEFAULT_ACCEPT_SOURCES,
    judge=None,
    segment_lines: bool = True,
    mode: str = MODE_SEARCH,
) -> list[LinkResult]:
    """Every hypothesis in one ``llm_extractions_*.jsonl`` record, resolved.

    A record whose reply is a decline (``"no phenotype"``, ``NONE``, ``{"findings": []}``) yields
    no results at all. That check is :func:`ensemble_eval.wrote_something`, reused, not
    reimplemented so a sentinel the ensemble driver ignores cannot become a phantom detection here, which would show up as a precision difference attributed to the linker.

    ``segment_lines`` falls back to :func:`split_conjuncts` for a line that does not resolve whole.
    The whole line is always tried **first**, so a multi-word term containing a comma is never
    shredded by a fallback that was not needed.
    """
    raw = record.get("llm_output")
    if not wrote_something(raw):
        return []

    sentence = record.get("sentence_text") or ""

    def _link(text):
        return link_line(
            text, surface2hpo=surface2hpo, index=index, sentence=sentence,
            k=k, accept_sources=accept_sources, judge=judge, mode=mode,
        )

    out: list[LinkResult] = []
    for line in _output_lines(raw):
        for candidate_text in term_candidates(line, shape):
            whole = _link(candidate_text)
            if whole.resolved or not segment_lines:
                out.append(whole)
                continue
            conjuncts = split_conjuncts(candidate_text)
            if not conjuncts:
                out.append(whole)
                continue
            out.extend(_link(part) for part in conjuncts)
    return out


def link_records(
    records: list[dict],
    report_ids: list[str],
    *,
    surface2hpo: dict,
    index: OntologyIndex | None = None,
    prompt_key: str = "",
    shape: str | None = None,
    k: int = 8,
    accept_sources: tuple[str, ...] = DEFAULT_ACCEPT_SOURCES,
    judge=None,
    segment_lines: bool = True,
    mode: str = MODE_SEARCH,
    model_key: str = "",
    logger_: logging.Logger | None = None,
) -> tuple[dict, list[dict]]:
    """``(sent_hpos, trace_rows)``, the drop-in replacement for ``run_phenobert_per_model``.

    ``sent_hpos`` is ``{patient_id: {sentence_number: {hpo_id: count}}}``, keyed as the
    PhenoBERT path keys it, so ``_write_detections`` writes an identical-schema file.

    ``trace_rows`` is the diagnostic artifact this method has and PhenoBERT does not: one row per
    *hypothesis line*, carrying the route it took and the candidates it saw. Every offline metric
    in the earlier runs (on-label rate, the link 2x2, marginal yield per hop) reads that file, and none of them
    needs a re-run to answer a new question.

    Report ids are filtered against *report_ids*, not trusted from the records: a cached
    extractions file may cover a superset of the cohort being scored (an earlier exploratory run reads the Free Listing generation run's
    228-report GSC+ cache while a pilot may ask for 20), and silently scoring the superset would
    make the run incomparable to the table it is meant to join.
    """
    log = logger_ or logger
    if shape is None:
        shape = _shape_of(prompt_key) if prompt_key else SHAPE_PLAIN

    wanted = {str(rid) for rid in report_ids}
    sent_hpos: dict[str, dict[int, dict[str, int]]] = {}
    trace: list[dict] = []
    n_lines = 0
    n_resolved = 0

    for record in records:
        pid = str(record.get("patient_id"))
        if pid not in wanted:
            continue
        try:
            sent_num = int(record.get("sentence_number"))
        except (TypeError, ValueError):
            log.warning("skipping record with unusable sentence_number: %r", record.get("patient_id"))
            continue

        for result in link_record(
            record,
            surface2hpo=surface2hpo,
            index=index,
            shape=shape,
            k=k,
            accept_sources=accept_sources,
            judge=judge,
            segment_lines=segment_lines,
            mode=mode,
        ):
            n_lines += 1
            if result.resolved:
                n_resolved += 1
                bucket = sent_hpos.setdefault(pid, {}).setdefault(sent_num, {})
                for hpo_id in result.hpo_ids:
                    bucket[hpo_id] = bucket.get(hpo_id, 0) + 1
            trace.append({
                "report_id": pid,
                "sentence_number": sent_num,
                "model": model_key or record.get("model", ""),
                "line": result.text,
                "route": result.route,
                "hpo_ids": list(result.hpo_ids),
                "n_candidates": result.n_candidates,
                "n_model_calls": result.n_model_calls,
                "top_candidates": [c.label for c in result.candidates[:3]],
                "top_sources": list(result.candidates[0].sources) if result.candidates else [],
            })

    log.info(
        "linked %s | %d line(s) over %d report(s) | %d resolved (%.1f%%) | %d unresolved",
        model_key or "?", n_lines, len(sent_hpos), n_resolved,
        100.0 * n_resolved / max(1, n_lines), n_lines - n_resolved,
    )
    return sent_hpos, trace


def route_summary(trace: list[dict]) -> dict:
    """Route mix and on-label rate over one model's trace rows.

    ``on_label_rate`` is the share of lines that resolved by ``exact``, the string the model wrote
    *was* an ontology label string. It is the metric ``findings/exp13_phenobert_input_format.md``
    is really about, and it is the one that predicts cost: an on-label line needs no retrieval and
    no model call, so a prompt that raises it makes the whole pipeline cheaper as well as better.
    """
    counts = {route: 0 for route in ROUTES}
    for row in trace:
        counts[row.get("route", "none")] = counts.get(row.get("route", "none"), 0) + 1
    n = max(1, len(trace))
    return {
        "n_lines": len(trace),
        "n_resolved": sum(1 for r in trace if r.get("hpo_ids")),
        "on_label_rate": counts["exact"] / n,
        "n_model_calls": sum(int(r.get("n_model_calls") or 0) for r in trace),
        **{f"n_route_{route}": counts[route] for route in ROUTES},
    }
