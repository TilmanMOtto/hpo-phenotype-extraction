"""What a phenotype code *means*, in the one wording every screen in this app uses.

The deep-dive names HPO terms in a dozen places, a chip under the report, an underline on a
sentence, a row in the terms table, an option in the node picker, a point in the ontology diagram,
a cell in the error table. Until now only three of those could tell you what the term was, and the
rest showed an id and a label and left the reader to go and look it up. A verdict on ``HP:0001250``
cannot be judged without knowing what HP:0001250 is.

So this module is the single reader, and :func:`tooltip` is the single wording, in the same spirit
as ``apps.curation_ui.search.HPOSearch.describe``: every mention of a phenotype hovers to the
same name, the same definition and the same synonyms, and two surfaces cannot drift into saying
different things about one code.

**Memoised, because the call count changed.** These were per-chip lookups. They are now per *drawn
mark*, once per term per sentence per render. Each one is a ``view.resolve()`` plus a dict hit into
a 15 MB ontology, which is cheap once and not cheap a few thousand times. The cache is per
``OntologyView``, process-wide, rebuilt only when the ontology is, which is never within a
session, in the same spirit as ``search.get_search``. See :data:`_CACHES` for why the view is
held rather than merely used as a key.

dash-free: it is imported by views, by the self-check and by the tests, and only the
first of those can afford a dash import.
"""

from __future__ import annotations

#: ``id(view) -> (view, labels, tooltips)``. The view is kept **in** the entry, not just used as
#: its key, and that is essential, not tidy: CPython recycles ``id()`` as soon as an
#: object is collected, so a cache keyed on the id alone will happily answer one ontology's
#: question with another ontology's labels once the first has been dropped. Holding the reference
#: both prevents the recycling and makes the identity check below meaningful. There is one
#: ``OntologyView`` per process, so this keeps nothing alive that was going away.
_CACHES: dict[int, tuple] = {}


def _cache(view) -> tuple[dict, dict]:
    entry = _CACHES.get(id(view))
    if entry is None or entry[0] is not view:
        entry = (view, {}, {})
        _CACHES[id(view)] = entry
    return entry[1], entry[2]


def entry(view, hpo_id: str) -> dict | None:
    """The raw ``hpo.json`` record behind a code, or ``None`` if the ontology does not know it."""
    if view is None or not hpo_id:
        return None
    return view.tree.data.get(view.resolve(hpo_id) or hpo_id)


def label(view, hpo_id: str) -> str:
    """The term's preferred name, or the code itself when the release does not name it."""
    if view is None or not hpo_id:
        return hpo_id
    cache, _ = _cache(view)
    if hpo_id not in cache:
        names = (entry(view, hpo_id) or {}).get("Name") or []
        cache[hpo_id] = names[0] if names else hpo_id
    return cache[hpo_id]


def definition(view, hpo_id: str) -> str:
    """The HPO definition, or ``""``. ``Def`` is a list with at most one entry."""
    return ((entry(view, hpo_id) or {}).get("Def") or [""])[0] or ""


def synonyms(view, hpo_id: str) -> list[str]:
    """Synonyms of *hpo_id* in *view*."""
    return list((entry(view, hpo_id) or {}).get("Synonym") or [])


def tooltip(view, hpo_id: str, prefix: str = "") -> str:
    """The hover text for a term: id and name, the definition, then the synonyms.

    *prefix* prepends context the caller has and this module does not ("curated ground truth →",
    "PhenoBERT →"), the same way ``HPOSearch.describe`` takes one.

    A code the ontology does not know says so, not showing a bare id as if it were a label:
    a predicted code this release has never heard of is either a hallucination or an id from
    another release, and that is worth knowing on hover.
    """
    _, cache = _cache(view)
    if hpo_id not in cache:
        cache[hpo_id] = _build(view, hpo_id)
    text = cache[hpo_id]
    return f"{prefix} {text}" if prefix else text


def _build(view, hpo_id: str) -> str:
    if entry(view, hpo_id) is None:
        return f"{hpo_id}, not in this HPO release"
    parts = [f"{hpo_id} · {label(view, hpo_id)}"]
    text = definition(view, hpo_id)
    if text:
        parts.append(text)
    also = synonyms(view, hpo_id)
    if also:
        parts.append("Also known as: " + ", ".join(also))
    return "\n\n".join(parts)


def known(view, hpo_id: str) -> bool:
    """Whether this ontology release names *hpo_id*."""
    return entry(view, hpo_id) is not None


def same_term(view, a: str, b: str) -> bool:
    """Whether two codes name one term once alt ids are resolved.

    A ground truth file predates the release it is scored against, so a ground truth code and a predicted code can
    be the same phenotype spelled two ways. Comparing the raw strings reads such a pair as a miss, which is how a term found by the method ends up drawn in red on the report.
    """
    if not a or not b:
        return False
    if a == b:
        return True
    if view is None:
        return False
    return (view.resolve(a) or a) == (view.resolve(b) or b)


def resolve_all(view, codes) -> set[str]:
    """*codes* with every alt id resolved, for membership tests against a prediction set."""
    if view is None:
        return {str(c) for c in codes}
    return {str(view.resolve(str(c)) or c) for c in codes}
