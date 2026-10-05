"""Searching the ontology without shipping it.

``app/annotation_ui`` puts every HPO term, roughly 19 000 ``{label, value}`` dicts, into a
``dcc.Store`` and filters them in the browser. That is several megabytes across an SSH tunnel on
every page load, and it is the single reason that app is unpleasant to use on the cluster.

Here the dropdown starts empty and is filled by a callback on ``search_value``: the index lives in
this process, the query crosses the wire, and at most :data:`MAX_RESULTS` options come back. The
index is built once from the shared ``HPOTree``, labels *and* synonyms, because a curator types
what the report says ("fits", "low muscle tone"), not the ontology's preferred label.

The index also carries each term's **definition**, because the screen needs it: a curator deciding
whether ``HP:0001252`` belongs on a sentence is deciding what the ontology means by *hypotonia*, and
sending them to a browser tab to find out is how a verdict gets guessed instead. Definitions are
served as hover text (:meth:`HPOSearch.describe`), never rendered inline, they are two or three
sentences each and would bury the terms they explain. About one term in seven has none. Those say
so rather than showing an empty tooltip.

A query that reads as an **HPO id** short-circuits all of that: ``HP:0003645``, ``hp0003645``,
``0003645`` and ``3645`` are the same lookup, and a digit string that is *already* zero-padded is
read as a prefix over the id space, so a half-remembered id narrows instead of failing. The digits
alone matter because that is how ids appear once a ground truth file's prefix has been stripped, and pasting
them used to find nothing.

Otherwise ranking is boring and total, so the same query always yields the same list:

1. exact label match
2. label starts with the query
3. exact synonym match
4. synonym starts with the query
5. label contains the query
6. synonym contains the query

with shorter labels first inside a tier, then alphabetically. Short labels first because a query
like "seizure" should surface *Seizure* above *Bilateral tonic-clonic seizure with focal onset*.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

#: Never return more than this, whatever the query. A two-letter query matches thousands of terms
#: and rendering them is what the browser cannot afford.
MAX_RESULTS = 50

#: Below this, return nothing. One character is not a query, it is a keystroke.
MIN_QUERY = 2

#: HPO ids are zero-padded to seven digits, so ``3645`` and ``0003645`` name the same term.
_ID_WIDTH = 7


class HPOSearch:
    """A label+synonym index over the ontology. Built once, queried per keystroke."""

    def __init__(self, tree):
        self._tree = tree
        self._labels: dict[str, str] = {}
        self._defs: dict[str, str] = {}
        self._entries: list[tuple[str, str, str]] = []   # (code, label_lower, synonyms_blob)
        self._build()

    def _build(self) -> None:
        for code in self._tree.hpo_list:
            try:
                label = str(self._tree.getNameByHPO(code))
            except Exception:  # noqa: BLE001 - a code the release does not name is still searchable
                label = code
            self._labels[code] = label
            self._defs[code] = _first_definition(self._tree, code)
            try:
                phrases = self._tree.getPhrasesByHPO(code)
            except Exception:  # noqa: BLE001
                phrases = []
            label_lower = label.lower()
            # The label reappears in getPhrasesByHPO. Dropping it keeps the synonym tiers honest,
            # so a label hit is never also reported as a synonym hit.
            synonyms = [p for p in phrases if p and p != label_lower]
            self._entries.append((code, label_lower, "␟".join(synonyms)))
        logger.info("HPO search index: %d terms", len(self._entries))

    # ── lookup ───────────────────────────────────────────────────────────────
    def label(self, code: str) -> str:
        """The preferred label, or the code itself when the release does not carry it.

        Sentence case, not title case: ``hpo.json`` stores labels lowercased, and title-casing them
        (as ``annotation_ui`` does) turns *Abnormality of the cardiovascular system* into
        *Abnormality Of The Cardiovascular System*, louder, and no longer the ontology's string.
        """
        label = self._labels.get(code)
        if not label:
            return code
        return label[:1].upper() + label[1:]

    def definition(self, code: str) -> str:
        """The ontology's definition, or ``""`` where the release carries none."""
        return self._defs.get(code, "")

    def describe(self, code: str, prefix: str = "") -> str:
        """The hover text for a term: name, id, then the definition on its own line.

        One function so every mention of a phenotype, a mark in the report, a chip in the ground truth
        panel, a row in Approve mode, hovers to the same words. *prefix* prepends context the
        caller has and this module does not ("PhenoBERT →", "in prior_annotation_2 only").
        """
        # An unknown code has no label but ``label()`` returns the code, so naming it twice is
        # what the obvious formatting produces. Say it once.
        head = code if not self.known(code) else f"{self.label(code)} ({code})"
        if prefix:
            head = f"{prefix} {head}"
        definition = self.definition(code)
        if definition:
            return f"{head}\n\n{definition}"
        if not self.known(code):
            return f"{head}\n\nNot a phenotypic-abnormality term in this hpo.json."
        return f"{head}\n\nNo definition in this HPO release."

    def known(self, code: str) -> bool:
        """Whether *code* is a phenotypic-abnormality term this ontology release names.

        Ground truth files predate HPO releases. A code that is not known can still be curated, but the
        screen says so, not silently showing the bare id as if it were a label.
        """
        return code in self._labels

    def search(self, query: str, limit: int = MAX_RESULTS) -> list[dict]:
        """``[{"label": "Seizure (HP:0001250)", "value": "HP:0001250"}, …]``, capped at *limit*."""
        query = (query or "").strip().lower()
        if len(query) < MIN_QUERY:
            return []

        # An HPO id is a lookup, not a search, however it was typed. Curators read ids off
        # annotation files and paste the digits alone as often as the full ``HP:`` form.
        by_id = self._by_id(query, limit)
        if by_id:
            return by_id

        tiers: list[list[tuple[int, str, str]]] = [[] for _ in range(6)]
        for code, label, synonyms in self._entries:
            tier = -1
            if label == query:
                tier = 0
            elif label.startswith(query):
                tier = 1
            elif synonyms:
                parts = synonyms.split("␟")
                if any(p == query for p in parts):
                    tier = 2
                elif any(p.startswith(query) for p in parts):
                    tier = 3
                elif query in label:
                    tier = 4
                elif query in synonyms:
                    tier = 5
            elif query in label:
                tier = 4
            if tier >= 0:
                tiers[tier].append((len(label), label, code))
                # Cheap early exit: once the top tier alone can fill the page there is no reason
                # to keep scanning 19 000 terms on every keystroke.
                if tier == 0 and len(tiers[0]) >= limit:
                    break

        out: list[dict] = []
        for tier in tiers:
            if len(out) >= limit:
                break
            for _, _, code in sorted(tier):
                out.append(self.option(code))
                if len(out) >= limit:
                    break
        return out

    def _by_id(self, query: str, limit: int) -> list[dict]:
        """Terms matching *query* read as an HPO id, or ``[]`` when it does not read as one.

        Every spelling a curator actually produces resolves to the same term: ``HP:0003645``,
        ``hp:0003645``, ``HP0003645``, ``0003645``, and ``3645``, the last two because that is
        what the id column of a ground truth file looks like once the prefix has been stripped, and pasting
        it used to find nothing at all.

        A *padded* fragment is read as a prefix instead (``00036`` → every ``HP:00036xx``), so an
        id you only half remember narrows, not fails. Any exact hit leads, then the prefix
        block in code order, so the term you named is never buried under its numeric neighbours.
        """
        digits = _id_digits(query)
        if digits is None:
            return []

        # Zero-padding is the tell. Digits that start with one are the id as the file writes it, so
        # they are read as a prefix and nothing else. Digits that do not are an id whose padding was
        # dropped ("3645"), so they are padded back out first. Reading "00012" both ways would put
        # HP:0000012 above the HP:00012xx block the curator was obviously narrowing towards.
        exact = ""
        if not digits.startswith("0") and len(digits) <= _ID_WIDTH:
            padded = "HP:" + digits.zfill(_ID_WIDTH)
            if self.known(padded):
                exact = padded

        prefix = "HP:" + digits
        # ``_entries`` follows ``hpo_list``, which is not sorted, so the tail is sorted here.
        tail = sorted(code for code, _, _ in self._entries
                      if code.startswith(prefix) and code != exact)
        codes = ([exact] if exact else []) + tail
        return [self.option(code) for code in codes[:limit]]

    def option(self, code: str) -> dict:
        """One dropdown option. Public because a caller holding a chosen code needs it too."""
        return {"label": f"{self.label(code)} ({code})", "value": code}


def _id_digits(query: str) -> str | None:
    """The digits of *query* read as an HPO id, or ``None`` when it is not one.

    Punctuation and spacing are noise here, ``HP:0003645``, ``hp 0003645`` and ``0003645`` are the
    same request, so everything but letters and digits is dropped, an ``hp`` prefix with it. What
    is left has to be *all* digits: ``11q`` is a text query that happens to contain a number.
    """
    text = re.sub(r"[^0-9a-z]", "", query.lower())
    if text.startswith("hp"):
        text = text[2:]
    if not text or not text.isdigit():
        return None
    return text


def _first_definition(tree, code: str) -> str:
    """``hpo.json`` stores ``Def`` as a list. A term has at most one, and many have none."""
    try:
        values = tree.data[code].get("Def") or []
    except (AttributeError, KeyError, TypeError):
        return ""
    for value in values:
        text = str(value).strip()
        if text:
            return text
    return ""


#: One index per ontology object. Repointing the app at another cohort rebuilds the Registry, and
#: rebuilding this with it would re-scan 18 000 terms for an index that cannot have changed, the
#: ontology is not part of the path set.
_INDEX: dict[int, HPOSearch] = {}


def get_search(tree) -> HPOSearch:
    """The index for *tree*, built at most once per process."""
    key = id(tree)
    index = _INDEX.get(key)
    if index is None:
        index = HPOSearch(tree)
        _INDEX[key] = index
    return index
