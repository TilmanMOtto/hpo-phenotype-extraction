"""Where the inputs are, and the one coordinate system every method is placed into.

Five methods wrote their evidence in four different frames of reference. PhenoBERT and AutoPCR
carry character offsets into the document *as they staged it*; RAG-HPO carries a sentence's text
and no offset at all; PhenoJury carries a ``sentence_number``; TreePhenoRAG carries a
``sent_index``. Drawing them in one gutter means agreeing on a single frame first, and getting that
agreement wrong is not a visible bug -- it is an underline three characters to the left, or a term
attributed to the sentence after the one it came from.

So the frame is not invented here. It is the pipeline's own, recovered the way the pipeline built
it and then *checked*:

1. The text is read as the drivers read it. HCY goes through
   ``hpo_extraction.data.loading.load_txt``, which decodes **latin1** -- every HCY report is German, and
   reading them as UTF-8 produces a different string for every report containing an umlaut, which
   is all of them. GSC+ goes through ``hpo_extraction.evaluation.datasets.gsc.load_gsc_reports``, which decodes
   **UTF-8** from a corpus whose files have no extension and whose directory holds Windows
   ``*:Zone.Identifier`` sidecars that must be skipped. Both are imported, never retyped.
2. ``split_sents(segment_dict(load_stanza(...)))`` gives the sentence list, and
   ``experiments/03_setup/segment_reports.py`` has already written it to a ``segmented_reports.csv``. Position
   in that list **is** ``sent_index``, ``sentence_number`` and the curated ground truth's ``segment_idx``
   -- one integer, three names. The GSC+ corpus needs its own run of that script. There is no
   cheaper substitute, because a locally-invented sentence split would renumber every method's
   recorded position and attribute real decisions to the wrong sentences.
3. ``spans.align_segments`` locates each sentence in the document, and ``spans.segment_texts``
   gives the segment as the document spells it. That string is what the reader sees and what every
   span in a bundle indexes.

A document whose recovered segmentation does not align to its own text is **dropped**, not drawn.
``exp13_ui.evidence`` made that choice first and it is the right one: a document drawn against a
segmentation that is not the run's would attribute real decisions to the wrong sentences, and
there is no banner that makes that safe.

What varies between the two cohorts -- the artifact subdirectory, the ground truth, the decoding, the
quotable row of ``comparison``'s table -- is named once in :mod:`apps.compare_ui.cohorts` and read
from there. Everything cross-app is imported, never copied: the placement ladder in
``exp13_ui.curated``, the alignment in ``hcy_curation_ui.spans`` and the frame reader in
``phenojury_generation_free_listing.frame`` each took several passes to get right, and a second copy would drift from
the original when it mattered.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

from apps.compare_ui import cohorts

logger = logging.getLogger(__name__)

#: Where the experiment directories and the deep-dive frames live, on the cluster.
from hpo_extraction.paths import lookup as _lookup  # noqa: E402

DEFAULT_OUTPUT_BASE = _lookup("results_dir")
#: The HCY cohort directory: reports, segmentation and the curated ground truth datasets.
DEFAULT_HCY_DIR = _lookup("hcy.dir")
#: The GSC+ corpus and RAG-HPO's vendored subset, both tracked in the repo.
DEFAULT_GSC_DIR = "resources/data/GSC_2024"
DEFAULT_RAGHPO_DIR = "resources/data/GSC_RAGHPO"
#: Where :mod:`apps.compare_ui.build` writes and :mod:`apps.compare_ui.app` reads. One directory per
#: cohort under one parent, so the app can be pointed at the parent and offer both.
DEFAULT_BUNDLE_ROOT = (
    os.path.join(_lookup("output_dir"), "compare_ui_bundles"))

#: The comparison whose tables this app explains. Its ``t1_overall.csv`` carries the only rates
#: that may be quoted. See :mod:`apps.compare_ui.views.scorecard`.
COMPARISON_EXP = "comparison"

#: The GSC+ segmentation ``experiments/03_setup/segment_reports.py --corpus gsc`` writes, relative to
#: ``output_base``. It is a build product of the corpus, not part of it, so it does not land in
#: ``resources/``.
GSC_SEGMENTS = os.path.join("gsc_segmentation", "segmented_reports.csv")


def bundles_dir(root: str, cohort) -> str:
    """``<root>/<cohort>`` -- where one cohort's bundles live."""
    key = getattr(cohort, "key", cohort)
    return os.path.join(root, str(key))


#: Kept for callers that predate the two-cohort split and mean HCY.
DEFAULT_BUNDLES = bundles_dir(DEFAULT_BUNDLE_ROOT, cohorts.HCY)


@dataclass
class Paths:
    """Resolved input locations. Nothing here is required to exist -- see :meth:`problems`."""

    output_base: str = DEFAULT_OUTPUT_BASE
    #: Which cohort is being built. Everything below that says "the documents" means this one's.
    cohort: str = cohorts.DEFAULT.key
    hcy_dir: str = DEFAULT_HCY_DIR
    #: The GSC+ corpus (``Text/`` + ``Annotations/``) and RAG-HPO's vendored 114-document subset.
    gsc_dir: str = DEFAULT_GSC_DIR
    raghpo_dir: str = DEFAULT_RAGHPO_DIR
    #: A ``curated_gold_<date>`` directory. Empty means "the newest one beside ``hcy_dir``".
    #: HCY only; RAG-HPO's annotation is a vendored file with no versions to choose between.
    gold_dir: str = ""
    #: A sampling-frame directory. Empty means "find one under ``output_base``" for HCY, and
    #: nothing at all for GSC+, whose frame is drawn explicitly.
    frame_dir: str = ""
    #: Override for the segmentation CSV. Empty takes the cohort's default.
    segments: str = ""

    @property
    def cohort_obj(self):
        """The cohort object these paths belong to."""
        return cohorts.get(self.cohort)

    @property
    def artifact_dir(self) -> str:
        """The subdirectory each driver wrote this cohort's artifacts into."""
        return self.cohort_obj.artifact_dir

    @property
    def input_dir(self) -> str:
        """The directory of document text -- for GSC+ the corpus root, which holds ``Text/``."""
        if self.cohort == cohorts.HCY.key:
            return os.path.join(self.hcy_dir, "input")
        return self.gsc_dir

    @property
    def segments_path(self) -> str:
        """Path of the cohort's ``segmented_reports.csv``."""
        if self.segments:
            return self.segments
        if self.cohort == cohorts.HCY.key:
            return os.path.join(self.hcy_dir, "segmented_reports.csv")
        return os.path.join(self.output_base, GSC_SEGMENTS)

    @property
    def comparison_dir(self) -> str:
        """Folder of the comparison run."""
        return os.path.join(self.output_base, COMPARISON_EXP)

    def exp(self, exp_id: str, *parts: str) -> str:
        """Path of a file inside the run folder *exp_id*."""
        return os.path.join(self.output_base, exp_id, *parts)

    def cohort_exp(self, exp_id: str, *parts: str) -> str:
        """``{output_base}/{exp_id}/{artifact_dir}/...`` -- an artifact of *this* cohort's run.

        The one place a driver's per-cohort layout is written down. Adapters call this rather than
        spelling ``"hcy"``, which is what they used to do and what made a second cohort a
        five-file change instead of a one-line one.
        """
        return os.path.join(self.output_base, exp_id, self.artifact_dir, *parts)

    def problems(self) -> list:
        """The missing inputs that would make a build meaningless, in the order to fix them.

        Reported, not raised so the caller can print all of them at once: discovering four
        wrong paths one cluster round trip at a time is how an afternoon goes.
        """
        out = []
        if not os.path.isdir(self.output_base):
            out.append("output_base does not exist: " + self.output_base)
        if self.cohort == cohorts.HCY.key:
            if not os.path.isdir(self.input_dir):
                out.append("no HCY reports at " + self.input_dir)
        else:
            if not os.path.isdir(os.path.join(self.gsc_dir, "Text")):
                out.append("no GSC+ corpus at " + os.path.join(self.gsc_dir, "Text"))
            if not os.path.isfile(os.path.join(self.raghpo_dir, "annotations.csv")):
                out.append("no RAG-HPO annotations at "
                           + os.path.join(self.raghpo_dir, "annotations.csv"))
        if not os.path.isfile(self.segments_path):
            out.append("no segmentation at " + self.segments_path
                       + " -- run experiments/03_setup/segment_reports.py --corpus " + str(self.cohort))
        return out


# -- the ontology ------------------------------------------------------------

#: One ``OntologyView`` and one traversal graph per process, memoised the way
#: ``app/exp13_ui/registry`` memoises them. They are 15 MB of ``hpo.json`` and a walk over 18 000
#: nodes. Rebuilding them per report would dominate the whole build.
_VIEW = None
_GRAPH = None


def get_view():
    """The shared ontology view.

    Built here, not borrowed from ``apps.treephenorag_ui.registry``, which imports Dash at module
    scope through its theme. The builder is a CPU job that runs where the artifacts are and must
    not need a UI toolkit installed. The *dash-free* modules of that app (``curated``, ``loaders``,
    ``terms``, ``evidence``, ``pruning``) are imported as before.
    """
    global _VIEW
    if _VIEW is None:
        from hpo_extraction.evaluation.metrics import OntologyView
        from hpo_extraction.ontology.hpo_tree import HPOTree

        tree = HPOTree()
        tree.buildHPOTree()          # populates depth_dict; __init__ does not
        _VIEW = OntologyView(tree)
    return _VIEW


def get_graph(view=None):
    """``(children_map, roots, depths)`` for the traversal DAG below the layer-1 roots."""
    global _GRAPH
    if _GRAPH is None:
        from apps.treephenorag_ui import pruning

        view = view if view is not None else get_view()
        children_map, roots = pruning.children_map_from_tree(view.tree)
        _GRAPH = (children_map, roots, pruning.bfs_depths(children_map, roots))
    return _GRAPH


def set_shared(view=None, graph=None) -> None:
    """Inject a prebuilt view and graph -- the seam the fixture and the tests use.

    ``hpo_extraction.ontology.hpo_tree`` reads NLTK stopwords in a class body, so on a machine without them it
    dies at *import*, not at first use. Without this seam the entire builder would be untestable
    anywhere the NLP stack is absent, which is most places it will be read.
    """
    global _VIEW, _GRAPH
    if view is not None:
        _VIEW = view
    if graph is not None:
        _GRAPH = graph


# -- the coordinate system ---------------------------------------------------

@dataclass
class ReportView:
    """One document in the single frame every method is placed into.

    ``display[i]`` is segment *i* as the document spells it; ``ranges[i]`` is where it sits in
    ``text``; ``sentences[i]`` is the tokenized string the pipeline actually passed to a model.
    The three are parallel and all are needed: offsets come from ``text``, placement comes from
    ``sentences``, and what the reader sees is ``display``.
    """

    report_id: str
    text: str
    sentences: list
    display: list
    ranges: list

    @property
    def n_segments(self) -> int:
        """Number of segments in the report."""
        return len(self.sentences)

    def segment_of_offset(self, offset: int):
        """Which segment a character offset into ``text`` falls in, or ``None``."""
        for idx, span in enumerate(self.ranges):
            if span is not None and span[0] <= offset < span[1]:
                return idx
        return None

    def local(self, offset: int, end: int):
        """``(segment_idx, start, end)`` for a span of ``text``, clamped to its segment.

        Clamped, not dropped: a phrase can run past a segment boundary where the tokenizer
        split on an abbreviation, and its head is still in the segment it starts in. Returns
        ``None`` when the start falls in no aligned segment.
        """
        idx = self.segment_of_offset(offset)
        if idx is None:
            return None
        seg_start, seg_end = self.ranges[idx]
        return idx, offset - seg_start, max(offset + 1, min(end, seg_end)) - seg_start


@dataclass
class Context:
    """Everything the adapters share: the paths, the ontology, the ground truth, one document at a time.

    Built once per run. The ontology and the ground truth are the expensive parts and both are
    process-wide by design, so holding them on the context costs nothing and makes every
    adapter's signature the same.
    """

    paths: Paths
    view: object = None
    #: An :mod:`apps.compare_ui.gold` source -- ``CuratedGold`` for HCY, ``RagHpoGold`` for GSC+.
    gold: object = None
    texts: dict = field(default_factory=dict)
    segments: dict = field(default_factory=dict)
    #: ``{name: stamp}`` accumulated by the adapters, so a bundle names every file it was built
    #: from. Adapters call :meth:`note_input`, not writing here.
    inputs: dict = field(default_factory=dict)

    @property
    def cohort(self):
        """The cohort being built."""
        return self.paths.cohort_obj

    def note_input(self, name: str, path: str) -> None:
        """Record *path* (size, time, hash) as the input *name* of this build."""
        from apps.compare_ui import bundles

        record = bundles.stamp(path, name_relative_to=self.paths.output_base)
        if record:
            self.inputs[name] = record

    def report_view(self, report_id: str):
        """The recovered, verified frame for one document -- or ``None`` if it cannot be trusted.

        ``None`` has one meaning and it is not "no data": it is "the segmentation on disk does not
        describe this document's text", which makes every offset in every method's records
        unplaceable. The builder skips such a document and names it in the index.
        """
        from apps.treephenorag_ui import evidence as evidence_mod
        from hpo_extraction.curation import spans

        text = self.texts.get(report_id)
        sentences = self.segments.get(report_id)
        if not text or not sentences:
            return None
        if not evidence_mod.aligns_to_text(sentences, text):
            logger.warning("%s: recovered segmentation does not align to the document; skipping",
                           report_id)
            return None
        ranges = spans.align_segments(text, sentences)
        display = spans.segment_texts(text, sentences, ranges)
        return ReportView(report_id=report_id, text=text, sentences=sentences,
                          display=display, ranges=ranges)


def build_context(paths: Paths, *, view=None, gold=None) -> Context:
    """Load the shared inputs once. *view* and *ground truth* are seams for the tests.

    The tests cannot build a real ``HPOTree`` -- ``hpo_extraction.ontology.hpo_tree`` reads NLTK stopwords in a
    class body, so importing it on a machine without them dies at import, not at first use.
    Passing a toy view in is what lets the whole builder be exercised without the cluster.
    """
    from hpo_extraction.curation import sources as hcy_sources

    if view is None:
        view = get_view()
    if gold is None:
        gold = load_gold(paths)

    texts = load_texts(paths)
    try:
        segments = hcy_sources.load_segments(paths.segments_path)
    except (FileNotFoundError, ValueError) as exc:
        logger.warning("no usable segmentation (%s); every document will be skipped", exc)
        segments = {}

    ctx = Context(paths=paths, view=view, gold=gold, texts=texts, segments=segments)
    ctx.note_input("segmented_reports", paths.segments_path)
    if gold is not None and getattr(gold, "path", ""):
        ctx.note_input("gold", gold.path)
    return ctx


def load_texts(paths: Paths) -> dict:
    """``{report_id: text}`` read the way this cohort's drivers read it.

    Two decoders and no third: see the module docstring. Both are the pipeline's own functions, so
    a change to how a driver reads its corpus reaches this app without anybody remembering to make
    it.
    """
    if paths.cohort == cohorts.HCY.key:
        from apps.treephenorag_ui import loaders

        return loaders.load_report_texts(paths.input_dir)

    from hpo_extraction.evaluation.datasets.gsc import load_gsc_reports

    if not os.path.isdir(os.path.join(paths.gsc_dir, "Text")):
        logger.warning("no GSC+ corpus at %s", paths.gsc_dir)
        return {}
    return load_gsc_reports(paths.gsc_dir)


def load_gold(paths: Paths):
    """This cohort's ground truth source, or ``None`` when it is not on disk."""
    from apps.compare_ui import gold as gold_mod

    if paths.cohort == cohorts.HCY.key:
        from apps.treephenorag_ui import curated as curated_mod

        gold_dir = paths.gold_dir or curated_mod.newest(paths.hcy_dir)
        dataset = curated_mod.load(gold_dir) if gold_dir else None
        return gold_mod.CuratedGold(dataset) if dataset is not None else None

    source = gold_mod.RagHpoGold(paths.raghpo_dir, paths.gsc_dir)
    return source if source.annotations else None


def frame_of(paths: Paths):
    """The sampling frame, via ``phenojury_generation_free_listing.frame`` -- or ``None``.

    An explicit ``frame_dir`` wins absolutely and does not fall back to a search. A frame is a
    record of which documents were drawn and when. Silently reading a *different* one because the
    named path was a typo would put the wrong twenty documents on screen under the right banner.

    Without one, only HCY searches: its frame has a conventional name and sits beside the
    experiment directories, whereas the GSC+ draw is named after the question it was drawn for and
    is always passed explicitly.
    """
    from apps.phenojury_ui import frame as frame_mod

    if not paths.frame_dir and paths.cohort != cohorts.HCY.key:
        return None
    found = frame_mod.find_frame(paths.output_base, paths.frame_dir or None)
    return frame_mod.load(found) if found else None
