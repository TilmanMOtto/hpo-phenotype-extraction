"""The app's read side: one bundle directory, its index, and a small LRU of parsed bundles.

The thinnest registry of any app here, and that is the payoff of the builder. Its
siblings discover experiment directories, reconcile configurations, scan gigabyte JSONLs into
byte-offset indexes and cache the result keyed on artifact mtimes. This one opens a directory of
JSON files, because everything expensive already happened somewhere else.

Two things it still owes the reader:

**Staleness.** A bundle records a digest for every artifact it was built from. When those files are
reachable from here and have moved, the header says so. It is a banner and not a refusal: the
bundle is still internally consistent, it has just stopped describing the cluster.

**Absence with a reason.** A document in the frame that is not in the bundle directory was skipped
by the builder, and the index says why. Reporting that as "no data" would make a deliberate,
recorded decision look like a missing file.

**One registry per cohort.** ``compare_bundles/hcy`` and ``compare_bundles/gsc`` are two builds
with two ground truths, two document sets and two rows of ``comparison``'s table, and the dataset picker
switches between whole registries rather than filtering one. :func:`discover` is what turns a
parent directory into that map. It reads each index's own ``cohort`` key and never infers a cohort
from a directory name.
"""

from __future__ import annotations

import logging
import os
from collections import OrderedDict

from apps.compare_ui import bundles, cohorts, roster, sources

logger = logging.getLogger(__name__)

#: Parsed bundles held in the process. A bundle is a few hundred kilobytes and a reader walks a
#: handful of reports back and forth, so this is about paging being free, not about memory.
_LRU_SIZE = 8


class Registry:
    """Everything the views read. One per process. See :mod:`apps.compare_ui.state`."""

    def __init__(self, bundles_dir: str, output_base: str = ""):
        self.bundles_dir = bundles_dir
        #: Only used to resolve the relative paths in a bundle's ``inputs`` when checking for
        #: drift, and to find ``comparison``'s tables. The app reads no experiment artifact itself.
        self.output_base = output_base or sources.DEFAULT_OUTPUT_BASE
        self._index = bundles.read_index(bundles_dir)
        self._cache: "OrderedDict[str, dict]" = OrderedDict()
        self._drift: dict = {}

    # -- the cohort ----------------------------------------------------------

    @property
    def index(self) -> dict:
        """The bundle index of this dataset."""
        return self._index or {}

    @property
    def ok(self) -> bool:
        """True when the index was read."""
        return bool(self._index)

    def problem(self) -> str:
        """Why there is nothing to show, in the words of the fix. Empty when all is well."""
        if self._index:
            return ""
        if not os.path.isdir(self.bundles_dir):
            return ("no bundle directory at {} -- run slurm/compare_ui_bundles.sbatch on "
                    "LeoMed first".format(self.bundles_dir))
        if not os.path.isfile(bundles.index_path(self.bundles_dir)):
            return ("{} has no index.json -- the build did not finish, or it wrote "
                    "somewhere else".format(self.bundles_dir))
        return ("the index at {} is unreadable or was written by another bundle schema "
                "(this app reads schema {})".format(self.bundles_dir, bundles.SCHEMA_VERSION))

    @property
    def cohort(self):
        """The cohort this build describes, from the index -- never from the directory name."""
        return cohorts.resolve(self.index.get("cohort"))

    def label(self) -> str:
        """Display name of the dataset."""
        return str(self.index.get("cohort_label") or self.cohort.label)

    def blurb(self) -> str:
        """One-line description of the cohort."""
        return str(self.index.get("cohort_blurb") or self.cohort.blurb)

    def unit(self, n=None) -> str:
        """``"report"`` / ``"abstract"``, pluralised when *n* is not 1."""
        one = str(self.index.get("cohort_unit") or self.cohort.unit)
        many = str(self.index.get("cohort_units") or self.cohort.units)
        return one if n == 1 else many

    def gold_note(self) -> str:
        """The note this cohort's ground truth requires on any screen that draws it, or ``""``."""
        return str(self.index.get("gold_note") or self.cohort.gold_note)

    def scored_cohort(self) -> str:
        """The ``cohort`` value ``comparison``'s tables use for this build."""
        return str(self.index.get("scored_cohort") or self.cohort.scored_cohort)

    def report_ids(self) -> list:
        """Reports with a bundle, in index order."""
        return [str(row.get("report_id")) for row in self.index.get("reports") or ()]

    def rows(self) -> dict:
        """``{report_id: index row}`` -- counts and cell, without opening any bundle."""
        return {str(row.get("report_id")): row for row in self.index.get("reports") or ()}

    def skipped(self) -> dict:
        """``{report_id: why}`` for reports the builder did not draw."""
        return dict(self.index.get("skipped") or {})

    def cell_of(self, report_id: str) -> str:
        """The selection cell *report_id* was drawn from, or an empty string."""
        return str((self.rows().get(str(report_id)) or {}).get("cell") or "")

    def methods(self) -> list:
        """The columns, in order, from the index -- falling back to the roster.

        The index wins because a bundle built before a roster change describes the columns it
        actually has. A view that trusted the roster instead would draw a sixth, empty column and
        blame the data.
        """
        listed = self.index.get("methods") or []
        if listed:
            return listed
        return [{"key": m.key, "label": m.label, "short": m.short, "family": m.family,
                 "evidence": m.evidence, "evidence_note": m.evidence_note} for m in roster.ROSTER]

    def method_keys(self) -> list:
        """Keys of the methods shown, in column order."""
        return [str(m.get("key")) for m in self.methods()]

    # -- one report ----------------------------------------------------------

    def bundle(self, report_id: str):
        """One parsed bundle, or ``None``. LRU'd. See :data:`_LRU_SIZE`."""
        report_id = str(report_id)
        if report_id in self._cache:
            self._cache.move_to_end(report_id)
            return self._cache[report_id]
        bundle = bundles.read(self.bundles_dir, report_id)
        if bundle is None:
            return None
        self._cache[report_id] = bundle
        while len(self._cache) > _LRU_SIZE:
            self._cache.popitem(last=False)
        return bundle

    def drift(self, report_id: str) -> list:
        """Which inputs have moved since this bundle was built. Computed once per report.

        Cached because it hashes files: on a login node where the experiment tree *is* reachable
        that is a 585 MB score cache, and doing it on every render would make paging unusable.
        """
        report_id = str(report_id)
        if report_id not in self._drift:
            bundle = self.bundle(report_id)
            self._drift[report_id] = (
                bundles.drifted(bundle, self.output_base) if bundle else [])
        return self._drift[report_id]

    # -- the quotable numbers ------------------------------------------------

    def comparison_rows(self) -> list:
        """``comparison``'s rows for **this** cohort from ``t1_overall.csv``, or ``[]``.

        Read straight out of the comparison's own table, not recomputed. The twenty
        documents here cannot support a rate -- they were drawn on the outcome -- so the only
        honest main number is the one the results chapter already prints, and printing it from
        its own file is what guarantees the two agree.

        The row is matched on ``scored_cohort``, not on the bundle's cohort key: GSC+ bundles are
        built from the ``gsc`` run directories and scored as ``gsc_raghpo_ann``, and matching on
        the wrong one of those prints the 228-document corpus's numbers under a 114-document
        heading.
        """
        import csv

        path = os.path.join(self.output_base, sources.COMPARISON_EXP, "tables", "t1_overall.csv")
        if not os.path.isfile(path):
            return []
        try:
            with open(path, newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
        except (OSError, csv.Error) as exc:
            logger.warning("could not read %s: %s", path, exc)
            return []
        wanted = set(self.method_keys())
        scored = self.scored_cohort()
        return [r for r in rows
                if str(r.get("cohort")) == scored and str(r.get("method")) in wanted]

    def comparison_path(self) -> str:
        """Path of the comparison's ``t1_overall.csv``."""
        return os.path.join(self.output_base, sources.COMPARISON_EXP, "tables", "t1_overall.csv")


# -- discovery ---------------------------------------------------------------

def discover(root: str, output_base: str = "") -> dict:
    """``{cohort key: Registry}`` for a bundle directory or a parent of several.

    Both shapes are accepted because both are reasonable things to be handed: ``--bundles`` may
    name one cohort's directory (what the old single-cohort invocation did, and what a debugging
    build writes) or the parent the builder lays cohorts out under. A directory holding an
    ``index.json`` is a cohort. Otherwise its immediate subdirectories are searched, one level and
    no deeper -- a recursive walk over an output tree would find bundle directories nobody meant
    to serve.

    Keys come from each index's own ``cohort`` field. Two builds claiming the same cohort would
    otherwise silently shadow one another, so the second is dropped with a warning, not
    quietly replacing the first.
    """
    root = str(root or "")
    found: dict = {}
    if not os.path.isdir(root):
        return found

    candidates = [root]
    if not os.path.isfile(bundles.index_path(root)):
        candidates = [os.path.join(root, name) for name in sorted(os.listdir(root))
                      if os.path.isdir(os.path.join(root, name))]

    for candidate in candidates:
        registry = Registry(candidate, output_base)
        if not registry.ok:
            continue
        key = registry.cohort.key
        if key in found:
            logger.warning("two builds claim cohort %r (%s and %s); keeping the first",
                           key, found[key].bundles_dir, candidate)
            continue
        found[key] = registry
    return found


def order(found: dict) -> list:
    """The discovered cohort keys, in the roster's own order and then anything unexpected."""
    known = [key for key in cohorts.KEYS if key in found]
    return known + sorted(set(found) - set(known))
