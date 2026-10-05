"""Which earlier runs exist on disk, and which files each (method, cohort, configuration) has.

Everything downstream is driven from here. The layout the earlier drivers write is three levels
deep and the older UIs assume one, which is why ``app/tree_ui`` sees nothing at all when pointed
at an earlier output folder::

    <output_base>/<exp_id>/<cohort>/                       ← the "run directory"
        {variant}{sfx}_calls.jsonl                         tau-independent
        {variant}{sfx}_node_metadata.jsonl                 tau-independent
        tau_{t:g}[_acc_{a:g}]/                             ← one per swept configuration
            {variant}{sfx}_nodes.jsonl
            {variant}{sfx}_predictions.jsonl
            {variant}{sfx}_traversal.jsonl
            {variant}{sfx}_timing.jsonl

``sfx`` is ``_s{i}of{N}`` until ``src/hpo_extraction/evaluation/merge_shards.py`` collapses the SLURM array's
output. Two rules follow from that, and both are essential:

**Merged wins, absolutely.** ``merge_shards.py`` leaves the shard files in place next to the
merged one. Reading both would double-count every record. Whenever the merged name exists, the
shards it was built from are ignored, not warned about, ignored.

**Unmerged shards are read, not refused.** ``result_tables`` refuses to compute a metric from unmerged
shards, and it is right to: a thesis table that silently describes 1/8 of a cohort is worse than a
blank. A diagnostic UI has the opposite obligation, the run you most want to look at is often the
one still finishing, so unmerged shards are unioned and the cell is flagged ``shards`` in every
header, with the shard count shown. What is never done is guessing: a shard set missing members of
its own ``of{N}`` declaration is reported as ``shards_incomplete`` with the missing indices named.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

# Copied from ``src/hpo_extraction/evaluation/merge_shards.py:35``. The lookahead is essential: consuming the
# trailing _ character would derive ``flat_topmpredictions.jsonl`` as the merged name, which never
# exists, so every merged run would be misreported as unmerged.
SHARD_RE = re.compile(r"_s(\d+)of(\d+)(?=_)")

COHORTS: tuple[str, ...] = ("hcy", "gsc")

#: Which artifact kinds a method family writes, and where. ``cohort`` files sit directly in the
#: run directory; ``op`` files sit inside each operating-point subdirectory.
_KIND_FILES: dict[str, dict[str, tuple[str, ...]]] = {
    "tree": {
        "cohort": ("calls", "node_metadata"),
        "op": ("nodes", "predictions", "traversal", "timing"),
    },
    "raghpo": {
        "cohort": ("predictions", "retrieved_segments"),
        "op": (),
    },
    "flat_topm": {
        "cohort": ("calls", "predictions", "timing"),
        "op": (),
    },
    "ensemble": {
        "cohort": ("detections",),
        "op": ("predictions",),
    },
    "phenobert": {
        "cohort": ("predictions", "detections", "timing"),
        "op": (),
    },
}

#: Artifact kinds without which a cell cannot be scored at all.
_REQUIRED = {"tree": ("nodes", "predictions"), "raghpo": ("predictions",),
             "flat_topm": ("predictions",), "ensemble": ("predictions",),
             "phenobert": ("predictions",)}


@dataclass(frozen=True)
class MethodSpec:
    """One method of the earlier comparison.

    ``variant`` is the filename prefix the driver writes: ``tree_experiment`` derives it from
    ``build_scores`` (so it differs per experiment), while the other three drivers hardcode it.
    ``kind`` selects the loader family; ``sweep`` globs the operating-point subdirectories, or is
    ``None`` when the run writes straight into the cohort directory.

    ``is_tree`` gates the pruning views. It is a property of the method's *design*, not of its
    files: a method with no traversal has no pruning decision to autopsy, and showing it an empty
    blocking-depth histogram would imply it had one that never fired.
    """

    key: str
    exp_id: str
    variant: str
    kind: str                      # "tree" | "raghpo" | "flat_topm" | "ensemble" | "phenobert"
    label: str
    sweep: str | None = None
    sweep_axis: str | None = None

    @property
    def is_tree(self) -> bool:
        """True for TreePhenoRAG runs."""
        return self.kind == "tree"


METHODS: tuple[MethodSpec, ...] = (
    MethodSpec("tree_gate_lr", "exp13_00_tree_gate_lr", "tree_gate_lr", "tree",
               "TreePhenoRAG (LR gate)", "tau_*", "tau_prune"),
    MethodSpec("tree_noisyor", "exp13_01_tree_noisyor", "tree_noisyor", "tree",
               "TreePhenoRAG (noisyOR)", "tau_*", "tau_prune"),
    MethodSpec("tree_no_prune", "exp13_02_tree_no_prune", "tree_no_prune", "tree",
               "TreePhenoRAG (single threshold)", "tau_*", "tau_prune"),
    MethodSpec("tree_lse_beta1", "exp13_09_tree_lse_beta1", "tree_lse_beta1", "tree",
               "TreePhenoRAG (lse_beta1)", "tau_*", "tau_prune"),
    MethodSpec("raghpo_8b", "baseline_raghpo_8b", "rag_hpo", "raghpo",
               "RAG-HPO (LLaMA-8B)"),
    MethodSpec("raghpo_70b", "baseline_raghpo_70b", "rag_hpo", "raghpo",
               "RAG-HPO (LLaMA-70B)"),
    MethodSpec("flat_topm", "exp13_05_topm_retrieval_slm", "flat_topm", "flat_topm",
               "Flat top-M + SLM"),
    MethodSpec("slm_ensemble", "phenojury_generation_free_listing", "slm_ensemble", "ensemble",
               "SLM ensemble (8 models)", "vote_k*", "vote_k"),
    MethodSpec("phenobert", "baseline_phenobert", "phenobert", "phenobert",
               "PhenoBERT"),
)

METHODS_BY_KEY: dict[str, MethodSpec] = {m.key: m for m in METHODS}
TREE_KEYS: tuple[str, ...] = tuple(m.key for m in METHODS if m.is_tree)


@dataclass
class FileSet:
    """The resolved paths for one artifact kind, plus how they were resolved.

    ``paths`` is what a loader should read, one merged file, or the union of the shards. Empty
    means the artifact is absent.
    """

    kind: str
    paths: list[str] = field(default_factory=list)
    shard_state: str = "missing"    # merged | shards | shards_incomplete | missing
    n_shards: int = 0
    missing_shards: list[int] = field(default_factory=list)
    n_bytes: int = 0

    @property
    def present(self) -> bool:
        """True when at least one file was found."""
        return bool(self.paths)

    @property
    def empty(self) -> bool:
        """The file exists and holds nothing.

        A real state on the cluster, not a hypothetical: several ``tau_*`` directories were
        created by the driver and then never written, because the array was killed before it
        reached that configuration. Distinguishing this from ``missing`` counts, the
        directory being there is what makes the point appear in the sweep dropdown at all.
        """
        return bool(self.paths) and self.n_bytes == 0


@dataclass
class Cell:
    """One (method, cohort) run directory and everything discovered inside it."""

    spec: MethodSpec
    cohort: str
    run_dir: str
    operating_points: list[str] = field(default_factory=list)
    cohort_files: dict[str, FileSet] = field(default_factory=dict)
    op_files: dict[str, dict[str, FileSet]] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)

    @property
    def cell_id(self) -> str:
        """The dropdown value, stable, and parseable back into (method, cohort)."""
        return f"{self.spec.key}/{self.cohort}"

    @property
    def label(self) -> str:
        """Display label, method and cohort."""
        return f"{self.spec.label} · {self.cohort.upper()}"

    @property
    def status(self) -> str:
        """``ok`` | ``shards`` | ``partial`` | ``missing``, what a header badge should say."""
        required = _REQUIRED[self.spec.kind]
        resolved = self._required_filesets()
        if not resolved or any(not fs.present for fs in resolved):
            return "missing" if not any(fs.present for fs in resolved) else "partial"
        if any(fs.shard_state == "shards_incomplete" for fs in resolved):
            return "partial"
        if any(fs.shard_state == "shards" for fs in resolved):
            return "shards"
        return "ok"

    def _required_filesets(self) -> list[FileSet]:
        out: list[FileSet] = []
        for kind in _REQUIRED[self.spec.kind]:
            if kind in _KIND_FILES[self.spec.kind]["cohort"]:
                out.append(self.cohort_files.get(kind, FileSet(kind)))
            elif self.operating_points:
                out.extend(self.op_files.get(op, {}).get(kind, FileSet(kind))
                           for op in self.operating_points)
            else:
                out.append(FileSet(kind))
        return out

    @property
    def written_operating_points(self) -> list[str]:
        """Configurations whose predictions artifact actually holds something.

        The default the UI opens on must come from this list, not from ``operating_points``:
        ``tau_0.02`` sorts first on the real gate-LR run and its directory is empty, so defaulting
        to the first point would greet every user with a blank scorecard on a run that is fine.
        """
        written = [op for op in self.operating_points
                   if not self.op_files.get(op, {}).get("predictions", FileSet("predictions")).empty]
        return written or list(self.operating_points)

    @property
    def empty_operating_points(self) -> list[str]:
        """Configurations declared by the run that wrote no predictions."""
        return [op for op in self.operating_points if op not in set(self.written_operating_points)]

    def files_for(self, kind: str, op: str | None = None) -> FileSet:
        """The FileSet for one artifact kind, at an configuration when the kind lives there."""
        if kind in _KIND_FILES[self.spec.kind]["cohort"]:
            return self.cohort_files.get(kind, FileSet(kind))
        if op is None:
            op = self.operating_points[0] if self.operating_points else None
        return self.op_files.get(op, {}).get(kind, FileSet(kind))


def _resolve(directory: str, variant: str, kind: str) -> FileSet:
    """Resolve ``{variant}[_sXofN]_{kind}.jsonl`` in one directory into a FileSet.

    Merged beats shards unconditionally (see the module docstring). Shard sets are checked
    against their own declared ``of{N}``, so a half-finished array is visible as such rather than
    silently under-reporting the cohort.
    """
    fs = FileSet(kind)
    if not os.path.isdir(directory):
        return fs

    merged_name = f"{variant}_{kind}.jsonl"
    merged_path = os.path.join(directory, merged_name)
    if os.path.isfile(merged_path):
        fs.paths = [merged_path]
        fs.shard_state = "merged"
        fs.n_bytes = os.path.getsize(merged_path)
        return fs

    shards: dict[int, str] = {}
    declared: set[int] = set()
    prefix, suffix = f"{variant}_s", f"_{kind}.jsonl"
    for name in sorted(os.listdir(directory)):
        if not (name.startswith(prefix) and name.endswith(suffix)):
            continue
        if SHARD_RE.sub("", name) != merged_name:
            continue  # a different variant that happens to share a prefix
        match = SHARD_RE.search(name)
        if not match:
            continue
        index, total = int(match.group(1)), int(match.group(2))
        shards[index] = os.path.join(directory, name)
        declared.add(total)

    if not shards:
        return fs

    fs.paths = [shards[i] for i in sorted(shards)]
    fs.n_shards = len(shards)
    fs.n_bytes = sum(os.path.getsize(p) for p in fs.paths)
    # Disagreeing ``of{N}`` declarations mean two different array sizes wrote here. Take the
    # largest, which is the only choice that cannot understate what is missing.
    total = max(declared) if declared else len(shards)
    missing = [i for i in range(total) if i not in shards]
    fs.missing_shards = missing
    fs.shard_state = "shards_incomplete" if missing else "shards"
    return fs


#: Tree operating-point directory names. Two layouts coexist by design, and both are on disk:
#: a run that fixes one accept threshold keeps the 1-D ``tau_{prune}`` (an earlier exploratory run/01/02, written
#: before the accept sweep existed), while a real 2-D sweep adds ``_acc_{accept}``
#: (an earlier exploratory run, 5 τ_prune × 5 τ_accept = 25 directories). Copied from
#: ``src/hpo_extraction/evaluation/result_tables/discovery.py:311`` so the UI and the thesis tables parse
#: The same names. The character class admits ``1e-05``, which ``%g`` emits for a small τ.
_TAU_DIR_RE = re.compile(r"^tau_(?P<prune>[0-9.eE+-]+)(?:_acc_(?P<accept>[0-9.eE+-]+))?$")


def _operating_points(run_dir: str, spec: MethodSpec) -> list[str]:
    """The operating-point subdirectory names, sorted by their numeric axis where they have one.

    ``tau_0.02`` must sort before ``tau_0.1``. Lexicographic order puts it after ``tau_0.1``, and
    a τ frontier plotted in that order is a scribble. A 2-D tree sweep sorts as a grid, τ_prune
    major, so consecutive points differ in the accept threshold only. ``agg_plurality`` has no
    number and is parked at the end.

    Tree directories that do not parse as an configuration are dropped, not guessed at:
    ``tau_*`` also matches a stray path, and one of those in the sweep would put a meaningless
    point on the frontier.
    """
    if not spec.sweep or not os.path.isdir(run_dir):
        return []
    stem = spec.sweep.rstrip("*")
    names = [n for n in os.listdir(run_dir)
             if os.path.isdir(os.path.join(run_dir, n)) and n.startswith(stem)]
    if spec.kind == "tree":
        names = [n for n in names if _TAU_DIR_RE.match(n)]
    elif spec.kind == "ensemble":
        # vote_k* plus the sibling agg_plurality, which the glob does not cover.
        extra = os.path.join(run_dir, "agg_plurality")
        if os.path.isdir(extra):
            names.append("agg_plurality")
    return sorted(names, key=_op_sort_key)


def _op_sort_key(name: str) -> tuple[int, float, float, str]:
    """(has-no-number, τ_prune, τ_accept, name), numeric points first, in grid order.

    A 1-D point sorts ahead of every 2-D point sharing its τ_prune, which is the only ordering
    under which a directory written before the accept sweep existed and one written after it read
    as the same sweep.
    """
    value = op_value(name)
    if value is None:
        return (1, 0.0, 0.0, name)
    accept = op_accept(name)
    return (0, value, -1.0 if accept is None else accept, name)


def op_value(name: str) -> float | None:
    """The **primary** axis of an configuration, τ_prune for a tree, or ``None``.

    ``tau_0.02`` → 0.02, ``tau_0.02_acc_0.9`` → 0.02, ``vote_k3`` → 3.0,
    ``agg_plurality`` → ``None``.

    τ_prune and not τ_accept for the 2-D case,: this value is the frontier's x-axis
    and the τ grid the culprit leaderboard asks "which swept threshold would have opened this
    node" against. Both questions are about the threshold that decides *expansion*; τ_accept
    changes no traversal, only what is kept from one, :func:`op_accept` reports it separately.
    """
    match = _TAU_DIR_RE.match(name)
    if match:
        return float(match.group("prune"))
    match = re.search(r"(\d+(?:\.\d+)?)$", name)
    return float(match.group(1)) if match else None


def op_accept(name: str) -> float | None:
    """The τ_accept of a 2-D tree configuration; ``None`` for every other name.

    ``None`` means "this point does not vary the accept threshold", which is true both of a 1-D
    ``tau_0.5`` and of ``vote_k3``, neither belongs to an accept slice.
    """
    match = _TAU_DIR_RE.match(name)
    return float(match.group("accept")) if match and match.group("accept") else None


def op_label(name: str) -> str:
    """How an configuration is named in a dropdown. Identity unless it has two axes.

    ``tau_0.00015_acc_0.9`` is unreadable in a 25-item list; ``τp 0.00015 · τa 0.9`` is not.
    The dropdown *value* stays the directory name, it is the cache key and the path.
    """
    accept = op_accept(name)
    if accept is None:
        return name
    return f"τp {op_value(name):g} · τa {accept:g}"


def accept_slice(operating_points: list[str], op: str | None) -> list[str]:
    """The points sharing *op*'s τ_accept, i.e. the 1-D τ_prune sweep running through it.

    A 2-D sweep is a grid, and a frontier drawn over the whole grid is not a frontier: five points
    share each τ_prune, so every curve doubles back on itself. Slicing at the selected τ_accept
    gives back the curve the plot is for, and costs a quarter of an hour less on first open, the
    frontier scores every point it shows, and an earlier exploratory run has 25 of them.

    Returns *operating_points* unchanged when there is no accept axis to slice on.
    """
    accept = op_accept(op) if op else None
    if accept is None:
        return list(operating_points)
    return [p for p in operating_points if op_accept(p) == accept]


def discover(output_base: str) -> list[Cell]:
    """Every (method, cohort) cell present under ``output_base``, in METHODS order.

    A cell is returned as soon as its run directory exists, even when nothing usable is inside, the sidebar must be able to say "the directory is there and empty", which is a different
    diagnosis from "the job never ran".
    """
    cells: list[Cell] = []
    if not os.path.isdir(output_base):
        return cells

    for spec in METHODS:
        for cohort in COHORTS:
            run_dir = os.path.join(output_base, spec.exp_id, cohort)
            if not os.path.isdir(run_dir):
                continue
            cell = Cell(spec=spec, cohort=cohort, run_dir=run_dir)
            for kind in _KIND_FILES[spec.kind]["cohort"]:
                cell.cohort_files[kind] = _resolve(run_dir, spec.variant, kind)
            cell.operating_points = _operating_points(run_dir, spec)
            for op in cell.operating_points:
                op_dir = os.path.join(run_dir, op)
                cell.op_files[op] = {
                    kind: _resolve(op_dir, spec.variant, kind)
                    for kind in _KIND_FILES[spec.kind]["op"]
                }
            cell.problems = _problems(cell)
            cells.append(cell)
    return cells


def _problems(cell: Cell) -> list[str]:
    """Human-readable warnings for one cell, shown, never silently corrected."""
    out: list[str] = []
    if cell.spec.sweep and not cell.operating_points:
        out.append(f"no {cell.spec.sweep} subdirectory found in {cell.run_dir}")

    empty = cell.empty_operating_points
    if empty:
        out.append(
            f"{len(empty)} configuration(s) exist but were never written: {' '.join(empty)}, "
            f"the run was stopped before reaching them; they are excluded from the default view")

    seen: set[str] = set()
    for scope, filesets in [("", cell.cohort_files)] + [
        (f"{op}/", cell.op_files[op]) for op in cell.operating_points
    ]:
        for kind, fs in filesets.items():
            if fs.shard_state == "shards_incomplete":
                out.append(
                    f"{scope}{cell.spec.variant}_{kind}: {fs.n_shards} shard(s) present, "
                    f"missing indices {fs.missing_shards}, the cohort is incomplete"
                )
            elif fs.shard_state == "shards" and kind not in seen:
                out.append(
                    f"{scope}{cell.spec.variant}_{kind}: reading {fs.n_shards} unmerged shards; "
                    f"run `python src/hpo_extraction/evaluation/merge_shards.py {cell.run_dir}` to collapse them"
                )
                seen.add(kind)
            elif not fs.present and kind in _REQUIRED[cell.spec.kind]:
                out.append(f"{scope}{cell.spec.variant}_{kind}.jsonl is missing")
    return out


def summarize(cells: list[Cell]) -> list[dict]:
    """A flat table of what was discovered, the body of ``--selfcheck`` and the sidebar list."""
    return [
        {
            "method": c.spec.key,
            "label": c.spec.label,
            "experiment": c.spec.exp_id,
            "cohort": c.cohort,
            "kind": c.spec.kind,
            "status": c.status,
            "n_operating_points": len(c.operating_points),
            "operating_points": " ".join(c.operating_points),
            "problems": "; ".join(c.problems),
        }
        for c in cells
    ]
