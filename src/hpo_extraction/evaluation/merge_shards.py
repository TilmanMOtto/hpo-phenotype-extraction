"""Consolidate a sharded / resumed earlier run directory into single unsharded JSONL artifacts.

A tree experiment run as an 8-way SLURM array writes ``{variant}_s{i}of8_{kind}.jsonl`` into one
shared run directory (plus the same under each ``tau_*/``). Every consumer expects the unsharded
name instead, ``apps/ui_common/loaders.py`` and ``app/data_loader.py`` glob ``*_predictions.jsonl``
and derive the variant from the filename (eight shards would masquerade as eight variants), and
``evaluation/thesis_metrics/traversal.py`` reads ``*_nodes.jsonl``. This script merges them:

    python src/hpo_extraction/evaluation/merge_shards.py output/an earlier exploratory run/hcy

**De-duplication.** Records for one report are written as a contiguous block, and a report is
modeled only after its block is flushed, but a hard kill can still leave a partial block that
the resumed session re-does and re-appends, right next to it. The merge therefore groups lines by
``report_id`` and drops **exact duplicate lines within a report**, keeping first-occurrence order:
decoding is greedy, so the re-done records are byte-identical to the partial ones, and no artifact
contains two identical lines for the same report by design (node/prediction records are unique per
``hpo_id``, call records per ``(hpo_id, ctx_type, rank)``, traversal/timing are one line per
report). Running the script twice is a no-op (shard inputs are left in place. The merged file is
simply rewritten).

Files without a ``report_id`` (the node-metadata sidecar) are merged by de-duplicating on
``hpo_id`` instead, keeping first occurrence, those records are static per node.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import OrderedDict

SHARD_RE = re.compile(r"_s(\d+)of(\d+)(?=_)")


def unsharded_name(filename: str) -> str:
    """``tree_gate_lr_s3of8_nodes.jsonl`` → ``tree_gate_lr_nodes.jsonl`` (unchanged if unsharded)."""
    return SHARD_RE.sub("", filename)


def _blocks_by_key(lines, key_field: str):
    """``{key: [unique lines]}`` in first-seen key order, exact duplicates dropped per key.

    The duplicates this removes are the re-done records of a report whose first attempt was killed
    mid-block. They are byte-identical because generation is greedy/deterministic.
    """
    blocks: "OrderedDict[str, list]" = OrderedDict()
    seen: dict[str, set] = {}

    for line in lines:
        line = line.rstrip("\n")
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue  # truncated tail of a killed writer
        key = rec.get(key_field)
        if key is None:
            continue
        key = str(key)
        if key not in blocks:
            blocks[key] = []
            seen[key] = set()
        if line in seen[key]:
            continue
        seen[key].add(line)
        blocks[key].append(line)
    return blocks


def collect(paths: list[str]) -> tuple[int, list[str]]:
    """Read ``paths`` and return ``(n_lines_in, merged_deduplicated_lines)``."""
    lines: list[str] = []
    for p in sorted(paths):
        with open(p, "r", encoding="utf-8") as f:
            lines.extend(f.readlines())
    n_in = sum(1 for ln in lines if ln.strip())

    # Pick the identity field from the first parsable record.
    key_field = None
    for ln in lines:
        if not ln.strip():
            continue
        try:
            rec = json.loads(ln)
        except json.JSONDecodeError:
            continue
        key_field = "report_id" if "report_id" in rec else ("hpo_id" if "hpo_id" in rec else None)
        break

    if key_field == "report_id":
        blocks = _blocks_by_key(lines, "report_id")
        out_lines = [ln for block in blocks.values() for ln in block]
    elif key_field == "hpo_id":
        seen: set[str] = set()
        out_lines = []
        for ln in lines:
            if not ln.strip():
                continue
            try:
                rec = json.loads(ln)
            except json.JSONDecodeError:
                continue
            hid = str(rec.get("hpo_id"))
            if hid in seen:
                continue
            seen.add(hid)
            out_lines.append(ln.rstrip("\n"))
    else:  # unknown schema, concatenate verbatim, dropping blanks
        out_lines = [ln.rstrip("\n") for ln in lines if ln.strip()]

    return n_in, out_lines


def merge_jsonl(paths: list[str], out_path: str) -> tuple[int, int]:
    """Merge ``paths`` into ``out_path``. Returns ``(n_lines_in, n_lines_out)``."""
    n_in, out_lines = collect(paths)
    _write_lines(out_path, out_lines)
    return n_in, len(out_lines)


def _write_lines(out_path: str, lines: list[str]) -> None:
    """Write via a temp file + rename, so an interrupted merge cannot truncate a good artifact."""
    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for ln in lines:
            f.write(ln + "\n")
    os.replace(tmp, out_path)


def merge_run_dir(run_dir: str, verbose: bool = True) -> list[tuple[str, int, int]]:
    """Merge every shard group in ``run_dir`` and its ``tau_*/`` subdirectories.

    A group is one or more shard files mapping to the same unsharded name. A single resumed
    (unsharded) run is also handled: the file is de-duplicated in place, which drops any stale
    partial report block.

    When a group has shards, an unsharded file of the same name is **not** one of the inputs, see
    the comment in the loop.
    """
    results: list[tuple[str, int, int]] = []
    dirs = [run_dir] + sorted(
        os.path.join(run_dir, d) for d in os.listdir(run_dir)
        if d.startswith("tau_") and os.path.isdir(os.path.join(run_dir, d))
    )

    for d in dirs:
        groups: dict[str, list[str]] = {}
        for fname in sorted(os.listdir(d)):
            if not fname.endswith(".jsonl") or fname.startswith("."):
                continue
            groups.setdefault(unsharded_name(fname), []).append(os.path.join(d, fname))

        for target, paths in sorted(groups.items()):
            sharded = any(SHARD_RE.search(os.path.basename(p)) for p in paths)
            if sharded:
                # Drop the unsharded file from its own group. `unsharded_name` is the identity on an
                # already-unsharded name, so a leftover artifact from an earlier whole-run attempt
                # lands in the same group as the shards that supersede it, and it is also the path
                # we are about to write. Merging it in silently mixes two runs: they share
                # report_ids but their records differ line-for-line (different n_slm_calls, timings),
                # so the exact-line dedup cannot catch it and every consumer sees each report twice.
                paths = [p for p in paths if SHARD_RE.search(os.path.basename(p))]
            n_in, out_lines = collect(paths)
            # An unsharded, duplicate-free file is already what we would write, leave it
            # untouched rather than rewriting every artifact in the directory. An unsharded file
            # that *does* hold duplicates is a resumed single-worker run and does need cleaning.
            if not sharded and len(out_lines) == n_in:
                continue
            out_path = os.path.join(d, target)
            _write_lines(out_path, out_lines)
            results.append((out_path, n_in, len(out_lines)))
            if verbose:
                rel = os.path.relpath(out_path, run_dir)
                kind = f"{len(paths)} shard(s)" if sharded else "dedup"
                print(f"  {rel:<52} {kind:>12}  {n_in:>8} → {len(out_lines):>8} lines")
    return results


def main(argv=None) -> int:
    """Merge the shard files of one run folder into single files.

    Args:
        argv: command-line arguments (``run_dir`` and ``--quiet``). None reads ``sys.argv``.

    Returns:
        The exit status, 0 on success.
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_dir", help="e.g. output/exp13_00_tree_gate_lr/hcy")
    ap.add_argument("-q", "--quiet", action="store_true")
    args = ap.parse_args(argv)

    if not os.path.isdir(args.run_dir):
        print(f"no such run directory: {args.run_dir}", file=sys.stderr)
        return 2

    if not args.quiet:
        print(f"Merging shards in {args.run_dir}")
    results = merge_run_dir(args.run_dir, verbose=not args.quiet)
    if not results:
        print("nothing to merge (no *_s{i}of{N}_*.jsonl files found)")
    elif not args.quiet:
        print(f"Merged {len(results)} artifact(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
