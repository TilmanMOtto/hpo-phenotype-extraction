#!/usr/bin/env python
"""Scan the tracked files for patient identifiers, cluster paths, user names and excluded words.

    python tools/scan_repository.py [--words] [--show N]

Checks every file ``git ls-files`` lists:

1. HCY report identifiers (``HCY`` followed by digits).
2. Absolute cluster or home paths (``/cluster/``, ``/home/``) outside ``configs/cluster_leomed.yaml``,
   outside URLs, and outside upstream code copied unchanged (``UPSTREAM_CODE``).
3. User names, a notification service and e-mail domains, outside
   ``configs/cluster_leomed.yaml`` (whose group folder carries a user name).
4. With ``--words``: the words and punctuation the documentation rules exclude, in prose only.
   Prose is Markdown outside code blocks and code spans, Python comments and docstrings, and YAML
   comments. Prompt templates, upstream documents of copied code, and names in code font are not
   prose.

Prints one line per hit (up to ``--show`` per check) and a count per check. Exit status 1 when
checks 1 to 3 find anything.
"""
from __future__ import annotations

import argparse
import ast
import io
import re
import subprocess
import sys
import tokenize
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

def _user_names() -> list[str]:
    """The cluster account (last folder of ``group_dir`` in the path file) and the local login."""
    import getpass
    names = {getpass.getuser()}
    for line in (REPO / "configs" / "cluster_leomed.yaml").read_text(encoding="utf-8").splitlines():
        if line.startswith("group_dir:"):
            names.add(line.split(":", 1)[1].strip().rstrip("/").rsplit("/", 1)[-1])
    return sorted(n for n in names if n)


PATTERNS = {
    "report identifier": re.compile(r"\bHCY[0-9]{2,}\b"),
    "absolute path": re.compile(r"/cluster/|/home/"),
    # The user names are read at run time, so this file does not have to contain them.
    "user name or contact": re.compile("|".join([rf"\b{re.escape(n)}\b" for n in _user_names()]
                                                + [r"ntfy", r"@ethz", r"@student"]), re.IGNORECASE),
}
PATH_FILE = "configs/cluster_leomed.yaml"
#: Upstream code copied with its own authors' text (third_party/README.md).
UPSTREAM_CODE = ("third_party/PhenoBERT/phenobert/", "third_party/RAG-HPO/")
#: Files that list the patterns themselves.
SELF = {"tools/scan_repository.py"}

BANNED = [
    r"gold terms?", r"gold pairs?", r"gold sets?", r"the gold\b(?![-_ ]standard)", r"deployment envelope",
    r"operating points?", r"fold-row", r"reachable recall", r"scorable ceiling", r"\breplay\w*",
    r"\banchor\w*", r"\bexemplar\w*", r"\w+-centric", r"\barms?\b", r"lexical route", r"surface forms?",
    r"\binherit(s|ed|ing)?\b", r"\bcheckpoint\w*", r"\baffirms?\b", r"exposed to", r"by construction",
    r"out of its reach", r"at the price of", r"\bpinned\b", r"\bfrozen\b", r"\bsnapshot\w*",
    r"pre-registered", r"\bheadline\w*", r"\bexactly\b", r"\bprecisely\b", r"\bdeliberately\b",
    r"the one exception", r"\bmatters\b", r"\bcaveats?\b", r"\beffectively\b", r"\bunderscores?\b",
    r"\bnuanced\b", r"\bcrucial\b", r"\brobust\w*", r"\bnovel\b", r"\bimpressive\b", r"\boutstanding\b",
    r"\bseamless\w*", r"\bleverag\w*", r"load-bearing", r"is not a detail", "—",
]
BANNED_RE = re.compile("|".join(BANNED), re.IGNORECASE)
SEMICOLON_RE = re.compile(r"[a-z)]; [a-z]")
#: Not prose: upstream documents, and files whose content is prompts or recorded data.
WORD_SKIP = ("third_party/RAG-HPO/", "third_party/PhenoBERT/phenobert/", "third_party/PhenoBERT/README.md",
             "resources/", "tests/prompt_hashes.json", "src/hpo_extraction/phenojury/prompts.py",
             "src/hpo_extraction/treephenorag/verifier_prompt.py", "third_party/AutoPCR/utils/prompts.py",
             "experiments/04_treephenorag/synthetic_sentences/run.py", "docs/stored_formats.md")
CODE_SPAN = re.compile(r"``[^`]*``|`[^`\n]*`")
#: Column names written into result files, which keep their names (docs/stored_formats.md) but
#: can sit in a code span that wraps across lines.
STORED_NAMES = {"headline_selection"}


def tracked() -> list[str]:
    """Paths of every tracked file."""
    out = subprocess.run(["git", "ls-files"], cwd=REPO, check=True, capture_output=True, text=True)
    return out.stdout.splitlines()


def prose_lines(path: Path) -> list[tuple[int, str]]:
    """``(line number, text)`` of the prose in *path* (see the module docstring)."""
    text = path.read_text(encoding="utf-8", errors="replace")
    out: list[tuple[int, str]] = []
    if path.suffix == ".md":
        fence = False
        for n, line in enumerate(text.splitlines(), 1):
            if line.lstrip().startswith("```"):
                fence = not fence
            elif not fence:
                out.append((n, line))
    elif path.suffix in (".yaml", ".yml", ".sh", ".sbatch"):
        for n, line in enumerate(text.splitlines(), 1):
            idx = line.find("#")
            if idx >= 0 and (idx == 0 or line[idx - 1] in " \t") and not line.startswith("#!") \
                    and not line.startswith("#SBATCH"):
                out.append((n, line[idx:]))
    elif path.suffix == ".py":
        try:
            docs = set()
            for node in ast.walk(ast.parse(text)):
                if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) \
                        and node.body and isinstance(node.body[0], ast.Expr) \
                        and isinstance(getattr(node.body[0], "value", None), ast.Constant) \
                        and isinstance(node.body[0].value.value, str):
                    docs.add((node.body[0].value.lineno, node.body[0].value.col_offset))
            for tok in tokenize.generate_tokens(io.StringIO(text).readline):
                if tok.type == tokenize.COMMENT or (tok.type == tokenize.STRING and tok.start in docs):
                    for i, line in enumerate(tok.string.splitlines()):
                        out.append((tok.start[0] + i, line))
        except (SyntaxError, tokenize.TokenError):
            pass
    return out


def main() -> int:
    """Run the checks and print the hits. Exit status 1 when checks 1 to 3 find anything."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--words", action="store_true", help="also check the documentation rules")
    ap.add_argument("--show", type=int, default=20, help="hits printed per check")
    args = ap.parse_args()
    files = [f for f in tracked() if (REPO / f).is_file()]
    failed = 0
    for name, pattern in PATTERNS.items():
        hits = []
        for rel in files:
            # The path file is the one place for cluster paths, and it names the group folder.
            if rel in SELF or (name != "report identifier" and rel == PATH_FILE):
                continue
            # Upstream code copied unchanged carries its authors' paths (a shebang), not ours.
            if name == "absolute path" and rel.startswith(UPSTREAM_CODE):
                continue
            try:
                text = (REPO / rel).read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            for n, line in enumerate(text.splitlines(), 1):
                # A URL is not a file path (``hp.obo`` cites web pages under /home/).
                if pattern.search(re.sub(r"\w+://\S+", "", line)):
                    hits.append(f"{rel}:{n}: {line.strip()[:120]}")
        print(f"== {name}: {len(hits)} hit(s)")
        for hit in hits[:args.show]:
            print("  " + hit)
        failed += bool(hits)
    if args.words:
        hits, semis = [], []
        for rel in files:
            if rel.startswith(WORD_SKIP) or rel in SELF or not rel.endswith((".md", ".py", ".yaml", ".sh", ".sbatch")):
                continue
            for n, line in prose_lines(REPO / rel):
                plain = CODE_SPAN.sub("", line)
                for m in BANNED_RE.finditer(plain):
                    if m.group(0) in STORED_NAMES:
                        continue
                    hits.append(f"{rel}:{n}: {m.group(0)!r} in {line.strip()[:100]}")
                if rel.endswith(".md") and SEMICOLON_RE.search(plain):
                    semis.append(f"{rel}:{n}: {line.strip()[:100]}")
        print(f"== excluded words and em dashes in prose: {len(hits)} hit(s)")
        for hit in hits[:args.show]:
            print("  " + hit)
        print(f"== semicolons in Markdown prose: {len(semis)} hit(s)")
        for hit in semis[:args.show]:
            print("  " + hit)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
