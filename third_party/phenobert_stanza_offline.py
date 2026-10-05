"""Patch Stanza 1.4.1 in PhenoBERT's environment so that it never downloads.

    <phenobert env>/bin/python third_party/phenobert_stanza_offline.py

PhenoBERT needs Stanza 1.4.1, and that version downloads ``resources.json`` on every start and
downloads each model file again unless its MD5 matches. Compute nodes have no outbound internet, so
both downloads fail. The two patches below make Stanza use any local ``resources.json`` and any local
model file as they are. The script edits ``stanza/resources/common.py`` of the interpreter that runs
it, and running it twice changes nothing. Afterwards ``grep -c "Air-gapped" <that file>`` prints 2.
"""
from __future__ import annotations

import re
import sys

GUARD_A = (
    "    # Air-gapped: use local resources.json if present, skip download.\n"
    "    import os as _os\n"
    "    if _os.path.exists(_os.path.join(model_dir, 'resources.json')):\n"
    "        logger.debug('resources.json already exists, skipping download.')\n"
    "        return\n"
)
OLD_B = """    if file_exists(path, md5):
        if log_info:
            logger.info(f'File exists: {path}')
        else:
            logger.debug(f'File exists: {path}')
        return"""
NEW_B = """    # Air-gapped: skip download if file exists locally (skip MD5 check).
    if os.path.exists(path):
        if log_info:
            logger.info(f'File exists: {path}')
        else:
            logger.debug(f'File exists: {path}')
        return"""


def patch(src: str) -> str:
    """*src* (the text of ``stanza/resources/common.py``) with both patches applied."""
    if "resources.json already exists, skipping download" not in src:
        m = re.search(r"def download_resources_json\(.*?\):\n", src, re.DOTALL)
        if not m:
            raise SystemExit("patch A: download_resources_json not found, is this Stanza 1.4.1?")
        src = src[:m.end()] + GUARD_A + src[m.end():]
    if NEW_B not in src:
        if OLD_B not in src:
            raise SystemExit("patch B: the file_exists guard of request_file not found, is this Stanza 1.4.1?")
        src = src.replace(OLD_B, NEW_B, 1)
    return src


def main() -> int:
    """Patch the Stanza of the running interpreter. Exit status 0."""
    import stanza.resources.common as common

    path = common.__file__
    with open(path, encoding="utf-8") as fh:
        src = fh.read()
    new = patch(src)
    if new != src:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(new)
    print(f"patched: {path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
