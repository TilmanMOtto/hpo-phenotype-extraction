#!/bin/bash
# Stage what AutoPCR's constituency-parse extraction (ee="neural++") needs and the cluster cannot
# fetch, then PROVE the set is complete by parsing real text with the network switched off.
#
# RUN THIS ON A MACHINE WITH INTERNET (the local WSL box), from the repo root:
#
#   bash experiments/06_comparison/baselines/stage_autopcr_parser_assets.sh   # -> ~/autopcr_assets/ee/
#
# The cluster's login node can `pip install` from PyPI, but its proxy 403s GitHub, and both of these
# live there:
#   * en_core_web_trf 3.7.3   the spaCy pipeline upstream loads (a wheel on GitHub, not on PyPI)
#   * benepar_en3_large       the constituency parser (a zip behind benepar's own nltk index)
# benepar_en3_large is a self-contained export: its tokenizer and T5 config ship inside the zip
# (benepar/parse_chart.py, ChartParser.from_trained), so no Hugging Face cache is needed. The probe
# below checks that claim instead of trusting it: empty HF_HOME, HF offline, and no network at all.
#
# Idempotent: downloads are skipped when the file is already there with the right checksum.
set -euo pipefail

OUT=${1:-$HOME/autopcr_assets/ee}
REPO=$(cd "$(dirname "$0")/.." && pwd)

TRF_VER=3.7.3
TRF_WHL=en_core_web_trf-${TRF_VER}-py3-none-any.whl
TRF_URL=https://github.com/explosion/spacy-models/releases/download/en_core_web_trf-${TRF_VER}/${TRF_WHL}
BENEPAR_ZIP=benepar_en3_large.zip
BENEPAR_URL=https://github.com/nikitakit/self-attentive-parser/releases/download/models/${BENEPAR_ZIP}
BENEPAR_MD5=395cf4036073b377346f59254430c2d1      # from https://kitaev.com/benepar/index.xml

mkdir -p "$OUT/wheels" "$OUT/zips"

# ---------------------------------------------------------------------------------------------
echo "[1/4] downloads"
[ -s "$OUT/wheels/$TRF_WHL" ] || curl -fL --retry 3 -o "$OUT/wheels/$TRF_WHL" "$TRF_URL"
if [ ! -s "$OUT/zips/$BENEPAR_ZIP" ] || [ "$(md5sum < "$OUT/zips/$BENEPAR_ZIP" | cut -d' ' -f1)" != "$BENEPAR_MD5" ]; then
    curl -fL --retry 3 -o "$OUT/zips/$BENEPAR_ZIP" "$BENEPAR_URL"
fi
got=$(md5sum < "$OUT/zips/$BENEPAR_ZIP" | cut -d' ' -f1)
[ "$got" = "$BENEPAR_MD5" ] || { echo "benepar zip md5 $got != $BENEPAR_MD5" >&2; exit 1; }
echo "  ok: $TRF_WHL, $BENEPAR_ZIP (md5 verified)"

# ---------------------------------------------------------------------------------------------
echo "[2/4] a local copy of autopcr_ee_venv (probe only — never transferred)"
PROBE_ENV=$OUT/probe_env
source "$(conda info --base)/etc/profile.d/conda.sh"
if [ ! -x "$PROBE_ENV/bin/python" ]; then
    conda create -y -q -p "$PROBE_ENV" python=3.10
fi
conda activate "$PROBE_ENV"
# CPU torch locally (2.5.1+cpu satisfies the ==2.5.1 pin); the cluster gets the CUDA wheel.
pip install -q --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple \
    -r "$REPO/src/AutoPCR/requirements_ee.txt"
pip install -q --no-deps "$OUT/wheels/$TRF_WHL"
pip check

NLTK_DIR=$OUT/probe_nltk_data
if [ ! -f "$NLTK_DIR/models/benepar_en3_large/benepar_model.bin" ]; then
    mkdir -p "$NLTK_DIR/models"
    python -m zipfile -e "$OUT/zips/$BENEPAR_ZIP" "$NLTK_DIR/models"   # stdlib: no unzip dependency
fi

# ---------------------------------------------------------------------------------------------
echo "[3/4] offline probe: parse two GSC+ abstracts with NO network"
# GSC+ is published abstracts, so it may be used here; HCY never leaves the cluster.
PROBE=$(mktemp -d)
trap 'rm -rf "$PROBE"' EXIT
python - "$REPO" "$PROBE/autopcr_corpus/corpus_test.tsv" <<'PY'
import os, sys
repo, corpus = sys.argv[1], sys.argv[2]
sys.path.insert(0, os.path.join(repo, "src"))
from core.autopcr_runner import stage_corpus
from evaluation.datasets.gsc import load_gsc_reports
# The loader the real runs use: its id filter drops the macOS ._ / Zone.Identifier sidecars.
reports = load_gsc_reports(os.path.join(repo, "resources", "data", "GSC_2024"))
ids = sorted(reports)[:2]
stage_corpus(reports, ids, corpus)
print("  staged", ids)
PY
mkdir -p "$PROBE/hf_empty"
# unshare -rn: a fresh network namespace with only loopback, so any fetch fails visibly.
unshare -rn env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME="$PROBE/hf_empty" \
    NLTK_DATA="$NLTK_DIR" \
    python "$REPO/src/AutoPCR/extract_phrases.py" --corpus "$PROBE/autopcr_corpus/corpus_test.tsv"
python - "$PROBE/autopcr_corpus" <<'PY'
import json, os, sys
d = sys.argv[1]
for name in ("phrases_benepar.json", "phrases_conjunct.json"):
    data = json.load(open(os.path.join(d, name)))
    n = sum(len(v["phrases"]) for v in data.values())
    print(f"  {name}: {len(data)} docs, {n} phrases")
    assert data, name
b = json.load(open(os.path.join(d, "phrases_benepar.json")))
assert all(v["phrases"] for v in b.values()), "a GSC+ abstract with zero constituents means the parse did not run"
print("  sample:", [p["phrase"] for p in next(iter(b.values()))["phrases"][:8]])
PY
conda deactivate

# ---------------------------------------------------------------------------------------------
echo "[4/4] stage to the cluster"
CL=<LEOMED_GROUP_DIR>/autopcr_ee
cat <<EOF
  All assets proven sufficient offline. Copy them (~1.2 GB):

    rsync -avh --progress $OUT/wheels/ $OUT/zips/ <euler>:$CL/staged/

  then on the login node:

    the parser environment steps in third_party/README.md
EOF
