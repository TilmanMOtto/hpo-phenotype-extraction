#!/bin/bash
# Generates the synthetic sentences, 40 per HPO term, with Llama-3.3-70B-Instruct behind an
# OpenAI-compatible gateway (thesis appendix B.8). Needs outbound internet, so it runs on a machine
# that has it, not on a compute node. No patient data is involved.
#   export LLM_GATEWAY_URL=<gateway URL> LITELLM_API_KEY=<your key>
#   bash slurm/synthetic_sentences.sh
set -e
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
: "${LLM_GATEWAY_URL:?set LLM_GATEWAY_URL}" "${LITELLM_API_KEY:?set LITELLM_API_KEY}"
python experiments/04_treephenorag/synthetic_sentences/run.py "$@"
