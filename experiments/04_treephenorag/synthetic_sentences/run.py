"""
Generate clinical context sentences for HPO terms using LLaMA 3.3 70B via LiteLLM API.


Replicates the contextual database generation described in:
  Berndt et al., 2025, Section 3.1 + Figure 3 / Appendix A.1

Key differences from an earlier exploratory run (predecessor):
  - Model: openrouter/meta-llama/llama-3.3-70b-instruct via LiteLLM gateway
  - Execution: parallel synchronous calls via ThreadPoolExecutor (n_workers configurable)
  - Cost tracking: response.usage.cost per call (provided directly by LiteLLM)
  - Resume: file-level skip (re-run skips already-written .txt files)
  - Seed: passed to API for reproducibility

⚠️  Parameters NOT specified in the paper (flagged assumptions, same as an earlier exploratory run):
  - System prompt: instruction text placed in system message
  - Temperature: model default (not set)
  - max_tokens: 2048 used as safe upper bound
  - Bullet parsing: strips leading "- ", "• ", "* ", "· "
  - HPO Comments field: node.comment[0] from hpo.json
  - HPO Synonyms field: node.synonym list joined with ", "

⚠️  Note on seed reproducibility:
  The `seed` parameter is forwarded to the API. LiteLLM/OpenRouter passes it to the
  upstream provider on a best-effort basis. Exact reproducibility depends on the
  provider. Outputs may still vary slightly across runs.

Usage:
    LITELLM_API_KEY=<key> python experiments/synthetic_sentences/run.py

Override output dir:
    LITELLM_API_KEY=<key> python experiments/synthetic_sentences/run.py \\
        output_dir=resources/context_data/context_HCY_llama_api

Resume interrupted run:
    Re-run the same command, already-written .txt files are skipped automatically.
"""

import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig
from tqdm import tqdm


from hpo_extraction.ontology.hpo_tree import HPO_class, HPOTree

try:
    from openai import OpenAI
except ImportError as e:
    raise ImportError(
        "openai package not installed. Run: pip install openai"
    ) from e

log = logging.getLogger(__name__)

# Shared instruction, identical for all HPOs → placed in system message
_INSTRUCTION_TEMPLATE = (
    "Generate {n} unique sentences that describe the provided HPO label as they "
    "would appear in a clinical narrative or interpretive report. Each sentence should "
    "offer a different perspective or detail, similar to how various clinicians might "
    "report observations or diagnoses in clinical notes. Avoid using the exact phrase "
    "in every sentence to ensure diversity and reflect the range of clinical expression. "
    "Sometimes, I will provide comments, synonyms, and definitions that can help "
    "provide more context for your thoughts. Return a bulleted list instead of numbered. "
    "Feel free to use clinical shorthand that a physician might use in writing reports."
)


def _build_user_content(node: HPO_class) -> str:
    """HPO-specific user message content."""
    label = node.name[0] if node.name else node.id
    definition = node.definition[0] if node.definition else ""
    comments = node.comment[0] if node.comment else ""
    synonyms = ", ".join(node.synonym) if node.synonym else ""
    return (
        f"HPO label: {label}\n"
        f"Definition: {definition}\n"
        f"Comments: {comments}\n"
        f"Synonyms: {synonyms}"
    )


def _parse_bullet_list(text: str, n: int) -> list[str]:
    """
    Parse the bulleted list returned by the LLM.
    ⚠️  Exact parsing logic not specified in paper, strips common bullet markers.
    """
    sentences = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^[-•*·]\s+", "", line).strip()
        if line:
            sentences.append(line)
    return sentences[:n]


def _write_context_file(path: Path, sentences: list[str]) -> None:
    """
    Write sentences in the asterisk-delimited format expected by prepare_context()
    in src/hpo_extraction/data/loading.py. Uses atomic rename to avoid partial files.
    """
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for sentence in sentences:
            f.write(f"* {sentence}\n")
    os.replace(tmp, path)


@hydra.main(
    config_path="../../../configs/experiments/04_treephenorag",
    config_name="synthetic_sentences",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    """Entry point: run with the Hydra config named in the decorator (see the config header for the command)."""
    # --- Load HPO IDs from CSV ---
    csv_path = Path(cfg.target_symptoms_csv)
    df = pd.read_csv(csv_path)
    hpo_ids: list[str] = df["target_codes"].dropna().tolist()
    log.info("Loaded %d HPO IDs from %s", len(hpo_ids), csv_path)

    # --- Load ontology ---
    hpo_tree = HPOTree(cfg.hpo_json)

    # --- Output directory ---
    model_slug = cfg.model.split("/")[-1]
    out_dir = Path(cfg.output_dir) / model_slug
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Filter HPOs that already have context files ---
    todo: list[str] = []
    skipped = 0
    unknown = 0
    for hpo_id in hpo_ids:
        filename = hpo_id.replace(":", "_") + ".txt"
        if (out_dir / filename).exists():
            skipped += 1
            continue
        if hpo_id not in hpo_tree.data:
            log.warning("HPO ID %s not in hpo.json — skipping", hpo_id)
            unknown += 1
            continue
        todo.append(hpo_id)

    log.info(
        "To process: %d | Already done: %d | Unknown: %d",
        len(todo), skipped, unknown,
    )

    if not todo:
        log.info("Nothing to do — all HPOs already have context files.")
        return

    # --- Build client ---
    api_key = cfg.get("api_key") or os.environ.get("LITELLM_API_KEY")
    client = OpenAI(api_key=api_key, base_url=cfg.base_url)

    system_text = _INSTRUCTION_TEMPLATE.format(n=cfg.n_sentences)
    seed = cfg.get("seed", None)

    log.info(
        "Model: %s | seed: %s | max_tokens: %d | workers: %d",
        cfg.model, seed, cfg.max_tokens, cfg.n_workers,
    )

    # --- Parallel inference loop ---
    total_cost = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    done = failed = 0
    n_total = len(todo)

    def _process_hpo(hpo_id: str) -> dict:
        """Run one API call and write the context file. Executed in a worker thread."""
        node = HPO_class(hpo_tree.data[hpo_id])
        messages = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": _build_user_content(node)},
        ]
        kwargs = dict(model=cfg.model, messages=messages, max_tokens=cfg.max_tokens)
        if seed is not None:
            kwargs["seed"] = int(seed)

        response = client.chat.completions.create(**kwargs)

        text = response.choices[0].message.content or ""
        sentences = _parse_bullet_list(text, cfg.n_sentences)

        if sentences:
            filename = hpo_id.replace(":", "_") + ".txt"
            _write_context_file(out_dir / filename, sentences)

        usage = response.usage
        return {
            "hpo_id": hpo_id,
            "ok": bool(sentences),
            "finish_reason": response.choices[0].finish_reason,
            "cost": getattr(usage, "cost", 0.0) or 0.0,
            "prompt_tokens": usage.prompt_tokens or 0,
            "completion_tokens": usage.completion_tokens or 0,
        }

    with ThreadPoolExecutor(max_workers=cfg.n_workers) as executor:
        futures = {executor.submit(_process_hpo, hpo_id): hpo_id for hpo_id in todo}
        with tqdm(total=n_total, desc="Generating context", unit="HPO") as pbar:
            for future in as_completed(futures):
                hpo_id = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    log.warning("Unhandled error for %s: %s", hpo_id, exc)
                    failed += 1
                    pbar.update(1)
                    continue

                if result["ok"]:
                    done += 1
                else:
                    log.warning(
                        "No sentences for %s (finish_reason=%s)",
                        hpo_id, result["finish_reason"],
                    )
                    failed += 1

                total_cost += result["cost"]
                total_prompt_tokens += result["prompt_tokens"]
                total_completion_tokens += result["completion_tokens"]
                pbar.update(1)

                # Print running cost every 10 completions
                if (done + failed) % 10 == 0 or (done + failed) == n_total:
                    print(
                        f"[{done + failed}/{n_total}] "
                        f"call: ${result['cost']:.6f} | "
                        f"running: ${total_cost:.6f} | "
                        f"prompt_tok: {total_prompt_tokens} | "
                        f"completion_tok: {total_completion_tokens}"
                    )

    # --- Final summary ---
    print(f"""
=== Cost Summary (LiteLLM / LLaMA 3.3 70B) ===
Model:              {cfg.model}
Seed:               {seed}
HPOs written:       {done}
HPOs failed:        {failed}
HPOs skipped:       {skipped}

Tokens:
  Prompt:           {total_prompt_tokens:>10,}
  Completion:       {total_completion_tokens:>10,}

TOTAL COST:         ${total_cost:.6f}

Output: {out_dir}
""")

    log.info(
        "Done. Written: %d | Failed: %d | Skipped: %d | Total cost: $%.6f",
        done, failed, skipped, total_cost,
    )


if __name__ == "__main__":
    main()
