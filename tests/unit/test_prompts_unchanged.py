"""Every prompt sent to a language model is byte-identical to the one the thesis runs used.

``tests/prompt_hashes.json`` holds the SHA-256 of each prompt as the thesis runs sent it: the juror
templates, the verifier and generator prompts rendered for two terms, the score store's system
prompt, and the prompt files of RAG-HPO and AutoPCR. Changing a prompt changes the experiment, so a
difference here is a failure, never a reason to update the reference file.
"""
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

#: Modules whose upper-case ``*PROMPT*`` and ``*TEMPLATE*`` strings are fingerprinted.
MODULES = {
    "prompt_library": "hpo_extraction.phenojury.prompts",
    "generation": "hpo_extraction.phenojury.generation",
    "score_store": "hpo_extraction.treephenorag.score_store",
    "verifier_prompt": "hpo_extraction.treephenorag.verifier_prompt",
    "rag_hpo_runner": "hpo_extraction.baselines.rag_hpo_runner",
    "rag_hpo_published_runner": "hpo_extraction.baselines.rag_hpo_published_runner",
    "autopcr_runner": "hpo_extraction.baselines.autopcr_runner",
    "autopcr_link": "hpo_extraction.baselines.autopcr_link",
}
#: Script files holding prompts.
SCRIPTS = {"synthetic_sentences": "experiments/04_treephenorag/synthetic_sentences/run.py"}
#: Prompt files of the copied baselines, compared as whole files.
FILES = {
    "raghpo_system_prompts": "third_party/RAG-HPO/system_prompts.json",
    "raghpo_system_prompts_paper": "third_party/RAG-HPO/system_prompts_paper.json",
    "raghpo_system_prompts_figs1": "third_party/RAG-HPO/system_prompts_figs1.json",
    "raghpo_lib": "third_party/RAG-HPO/rag_hpo_lib.py",
    "raghpo_lib_paper": "third_party/RAG-HPO/rag_hpo_lib_paper.py",
    "autopcr_prompts": "third_party/AutoPCR/utils/prompts.py",
}
#: Terms the verifier and generator prompts are rendered for: one with a definition and synonyms,
#: and one whose fields are partly empty, so both branches of the templates are covered.
TERMS = ("HP:0001250", "HP:0000118")


def sha(text: "str | bytes") -> str:
    """SHA-256 of *text* (UTF-8 for strings)."""
    data = text.encode("utf-8") if isinstance(text, str) else text
    return hashlib.sha256(data).hexdigest()


def fingerprints() -> dict:
    """``{prompt name: SHA-256}`` of every prompt in this repository."""
    from hpo_extraction.ontology.hpo_tree import HPO_class, HPOTree

    out = {}
    mods = {key: importlib.import_module(name) for key, name in MODULES.items()}
    library = mods["prompt_library"]
    for key in library.PROMPT_KEYS:
        spec = library.get_prompt(key)
        out[f"prompt:{key}:system"] = sha(spec.system)
        out[f"prompt:{key}:user_template"] = sha(spec.user_template)
    for key, module in mods.items():
        for attr in sorted(vars(module)):
            value = getattr(module, attr)
            if isinstance(value, str) and ("PROMPT" in attr or "TEMPLATE" in attr) and attr.isupper():
                out[f"{key}:{attr}"] = sha(value)
    for term in TERMS:
        out[f"verifier:{term}"] = sha("\n".join(mods["verifier_prompt"].embed_symptom_in_prompt_v1(
            ["The patient had recurrent seizures.", "No abnormality."], term)))
    tree = HPOTree()
    for key, rel in SCRIPTS.items():
        spec = importlib.util.spec_from_file_location(f"_prompts_{key}", REPO / rel)
        script = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(script)
        out[f"{key}:_INSTRUCTION_TEMPLATE"] = sha(script._INSTRUCTION_TEMPLATE.format(n=40))
        for term in TERMS:
            out[f"{key}:user:{term}"] = sha(script._build_user_content(HPO_class(tree.data[term])))
    for key, rel in FILES.items():
        # Line endings are normalised: a checkout may convert them, the prompt text does not change.
        out[f"file:{key}"] = sha((REPO / rel).read_bytes().replace(b"\r\n", b"\n"))
    return out


def test_every_prompt_matches_the_thesis_runs():
    reference = json.loads((REPO / "tests" / "prompt_hashes.json").read_text(encoding="utf-8"))
    current = fingerprints()
    assert set(current) == set(reference)
    changed = sorted(k for k in reference if current[k] != reference[k])
    assert not changed, f"prompts differ from the thesis runs: {changed}"
