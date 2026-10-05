"""Prompt construction from retrieved sentences and HPO metadata."""

from hpo_extraction.ontology.hpo_tree import HPOTree


def embed_symptom_in_prompt_v1(sents: list[str], hpo_code: str) -> list[str]:
    """
    Build one LLM prompt per sentence embedding symptom name, definition, and synonyms.

    Args:
        sents: List of text segments to evaluate.
        hpo_code: HPO code for the target symptom.

    Returns:
        List of prompt strings (same length as sents).
    """
    hpo_tree = HPOTree()
    data = hpo_tree.data
    hpo_label = data[hpo_code]["Name"][0]
    definition = data[hpo_code]["Def"][0] if data[hpo_code]["Def"] else ""
    synonyms = ", ".join(data[hpo_code]["Synonym"])

    prompts = []
    for sent in sents:
        intro = f"The symptom {hpo_label}"
        if definition:
            intro += f" is defined as {definition}."
        else:
            intro += "."
        if synonyms:
            intro += f" {hpo_label} is also referred to as {synonyms}."

        prompt = (
            intro
            + " Does the following text segment explicitly confirm that the patient"
            + f" has this symptom:'{sent}'. "
        )
        prompts.append(prompt)
    return prompts




class BaselinePrompter:
    """Wraps embed_symptom_in_prompt_v1 as a PrompterProtocol-compatible class."""

    def build_prompts(self, sents: list[str], hpo_code: str) -> list[str]:
        """One verifier prompt per segment for term *hpo_code* (:func:`embed_symptom_in_prompt_v1`)."""
        return embed_symptom_in_prompt_v1(sents, hpo_code)
