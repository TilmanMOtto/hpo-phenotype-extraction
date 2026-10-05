"""
Cosine-similarity-based symptom retriever.

SymptomScoreCalculator holds pre-encoded context embeddings and exposes
retrieve() (matching RetrieverProtocol) as well as all original methods
for backwards compatibility.
"""

import numpy as np

from hpo_extraction.ontology.hpo_tree import HPOTree


class SymptomScoreCalculator:
    """
    Computes similarity scores between patient sentences and symptom context vectors.

    Attributes:
        context_dict: {hpo_code → ndarray[N_contexts × dim]} pre-encoded context embeddings.
        mode: Scoring strategy, "max" or "mean".
        top_n: Number of top context vectors to consider for "mean" mode. "all" for all.
    """

    def __init__(self, context_dict: dict[str, np.ndarray], sentScoring: str = "max", top_n="all"):
        self.context_dict = context_dict
        self.mode = sentScoring
        self.top_n = top_n

    def cos_sim(self, A: np.ndarray, B: np.ndarray) -> float:
        """Cosine similarity of two vectors, between -1 and 1."""
        return np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))

    def sent_2_context(self, concept_key: str, sent: np.ndarray) -> float:
        """Similarity of one segment to a term's synthetic sentences, pooled by the configured mode."""
        values = [self.cos_sim(context, sent) for context in self.context_dict[concept_key]]
        if self.mode == "max":
            return max(values)
        elif self.mode == "mean":
            if self.top_n == "all":
                return float(np.mean(sorted(values)))
            elif isinstance(self.top_n, int):
                return float(np.mean(sorted(values)[-self.top_n :]))
        return float(np.mean(values))

    def text_2_context(self, sent_list: list[np.ndarray], concept_key: str) -> list[float]:
        """Similarity of every segment to a term's synthetic sentences."""
        return [self.sent_2_context(concept_key, sent) for sent in sent_list]

    def symptom_score(self, concept_key: str, enc_summary_sent_dict: dict, patient_key: str) -> tuple[int, float]:
        """Index and similarity of the segment most similar to a term, for one report."""
        sent_list = enc_summary_sent_dict[patient_key]
        text_context = self.text_2_context(sent_list, concept_key)
        max_value = max(text_context)
        max_index = text_context.index(max_value)
        return max_index, max_value

    def check_patient(self, concept_keys: list[str], enc_summary_sent_dict: dict, patient_key: str) -> dict:
        """``{term: {max_index, max_value}}`` for every term of *concept_keys* on one report."""
        patient_check: dict = {}
        for concept_key in concept_keys:
            patient_check[concept_key] = {}
            patient_check[concept_key]["max_index"], patient_check[concept_key]["max_value"] = self.symptom_score(
                concept_key, enc_summary_sent_dict, patient_key
            )
        return patient_check

    def resetSentScoring(self, set_mode: str = "mean", set_top_n="all"):
        """Set how sentence similarities are pooled (``max`` or ``mean``) and over how many."""
        self.mode = set_mode
        self.top_n = set_top_n

    def add_info(self, summary_sent_dict: dict, patient_key: str, patient_check: dict) -> dict:
        """Add the best segment's text and the term label to every entry of *patient_check*."""
        hpo_tree = HPOTree()
        data = hpo_tree.data
        for symptom in patient_check:
            top_sent = summary_sent_dict[patient_key][patient_check[symptom]["max_index"]]
            patient_check[symptom]["top_sent"] = top_sent
            patient_check[symptom]["symptom"] = data[symptom]["Name"][0]
        return patient_check

    def symptom_sents(
        self,
        enc_summary_sent_dict: dict[str, np.ndarray],
        summary_sent_dict: dict[str, list[str]],
        symptom_list: list[str],
        top_n: int = 5,
    ) -> dict:
        """
        Select the top-N most relevant sentences per symptom per patient.

        Returns:
            {patient → {hpo_code → {top_sents, top_scores, symptom name, top_indices}}}
        """
        hpo_tree = HPOTree()
        data = hpo_tree.data
        filtered_sents: dict = {}
        for key in enc_summary_sent_dict:
            filtered_sents[key] = {}
            for symptom in symptom_list:
                filtered_sents[key][symptom] = {}
                score_list = self.text_2_context(enc_summary_sent_dict[key], symptom)
                sorted_indices = np.argsort(np.array(score_list))[::-1]
                top_sents: list[str] = []
                top_scores: list[float] = []
                top_indices: list[int] = []
                for index in sorted_indices:
                    if len(top_sents) >= top_n:
                        break
                    candidate = summary_sent_dict[key][index]
                    if candidate not in top_sents:
                        top_sents.append(candidate)
                        top_scores.append(score_list[index])
                        top_indices.append(int(index))
                filtered_sents[key][symptom]["top_sents"] = top_sents
                filtered_sents[key][symptom]["top_scores"] = top_scores
                filtered_sents[key][symptom]["symptom name"] = data[symptom]["Name"][0]
                filtered_sents[key][symptom]["top_indices"] = top_indices
        return filtered_sents

    def retrieve(
        self,
        enc_patient_sents: dict[str, np.ndarray],
        patient_sents: dict[str, list[str]],
        symptom_list: list[str],
        top_n: int = 5,
    ) -> dict:
        """RetrieverProtocol-compatible alias for symptom_sents."""
        return self.symptom_sents(enc_patient_sents, patient_sents, symptom_list, top_n)
