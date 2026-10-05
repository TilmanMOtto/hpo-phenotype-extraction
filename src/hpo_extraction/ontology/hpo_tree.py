"""
HPO ontology graph structure (HPOTree) and node class (HPO_class).

Re-used from previous work of Feng et al., 2023
@article{Feng2023,
  title = {PhenoBERT: A Combined Deep Learning Method for Automated Recognition of Human Phenotype Ontology},
  volume = {20},
  ISSN = {2374-0043},
  url = {http://dx.doi.org/10.1109/TCBB.2022.3170301},
  DOI = {10.1109/tcbb.2022.3170301},
  number = {2},
  journal = {IEEE/ACM Transactions on Computational Biology and Bioinformatics},
  publisher = {Institute of Electrical and Electronics Engineers (IEEE)},
  author = {Feng,  Yuhao and Qi,  Lei and Tian,  Weidong},
  year = {2023},
  month = mar,
  pages = {1269-1277}
}
"""

import functools
import hashlib
import json
from collections import deque
from pathlib import Path

from hpo_extraction.ontology.hpo_items import WordItem, processStr

from hpo_extraction.paths import REPO_ROOT as _REPO_ROOT
DEFAULT_HPO_JSON = _REPO_ROOT / "resources" / "util" / "hpo.json"


@functools.lru_cache(maxsize=4)
def ontology_fingerprint(hpo_json_path: str | Path | None = DEFAULT_HPO_JSON) -> str:
    """First 12 hex of the SHA-256 of the ontology file, the release identifier we can honestly give.

    ``resources/util/hpo.json`` carries no ``data-version``, no provenance and no download script:
    it arrived with the original PhenoBERT import and the upstream HPO release it was built from is
    unknown (see ``resources/util/PROVENANCE.md``). Every recall ceiling this project quotes, 0.851
    on GSC+, 0.887 and 0.948 on the two RAG-HPO cohorts, is a property of *this file*, so what an
    output record needs to fix is the file, not a release string we would have to invent.

    A content hash does that: it is stable, it is checkable offline, and it changes the
    moment the graph does. Records stamp it as ``ontology_version`` so a number can always be traced
    back to the graph that produced it.

    Cached because it is stamped onto every emitted record and the file is 15 MB.
    """
    # A blank Hydra key resolves to None, not to a missing argument, so the default has to be
    # re-applied here rather than relied on in the signature.
    digest = hashlib.sha256(Path(hpo_json_path or DEFAULT_HPO_JSON).read_bytes()).hexdigest()
    return digest[:12]


class HPO_class:
    """Represents a single HPO node."""

    def __init__(self, dic: dict):
        self.id = dic["Id"]
        self.name = dic["Name"]
        self.alt_id = dic["Alt_id"]
        self.definition = dic["Def"]
        self.comment = dic["Comment"]
        self.synonym = dic["Synonym"]
        self.xref = dic["Xref"]
        self.is_a = dic["Is_a"]
        self.father = set(dic["Father"].keys())
        self.child = set(dic["Child"].keys())
        self.son = set(dic["Son"].keys())


def getNames(struct: HPO_class) -> list[str]:
    """Return deduplicated list of name + synonyms for an HPO node."""
    names = list(struct.name)
    names.extend(struct.synonym)
    return list(set(names))


def hpo_label(tree: "HPOTree", code: str) -> str:
    """The node's primary name, or the code itself when this release doesn't carry it.

    Labels are cosmetic, they exist so a predictions file can be read without a second lookup, so an unknown code degrades to the code, not raising.
    """
    entry = tree.data.get(code, {})
    names = entry.get("Name")
    if isinstance(names, list) and names:
        return names[0]
    return names if isinstance(names, str) and names else code


def resolve_hpo(tree: "HPOTree", code: str) -> str | None:
    """*code* mapped onto this ontology release's primary id, or ``None`` if it is not in it.

    External annotators ship their own HPO release: PhenoBERT's is older than
    ``resources/util/hpo.json``, so some of what it emits is an alt (merged) id and some is a term
    this release no longer has at all. Passing an alt id straight through would score a correct
    detection as a miss, every ancestor lookup on it comes back empty, so it is resolved here,
    and a code that resolves to nothing is returned as ``None`` for the caller to *count*, never to
    drop silently.
    """
    if code in tree.data:
        return code
    return tree.alt_id_dict.get(code)


class HPOTree:
    """
    Directed acyclic graph structure for the HPO ontology.
    Root node defaults to HP:0000118 (Phenotypic abnormality).
    """

    def __init__(self, hpo_json_path: str | Path = DEFAULT_HPO_JSON):
        with open(hpo_json_path, encoding="utf-8") as f:
            self.data = json.loads(f.read())

        self.root = "HP:0000118"
        self.phenotypic_abnormality = HPO_class(self.data[self.root]).child
        self.phenotypic_abnormalityNT = set(list(self.phenotypic_abnormality))
        self.phenotypic_abnormality.add(self.root)

        self.hpo_list = sorted(list(self.phenotypic_abnormality))
        self.n_concept = len(self.hpo_list)
        self.hpo2idx = {hpo: idx for idx, hpo in enumerate(self.hpo_list)}
        self.hpo2idx["None"] = len(self.hpo_list)
        self.idx2hpo = {self.hpo2idx[hpo]: hpo for hpo in self.hpo2idx}

        self.alt_id_dict: dict[str, str] = {}
        self.p_phrase2HPO: dict[str, str] = {}
        self.depth = 0

        self.layer1 = sorted(list(HPO_class(self.data[self.root]).son))
        self.layer1_set = set(self.layer1)
        self.n_concept_l1 = len(self.layer1)
        self.hpo2idx_l1 = {hpo: idx for idx, hpo in enumerate(self.layer1)}
        self.hpo2idx_l1["None"] = len(self.layer1)
        self.idx2hpo_l1 = {self.hpo2idx_l1[hpo]: hpo for hpo in self.hpo2idx_l1}

        for hpo_name in self.data:
            struct = HPO_class(self.data[hpo_name])
            for sub_alt_id in struct.alt_id:
                self.alt_id_dict[sub_alt_id] = hpo_name
            for phrase in getNames(struct):
                key = " ".join(sorted(processStr(phrase)))
                self.p_phrase2HPO[key] = hpo_name

    def buildHPOTree(self):
        """Build depth hash via BFS."""
        self.depth_dict: dict[str, int] = {}
        queue = {self.root}
        visited = {self.root}
        depth = 0
        while queue:
            tmp: set[str] = set()
            for node in queue:
                self.depth_dict[node] = depth
                for sub_node in HPO_class(self.data[node]).son:
                    if sub_node not in visited:
                        visited.add(sub_node)
                        tmp.add(sub_node)
            queue = tmp
            depth += 1
        self.depth = depth - 1
        self._buildLongestDepth()

    def _buildLongestDepth(self):
        """Longest-path depth from the root, a depth that is monotone along ancestry.

        ``depth_dict`` is the *shortest* path from the root, and HPO is a DAG, not a tree.
        A node reachable by a short branch can therefore have an ancestor, reachable only by a
        long branch, whose ``depth_dict`` value is *larger* than its own (this holds for 1.4% of
        ancestor/descendant pairs). Wu-Palmer divides by those depths, so such a pair drove
        ``getNodeSimilarityByID`` above 1.0, up to 1.6 in practice.

        Longest-path depth cannot do that: every root→ancestor path extends to the descendant, so
        an ancestor is always strictly shallower. Used only for node similarity; ``depth_dict``
        keeps its BFS meaning everywhere else (depth-stratified recall, error classification).
        """
        nodes = self.phenotypic_abnormality
        parents = {n: [p for p in HPO_class(self.data[n]).is_a if p in nodes] for n in nodes}
        children: dict[str, list[str]] = {n: [] for n in nodes}
        in_degree = {n: len(parents[n]) for n in nodes}
        for node, ps in parents.items():
            for p in ps:
                children[p].append(node)

        self.depth_long: dict[str, int] = {n: 0 for n in nodes}
        queue = deque(n for n in nodes if in_degree[n] == 0)
        while queue:
            node = queue.popleft()
            for child in children[node]:
                self.depth_long[child] = max(self.depth_long[child], self.depth_long[node] + 1)
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

    def getNameByHPO(self, hpo_num: str) -> str:
        """Lower-case label of term *hpo_num*."""
        return HPO_class(self.data[hpo_num]).name[0].lower()

    def getFatherHPOByHPO(self, hpo_num: str):
        """Direct parents of *hpo_num*, or None outside *Phenotypic abnormality*."""
        if hpo_num not in self.phenotypic_abnormalityNT:
            return None
        return HPO_class(self.data[hpo_num]).is_a

    def getLayer1HPOByHPO(self, hpo_num: str) -> list[str]:
        """Organ-system terms (children of HP:0000118) above *hpo_num*, or ``['None']`` outside that subtree."""
        if hpo_num not in self.phenotypic_abnormalityNT:
            return ["None"]
        if hpo_num in self.layer1_set:
            return [hpo_num]
        return list(self.layer1_set & HPO_class(self.data[hpo_num]).father)

    def getAllFatherHPOByHPO(self, hpo_num: str) -> set[str]:
        """All ancestors of *hpo_num* within *Phenotypic abnormality*."""
        if hpo_num not in self.phenotypic_abnormalityNT:
            return set()
        return HPO_class(self.data[hpo_num]).father

    def getPhrasesByHPO(self, hpo_num: str) -> list[str]:
        """Lower-case label and synonyms of *hpo_num*."""
        return [i.lower() for i in getNames(HPO_class(self.data[hpo_num]))]

    def getAllPhrasesAbnorm(self) -> list[str]:
        """Labels and synonyms of every term in ``hpo_list``."""
        phrases_list = []
        for hpo_name in self.hpo_list:
            phrases_list.extend(getNames(HPO_class(self.data[hpo_name])))
        return phrases_list

    def matchPhrase2HPO(self, phrase: str) -> str:
        """The term whose label or synonym equals *phrase* after normalisation (word order and lemmas ignored), or ``'None'``."""
        p_phrase = " ".join(sorted(processStr(phrase)))
        p_l_phrase = " ".join(
            [WordItem.lemma_dict[i] if i in WordItem.lemma_dict else i for i in p_phrase.split()]
        )
        if p_phrase in self.p_phrase2HPO:
            return self.p_phrase2HPO[p_phrase]
        elif p_l_phrase in self.p_phrase2HPO:
            return self.p_phrase2HPO[p_l_phrase]
        return ""

    def getHPO2idx(self, hpo_num: str) -> int:
        """Index of *hpo_num* in the term list."""
        return self.hpo2idx[hpo_num]

    def getHPO2idx_l1(self, hpo_num: str) -> int:
        """Index of organ-system term *hpo_num*."""
        return self.hpo2idx_l1[hpo_num]

    def getIdx2HPO(self, idx: int) -> str:
        """Term at index *idx* of the term list."""
        return self.idx2hpo[idx]

    def getIdx2HPO_l1(self, idx: int) -> str:
        """Organ-system term at index *idx*."""
        return self.idx2hpo_l1[idx]

    def getMaterial4L1(self, root_l1: str):
        """Index, term list, size and index maps of the subtree below organ-system term *root_l1*."""
        root_idx = self.getHPO2idx_l1(root_l1)
        hpo_list = HPO_class(self.data[root_l1]).child
        hpo_list.add(root_l1)
        hpo_list = sorted(hpo_list)
        n_concept = len(hpo_list)
        hpo2idx = {hpo: idx for idx, hpo in enumerate(hpo_list)}
        hpo2idx["None"] = len(hpo_list)
        idx2hpo = {hpo2idx[hpo]: hpo for hpo in hpo2idx}
        return root_idx, hpo_list, n_concept, hpo2idx, idx2hpo

    def getNodeSimilarityByID(self, hpoNum1: str, hpoNum2: str) -> float:
        """Wu-Palmer node similarity in [0, 1].

        Uses ``depth_long`` (longest path from the root), not ``depth_dict`` (shortest).
        On a DAG only the longest-path depth is monotone along ancestry, and Wu-Palmer needs that
        to stay bounded: with the shortest-path depth the lowest common subsumer can be *deeper*
        than the nodes it subsumes, which pushed this ratio above 1. See ``_buildLongestDepth``.
        """
        if hpoNum1 not in self.phenotypic_abnormality or hpoNum2 not in self.phenotypic_abnormality:
            return 0.0
        if hpoNum1 == self.root and hpoNum2 == self.root:
            return 1.0
        depth1 = self.depth_long[hpoNum1]
        depth2 = self.depth_long[hpoNum2]
        if depth1 + depth2 == 0:
            return 1.0 if hpoNum1 == hpoNum2 else 0.0
        struct1 = HPO_class(self.data[hpoNum1])
        struct2 = HPO_class(self.data[hpoNum2])
        father1 = struct1.father | {hpoNum1}
        father2 = struct2.father | {hpoNum2}
        ancestor = father1 & father2
        LCS = sorted(
            [[a, self.depth_long[a]] for a in ancestor if a in self.phenotypic_abnormality],
            key=lambda x: x[1],
            reverse=True,
        )[0][0]
        depth3 = self.depth_long[LCS]
        return 2 * depth3 / (depth1 + depth2)

    def getHPO_set_similarity_max(self, hpo_set1: set, hpo_set2: set) -> float:
        """Symmetric set similarity of two term sets from best-match node similarities, between 0 and 1."""
        if not hpo_set1 and not hpo_set2:
            return 1.0
        if not hpo_set1 or not hpo_set2:
            return 0.0
        part1 = sum(
            1 - max((self.getNodeSimilarityByID(h1, h2) for h2 in hpo_set2), default=0)
            for h1 in hpo_set1 if h1 not in hpo_set2
        )
        part2 = sum(
            1 - max((self.getNodeSimilarityByID(h1, h2) for h1 in hpo_set1), default=0)
            for h2 in hpo_set2 if h2 not in hpo_set1
        )
        return 1 - (part1 + part2) / len(hpo_set1 | hpo_set2)

    def getAdjacentMatrixAncestors(self, root_l1: str, num_nodes: int):
        """Sparse ancestor-weight matrix of the subtree below *root_l1*, *num_nodes* square."""
        import scipy.sparse as ss

        root_idx, hpo_list, n_concept, hpo2idx, idx2hpo = self.getMaterial4L1(root_l1)
        ancestors_weight: dict = {}
        for hpo_num in hpo_list:
            concept_id = hpo2idx[hpo_num]
            self.getAdjacentMatrixAncestorsAssist(ancestors_weight, concept_id, hpo2idx, idx2hpo)
        sparse_indexes = []
        sparse_values = []
        for concept_id in ancestors_weight:
            sparse_indexes.extend([[concept_id, a] for a in ancestors_weight[concept_id]])
            sparse_values.extend(ancestors_weight[concept_id][a] for a in ancestors_weight[concept_id])

        import numpy as np

        indices = np.array(sparse_indexes)
        A = ss.coo_matrix(
            (np.array(sparse_values), (indices[:, 0], indices[:, 1])),
            shape=(num_nodes, num_nodes),
            dtype=float,
        )
        return A.tocoo()

    def getAdjacentMatrixAncestorsAssist(self, ancestors_weight: dict, concept_id: int, hpo2idx: dict, idx2hpo: dict):
        """Fill *ancestors_weight* for *concept_id* and its ancestors (recursive helper), and return its ancestors."""
        if concept_id in ancestors_weight:
            return ancestors_weight[concept_id].keys()
        ancestors_weight[concept_id] = {concept_id: 1.0}
        fathers = [i for i in HPO_class(self.data[idx2hpo[concept_id]]).is_a if i in hpo2idx]
        for father_hpo_num in fathers:
            father_id = hpo2idx[father_hpo_num]
            ancestors = self.getAdjacentMatrixAncestorsAssist(ancestors_weight, father_id, hpo2idx, idx2hpo)
            for ancestor_id in ancestors:
                if ancestor_id not in ancestors_weight[concept_id]:
                    ancestors_weight[concept_id][ancestor_id] = 0.0
                ancestors_weight[concept_id][ancestor_id] += ancestors_weight[father_id][ancestor_id] / len(fathers)
        return ancestors_weight[concept_id].keys()
