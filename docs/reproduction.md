# Reproduction

How far the thesis results were reproduced with this code, and how to repeat the checks.
Abbreviations: HCY (the clinical cohort, LeoMed only), GSC+ (the public corpus).

## What was rerun (LeoMed, 2026-10-04 and 2026-10-05)

Every analysis behind the thesis tables was rerun from the stored model outputs, in the
environment of `environment_leomed.yml`, and its tables were compared with the stored ones that
the thesis was built from.

| Run | Result |
|---|---|
| dataset statistics (`slurm/dataset_statistics.sbatch`) | all HCY rows equal. The GSC+ rows, rerun on a laptop, are equal as well. |
| TreePhenoRAG protocol (`slurm/treephenorag_protocol.sbatch`) | 21 of 37 tables byte-identical. 13 differ by floating-point noise below 1e-15, one only in row order, one in threshold keys at the 17th digit. In one table a count of distinct scores differs by 1 (442,746 against 442,745). The thesis does not print it. |
| PhenoJury protocol, HCY and GSC+ (206) (`slurm/phenojury_protocol.sbatch`) | all 83 tables byte-identical |
| PhenoJury rows of the comparison (`slurm/comparison_inputs.sbatch`) | all prediction files byte-identical |
| comparison (`slurm/comparison.sbatch`) | every number equal (largest difference 1e-16) |
| retrieval curves (`slurm/retrieval_curves.sbatch`) | the shares of evidence retrieved, which Figure 4.5 plots, are equal. Encoder cosines differ by up to 7e-6, which moves the mean number of segments forwarded under score-threshold gates by up to 0.009. |
| synthetic-sentence statistics (`experiments/04_treephenorag/index_statistics.py`) | byte-identical |
| labels and synonyms (`experiments/05_phenojury/count_surface_strings.py`) | 40,335 strings over 18,354 terms, as in section 5.3 |
| every generated table, figure and inline number (`experiments/figures/make_all_ch{3,4,5,6}.py`) | all 54 equal to the thesis versions. 20 differ only in LaTeX comments that name the source folder. |
| PhenoBERT baseline on five GSC+ abstracts (laptop) | predicted terms equal to the stored run |

Not rerun: the GPU runs that produced the stored model outputs (stored verifier scores, juror
generations, baselines). Their commands are in `docs/thesis_map.md`.

The comparison's `t1_literature.csv` reruns with a different text in two columns that describe how
the RAG-HPO paper averages its scores. The stored table predates the final config. The tables built
from it are identical either way.

## The applications

On synthetic text (`examples/synthetic_reports/`):

| Check | Result |
|---|---|
| both methods with stand-in models (`examples/stand_in_demo.py`, `tests/unit/test_applications.py`) | run end to end on a CPU. The stand-ins replace the language models only. |
| both methods with the real models, one V100 GPU on LeoMed | TreePhenoRAG, 81 minutes, 8,518 verifier calls: Seizure, Diminished movement, Abnormal muscle tone and two hypotonia terms in report 1, Hepatomegaly and Feeding difficulties in report 2. PhenoJury, 37 minutes: Seizure, Generalized hypotonia, Delayed gross motor development, and Hyperhomocystinemia, Hepatomegaly, Feeding difficulties. |
| the online pooling of the application against the pooling behind the thesis numbers | equal to 1e-12 (`tests/unit/test_applications.py`) |

## Repeating the checks

On LeoMed, with a copy of `configs/cluster_leomed.yaml` whose `output_dir` and `thesis_dir` point to
new folders (so that no stored result is overwritten):

```bash
export HPO_PATHS=<your copy>
sbatch slurm/treephenorag_protocol.sbatch      # and the other templates in the table above
python experiments/figures/make_all.py         # needs the thesis LaTeX source in thesis_dir
```

Then compare each new table with the stored one of the same name under `results_dir`.

Locally:

```bash
python -m pytest tests -q                      # unit tests on synthetic data, CPU
python examples/stand_in_demo.py               # both methods with stand-in models
```

## Privacy checks

```bash
python tools/scan_repository.py --words        # report ids, absolute paths, user names
python tools/scan_report_text.py               # on LeoMed: runs of eight words shared with HCY reports
python tools/scan_report_text.py --n 6         # and of six (ontology files and GSC+ abstracts match)
```

On 2026-10-05 the eight-word scan found one line, a stock phrase in a synthetic sentence ("should
also be considered in the differential diagnosis"). Runs of six words are shared by ontology
files, GSC+ abstracts, 1,106 lines of synthetic sentences (stock clinical phrasing: they were
generated from the ontology alone, with no report as input), that phrase and one line of
plotting code.
