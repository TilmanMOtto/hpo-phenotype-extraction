# apps

Four browser apps built with Dash. They read result folders and HCY data, so on real data they run
on LeoMed and you open them through an SSH tunnel (`docs/cluster.md`). Each one also runs a self-test
on synthetic data, which needs no data and no cluster.

| App | What it is for | Self-test |
|---|---|---|
| [`curation_ui/`](curation_ui/README.md) | Review and repair of the HCY annotations, report by report. Its event log is what the curated ground truth is built from. | `python apps/curation_ui/app.py --selftest` |
| [`compare_ui/`](compare_ui/README.md) | The five methods of chapter 6 side by side on one document, for the 20 HCY reports and 20 GSC+ abstracts of the qualitative appendix. | `python apps/compare_ui/app.py --selftest` |
| [`treephenorag_ui/`](treephenorag_ui/README.md) | Inspection of TreePhenoRAG runs that wrote one result folder per threshold: traversal, pruning, errors, calibration. | `python apps/treephenorag_ui/app.py --selfcheck` (needs runs) |
| [`phenojury_ui/`](phenojury_ui/README.md) | Inspection of the Free Listing generation run of PhenoJury: per-juror outputs, votes, misses, and manual annotation of juror replies. | `python apps/phenojury_ui/app.py --selftest` (needs runs) |

`ui_common/` holds the theme and loaders the apps share.

Every app reads its default locations from the path file (`configs/cluster_leomed.yaml`, or the file
the environment variable `HPO_PATHS` names), and every location can be overridden on the command
line (`--help`). Run the apps from the repository root with the package installed (`pip install -e .`).

## Starting an app on LeoMed

```bash
# terminal 1, on the LeoMed login node
cd <your checkout of this repository>
conda activate <environment>                 # docs/cluster.md
python apps/compare_ui/app.py --port 8057

# terminal 2, on your own computer
ssh -L 8057:localhost:8057 <LeoMed login node> # then open http://localhost:8057
```

The apps bind to `127.0.0.1` by default, so they are reachable only through the tunnel. HCY report
text is shown in the browser but never leaves the cluster in any file the apps write.
