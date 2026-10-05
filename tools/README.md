# tools

Maintenance scripts. None of them is needed to run the methods or the thesis scripts.

| Script | Purpose |
|---|---|
| `link_stored_results.py` | Adds links with the current folder names beside stored result folders that carry older names (`docs/cluster.md`). Lists what it would do unless `--apply` is given. Nothing is moved or deleted. |
| `scan_repository.py` | Looks through every tracked file for report identifiers, absolute cluster or home paths, user names and contact addresses, and (with `--words`) for the words and punctuation the documentation avoids. Exit status 1 when one of the first three finds anything. |
| `scan_report_text.py` | Looks for runs of eight (or `--n`) words that a tracked file shares with the HCY reports, without printing them. Runs on LeoMed, where the reports are. |

Run both scans before anything is published (`docs/reproduction.md`).
