"""Print one value of the path file: ``python -m hpo_extraction.path_lookup hcy.input_dir``.

The SLURM templates read their locations this way, so no template holds an absolute path.
"""
import argparse

from hpo_extraction.paths import lookup


def main() -> int:
    """Print the value of the key named on the command line. Exit status 0."""
    ap = argparse.ArgumentParser(description="Print one value of the path file (HPO_PATHS, "
                                             "default configs/cluster_leomed.yaml).")
    ap.add_argument("key", help="dotted key, for example hcy.input_dir or models.sapbert")
    print(lookup(ap.parse_args().key))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
