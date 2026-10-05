"""The PhenoBERT launcher must be invisible except for the recursion limit.

``annotate.py`` is a vendored third-party script run as a subprocess. Wrapping it is only
acceptable if the wrapper reproduces direct execution, same ``__name__``, same ``argv``,
same import resolution. The one deliberate difference is a raised recursion limit, so stanza's
recursive Chu-Liu/Edmonds cycle detection survives a single very long input line (the failure that
killed the ``medgemma`` cells of an earlier exploratory run).

The ``sys.path`` case below is not hypothetical: ``runpy.run_path`` does *not* prepend the script's
directory the way ``python script.py`` does, so the first version of the launcher broke
annotate.py's bare ``from util import …`` imports outright.
"""
from __future__ import annotations

import pathlib
import subprocess
import sys
import textwrap

import pytest

LAUNCHER = pathlib.Path(__file__).resolve().parents[2] / "src" / "hpo_extraction" / "phenojury" / "phenobert_launcher.py"

TARGET = textwrap.dedent(
    """
    import sys
    from helper import VALUE          # bare import, exactly as annotate.py does

    def rec(n):
        return 0 if n == 0 else 1 + rec(n - 1)

    print("NAME", __name__)
    print("ARGV", " ".join(sys.argv))
    print("HELPER", VALUE)
    print("LIMIT", sys.getrecursionlimit())
    print("DEEP", rec(5000))
    """
)


@pytest.fixture
def phenobert_like(tmp_path):
    """A directory shaped like PhenoBERT's utils/: a script plus a bare-importable sibling."""
    (tmp_path / "helper.py").write_text("VALUE = 'bare-import-works'\n")
    (tmp_path / "annotate.py").write_text(TARGET)
    return tmp_path


def _run(cmd, cwd):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


class TestLauncherReproducesDirectExecution:
    def test_direct_execution_of_the_target_hits_the_recursion_wall(self, phenobert_like):
        """Establishes the failure the launcher exists to prevent."""
        r = _run([sys.executable, "annotate.py", "-i", "in"], phenobert_like)
        assert r.returncode != 0
        assert "RecursionError" in r.stderr

    def test_launcher_runs_the_target_as_main(self, phenobert_like):
        r = _run([sys.executable, str(LAUNCHER), "annotate.py", "-i", "in"], phenobert_like)
        assert r.returncode == 0, r.stderr
        assert "NAME __main__" in r.stdout

    def test_launcher_passes_argv_through_as_if_invoked_directly(self, phenobert_like):
        r = _run(
            [sys.executable, str(LAUNCHER), "annotate.py", "-i", "in", "-o", "out", "-t", "4"],
            phenobert_like,
        )
        assert r.returncode == 0, r.stderr
        assert "ARGV annotate.py -i in -o out -t 4" in r.stdout

    def test_launcher_keeps_bare_sibling_imports_resolvable(self, phenobert_like):
        """runpy.run_path does not do this for us, annotate.py breaks without it."""
        r = _run([sys.executable, str(LAUNCHER), "annotate.py", "-i", "in"], phenobert_like)
        assert r.returncode == 0, r.stderr
        assert "HELPER bare-import-works" in r.stdout

    def test_launcher_lifts_the_recursion_limit(self, phenobert_like):
        r = _run([sys.executable, str(LAUNCHER), "annotate.py", "-i", "in"], phenobert_like)
        assert r.returncode == 0, r.stderr
        assert "LIMIT 20000" in r.stdout
        assert "DEEP 5000" in r.stdout

    def test_recursion_limit_is_overridable_from_the_environment(self, phenobert_like, monkeypatch):
        import os

        env = dict(os.environ, PHENOBERT_RECURSION_LIMIT="31337")
        r = subprocess.run(
            [sys.executable, str(LAUNCHER), "annotate.py"],
            cwd=phenobert_like, capture_output=True, text=True, env=env,
        )
        assert r.returncode == 0, r.stderr
        assert "LIMIT 31337" in r.stdout

    def test_launcher_without_a_script_fails_loudly(self, phenobert_like):
        r = _run([sys.executable, str(LAUNCHER)], phenobert_like)
        assert r.returncode != 0
        assert "usage" in r.stderr.lower()


class TestRunnerUsesTheLauncher:
    def test_phenobert_runner_invokes_annotate_through_the_launcher(self):
        """A direct ``python annotate.py`` in the runner would silently drop the fix."""
        runner = (LAUNCHER.parent / "phenobert.py").read_text()
        assert "phenobert_launcher.py" in runner
        assert 'python_exe, launcher, "annotate.py"' in runner
