"""The ground-truth build: the curated HCY ground truth, and the report subset it is trustworthy over.

Three things are being guarded, and they fail in opposite directions:

**The evidence rule.** A term is in the ground truth when it sits on a segment and a trigger word that
actually occurs in that segment. Getting it wrong in one direction ships a ground truth whose terms nobody
can point at in the report, which is the code-only ground truth this whole pass exists to replace. Getting it wrong in the other silently deletes annotations that *were* placed, because the file that
recorded them badly was the one adjudication happened to pick.

**The cohort filter.** Admitting a report no annotator ever touched would score every method against
an empty ground truth and hand it a free precision penalty. Excluding one that *was* annotated throws away
the evidence the experiment exists to gather. Both are silent, the numbers still come out, so
every admission signal has its own test.

**The ground truth policy.** The user-facing rules are: prior_annotation, daphne and suggestions all count, unruled
ones included. Unapproved suggestions count. Anything a curator ruled against, proposed deleting, or
labelled ``unsure``/``family`` does not. Each is one branch of ``_decide``, and each is the kind of
rule that is easy to invert by accident and impossible to notice afterwards from a P/R table.

The inputs are the curation app's own fixture (``apps.curation_ui.fixture``), so the files these
tests read are byte-identical in shape to the ones on the cluster, including the ``"True."`` with a
trailing full stop in the confirmed flag, the prior_annotation row whose trigger word occurs nowhere in the
report, and the patient with no segmentation at all.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def curated_gold(monkeypatch):
    """``experiments/03_setup/ground_truth/curated_gold.py`` as a top-level module.

    Imported the way ``run.py`` sets it up, the experiment directory on ``sys.path``, so a test
    exercises the same import graph the cluster does.
    """
    exp_dir = _REPO / "experiments" / "03_setup" / "ground_truth"
    monkeypatch.syspath_prepend(str(exp_dir))
    monkeypatch.syspath_prepend(str(_REPO))
    sys.modules.pop("curated_gold", None)
    module = importlib.import_module("curated_ground_truth")
    yield module
    sys.modules.pop("curated_gold", None)


@pytest.fixture
def hcy_dir(tmp_path):
    """The curation app's fixture cohort on disk, plus an empty curation directory."""
    from apps.curation_ui import fixture

    paths = fixture.write(str(tmp_path), with_phenobert=False)
    curation = Path(paths["hcy_dir"]) / "curation"
    curation.mkdir(parents=True, exist_ok=True)
    return Path(paths["hcy_dir"])


@pytest.fixture
def log(hcy_dir):
    """An :class:`store.EventLog` on the fixture's curation directory."""
    from hpo_extraction.curation import store

    return store.EventLog(str(hcy_dir / "curation"), author="test")


def _load(curated_gold, hcy_dir):
    paths = curated_gold.resolve_paths(str(hcy_dir))
    return (curated_gold.load_annotations(paths),
            curated_gold.load_corpus(paths),
            curated_gold.load_curation(paths["curation_dir"]))


def _build(curated_gold, hcy_dir, policy=None, criteria=None):
    annotations, corpus, fold = _load(curated_gold, hcy_dir)
    return curated_gold.build(
        annotations, fold,
        policy or curated_gold.GoldPolicy(),
        criteria or curated_gold.DEFAULT_CRITERIA,
        corpus=corpus,
    )


def _reason(result, patient_id, hpo_code):
    return next(t.reason for t in result.terms
                if t.patient_id == patient_id and t.hpo_code == hpo_code)


# ── the evidence rule ────────────────────────────────────────────────────────

def test_an_annotation_whose_trigger_is_not_in_the_report_is_out(curated_gold, hcy_dir):
    """The main rule, on the fixture's own case.

    SYN001's HP:0001873 is annotated on the word *purpura*, which the report does not contain, what
    a report re-exported after annotation looks like. Nothing can point at it, so nothing downstream
    can tell a model that missed it from a model that was never given the chance.
    """
    result = _build(curated_gold, hcy_dir)
    assert "HP:0001873" not in result.gold["SYN001"]
    assert _reason(result, "SYN001", "HP:0001873") == "no_evidence"


def test_an_annotation_with_no_trigger_word_at_all_is_out(curated_gold, hcy_dir):
    """The state that reads as "fine" to anything asking only whether a stated trigger was found.

    SYN002's HP:0002187 names a code and nothing else. It never failed to be placed. There was
    never anything to place, which is the same absence of evidence by a different route.
    """
    result = _build(curated_gold, hcy_dir)
    assert "HP:0002187" not in result.gold["SYN002"]
    assert _reason(result, "SYN002", "HP:0002187") == "no_evidence"


def test_a_report_with_no_segmentation_yields_no_gold(curated_gold, hcy_dir):
    """SYN004 has annotations and no segmented report, so nothing on it can be located.

    It stays *in* the cohort, its annotations are a real signal, and comes out with an empty ground truth
    set, which is a loud state rather than a quiet one: ``run.py`` warns by name, because a report
    scored against an empty ground truth turns every prediction on it into a false positive and the cause
    (a segmentation that does not cover the cohort) is invisible from a P/R table.
    """
    result = _build(curated_gold, hcy_dir)
    assert result.gold["SYN004"] == set()
    row = next(r for r in result.reports if r.patient_id == "SYN004")
    assert row.in_cohort and row.n_gold_terms == 0 and row.n_unanchored


def test_a_curators_location_puts_an_annotation_back(curated_gold, hcy_dir, log):
    """The round trip the Evidence location tab exists for.

    ``anchors.place_all`` reads the annotation files and never sees the curation log, so without
    folding the log back in a term somebody placed by hand would stay out of the ground truth for ever, the queue would be uncloseable, which is the failure ``reader.needs_anchor`` fixes on screen and
    this fixes in the export.
    """
    from hpo_extraction.curation import store

    log.append("anchor", "SYN001",
               target_key=store.target_key("prior_annotation", "SYN001", "HP:0001873"),
               hpo_code="HP:0001873", segment_idx=3, trigger_word="thrombocytopenia")
    result = _build(curated_gold, hcy_dir)
    assert "HP:0001873" in result.gold["SYN001"]
    row = next(t for t in result.terms if t.hpo_code == "HP:0001873")
    assert (row.anchored, row.how, row.segment_idx) == (True, "curated", 3)


def test_a_curators_location_outranks_the_files_own_claim(curated_gold, hcy_dir, log):
    """``how`` says where a placement came from, and a person's beats a file's.

    The distinction is not decoration: ``lexical`` means the tool scanned the report and found the
    words somewhere, which is a guess, and a reader is entitled to know which they are looking at.
    """
    from hpo_extraction.curation import store

    plain = _build(curated_gold, hcy_dir)
    assert next(t for t in plain.terms
                if t.patient_id == "SYN001" and t.hpo_code == "HP:0001250").how == "segment"

    log.append("anchor", "SYN001", target_key=store.target_key("daphne", "SYN001", "HP:0001250"),
               hpo_code="HP:0001250", segment_idx=1, trigger_word="seizures")
    result = _build(curated_gold, hcy_dir)
    assert next(t for t in result.terms
                if t.patient_id == "SYN001" and t.hpo_code == "HP:0001250").how == "curated"


def test_a_location_missing_either_half_is_not_a_location(curated_gold, hcy_dir, log):
    """A segment with no trigger word, or a trigger the named sentence does not contain.

    Both are the unplaced state with something typed into it, ``reader._curated_anchor``'s rule,
    and the reason it is enforced here too: a saved trigger that cannot be found in its own segment
    cannot be matched back to the report by anything downstream.
    """
    from hpo_extraction.curation import store

    key = store.target_key("prior_annotation", "SYN001", "HP:0001873")
    log.append("edit", "SYN001", target_key=key, hpo_code="HP:0001873", segment_idx=3,
               trigger_word="")
    assert "HP:0001873" not in _build(curated_gold, hcy_dir).gold["SYN001"]

    log.append("edit", "SYN001", target_key=key, hpo_code="HP:0001873", segment_idx=0,
               trigger_word="thrombocytopenia")   # segment 0 is the case-history line
    assert "HP:0001873" not in _build(curated_gold, hcy_dir).gold["SYN001"]


def test_evidence_is_judged_per_code_not_per_row(curated_gold, hcy_dir, log):
    """One phenotype, two files, one of them wrong about the words.

    SYN002's HP:0001250 is this in reverse, prior_annotation says *convulsions*, daphne says
    *fits*, and neither occurs, so give the prior_annotation row a working evidence location and the term must
    survive even though the row adjudication picked (daphne's) still cannot be placed. Judging the
    rule row by row would delete a phenotype the report demonstrably supports, because the file that
    got it right was not the file that won the verdict.
    """
    from hpo_extraction.curation import store

    assert "HP:0001250" not in _build(curated_gold, hcy_dir).gold["SYN002"]

    log.append("anchor", "SYN002", target_key=store.target_key("prior_annotation", "SYN002", "HP:0001250"),
               hpo_code="HP:0001250", segment_idx=1, trigger_word="Feeding")
    result = _build(curated_gold, hcy_dir)
    assert "HP:0001250" in result.gold["SYN002"]
    # The row that carries the verdict is still daphne's, the evidence location is evidence, not a re-keying.
    row = next(t for t in result.terms
               if t.patient_id == "SYN002" and t.hpo_code == "HP:0001250")
    assert row.source == "daphne" and row.anchored


def test_keep_unanchored_prices_the_evidence_rule(curated_gold, hcy_dir):
    """The variant that turns the rule off recovers the unanchorable terms."""
    annotations, corpus, fold = _load(curated_gold, hcy_dir)
    built = curated_gold.variants(annotations, fold, names=["default", "keep_unanchored"],
                                  corpus=corpus)
    assert (built["keep_unanchored"].gold["SYN001"] - built["default"].gold["SYN001"]
            == {"HP:0001873"})
    assert built["keep_unanchored"].gold["SYN004"] == {"HP:0001250"}
    assert built["default"].gold["SYN004"] == set()


def test_building_with_the_rule_on_and_no_corpus_raises(curated_gold, hcy_dir):
    """Defaulting to "no evidence anywhere" would empty the ground truth silently.

    An empty ground truth is the one failure a P/R table cannot show you, every method scores 0 and the
    output looks like a result. So the missing input is a message naming the fix, not a default.
    """
    annotations, _, fold = _load(curated_gold, hcy_dir)
    with pytest.raises(ValueError, match="require_evidence"):
        curated_gold.build(annotations, fold)
    assert curated_gold.build(
        annotations, fold, curated_gold.VARIANTS["keep_unanchored"]).gold


# ── the cohort filter ────────────────────────────────────────────────────────

def test_the_cohort_is_every_annotated_report(curated_gold, hcy_dir):
    """SYN003 is the control: the fixture gives it no annotation from any source.

    The narrow filter this experiment shipped with existed because curation was half-done. It is
    finished, so the only reports left out are the ones nobody ever annotated, and those cannot be
    scored, because an empty ground-truth set would claim they have no phenotypes, not that nobody
    looked.
    """
    result = _build(curated_gold, hcy_dir)
    assert sorted(result.gold) == ["SYN001", "SYN002", "SYN004", "SYN005"]
    assert {r.patient_id for r in result.reports if not r.in_cohort} == {"SYN003"}


def test_an_excluded_report_is_absent_not_empty(curated_gold, hcy_dir):
    """An excluded report gets no ground truth entry at all, but still a line in the manifest."""
    result = _build(curated_gold, hcy_dir)
    assert "SYN003" not in result.gold
    assert next(r for r in result.reports
                if r.patient_id == "SYN003").reason == "no annotation from any source"


def test_a_suggestion_admits_a_report_no_file_annotated(curated_gold, hcy_dir, log):
    """One `suggest` event is enough, somebody read the report and proposed a term."""
    log.append("suggest", "SYN003", hpo_code="HP:0001250", hpo_name="Seizure",
               trigger_word="Routine", segment_idx=0)
    result = _build(curated_gold, hcy_dir)
    assert result.gold["SYN003"] == {"HP:0001250"}
    row = next(r for r in result.reports if r.patient_id == "SYN003")
    assert row.reason == "suggestion" and row.n_suggestions == 1


def test_approval_alone_does_not_admit_a_report_by_default(curated_gold, hcy_dir, log):
    """`confirm_patient` on a report with nothing on it is not an annotation.

    It would enter the cohort with an empty ground truth and take a false positive for every prediction any
    method makes on it. The signal stays in the vocabulary and can be switched on from config, this pins that it is off by default, which is the whole difference between the two settings.
    """
    log.append("confirm_patient", "SYN003")

    assert "approved" in curated_gold.CRITERIA
    assert "approved" not in curated_gold.DEFAULT_CRITERIA
    assert "SYN003" not in _build(curated_gold, hcy_dir).gold

    annotations, corpus, fold = _load(curated_gold, hcy_dir)
    opted_in = curated_gold.build(annotations, fold, criteria=("prior_annotation", "approved"),
                                  corpus=corpus)
    assert "SYN003" in opted_in.gold
    assert next(r for r in opted_in.reports if r.patient_id == "SYN003").reason == "approved"


def test_the_segmented_criterion_admits_reports_with_no_annotation(curated_gold, hcy_dir):
    """The opt-in cohort: every report the segmentation covers, empty ground truth included.

    That is a precision measurement, not a recall one, and defensible now every report has
    been read, but it is a different measurement, so it is a config flip and not the default. It
    also needs the corpus, and says so, not silently admitting nothing.
    """
    annotations, corpus, fold = _load(curated_gold, hcy_dir)
    criteria = curated_gold.DEFAULT_CRITERIA + ("segmented",)
    result = curated_gold.build(annotations, fold, criteria=criteria, corpus=corpus)
    assert result.gold["SYN003"] == set()
    assert "segmented" in next(r for r in result.reports if r.patient_id == "SYN003").reason

    with pytest.raises(ValueError, match="segmented"):
        curated_gold.cohort(annotations, fold, criteria)


def test_the_original_gold_criterion_admits_a_report_that_file_calls_empty(curated_gold, hcy_dir):
    """A row with an empty code list is an assertion of emptiness, and is admitted as one.

    SYN003 carries no annotation from any source, so the default cohort refuses it: an empty ground truth
    would claim it has no phenotypes, not that nobody looked. But the ORIGINAL ground truth listing
    it *is* somebody looking, that file holds the report to have nothing, so with that signal in
    `criteria` it enters with an empty ground truth and every prediction on it scores as a false positive.
    """
    annotations, corpus, fold = _load(curated_gold, hcy_dir)
    criteria = curated_gold.DEFAULT_CRITERIA + ("original_gold",)

    result = curated_gold.build(annotations, fold, criteria=criteria, corpus=corpus,
                                prior_gold_ids={"SYN001", "SYN003"})
    assert result.gold["SYN003"] == set()
    assert next(r for r in result.reports if r.patient_id == "SYN003").reason == "original_gold"


def test_the_original_gold_criterion_ignores_a_report_that_file_omits(curated_gold, hcy_dir):
    """The whole point of the signal: presence in the file, not presence in the corpus.

    A report the original ground truth never mentions carries no assertion at all, and is the case
    `segmented` admits and this one must not, otherwise the two criteria are the same switch.
    """
    annotations, corpus, fold = _load(curated_gold, hcy_dir)

    narrow = curated_gold.build(annotations, fold,
                                criteria=curated_gold.DEFAULT_CRITERIA + ("original_gold",),
                                corpus=corpus, prior_gold_ids={"SYN001"})
    assert "SYN003" not in narrow.gold

    wide = curated_gold.build(annotations, fold,
                              criteria=curated_gold.DEFAULT_CRITERIA + ("segmented",),
                              corpus=corpus)
    assert "SYN003" in wide.gold


def test_the_original_gold_criterion_needs_the_prior_report_list(curated_gold, hcy_dir):
    """Failing open would quietly drop the reports the criterion exists to admit."""
    annotations, _corpus, fold = _load(curated_gold, hcy_dir)
    with pytest.raises(ValueError, match="original_gold"):
        curated_gold.cohort(annotations, fold,
                            curated_gold.DEFAULT_CRITERIA + ("original_gold",))


def test_criteria_are_configurable_and_validated(curated_gold, hcy_dir):
    """Narrowing `criteria` narrows the cohort. An unknown signal raises, not being ignored.

    A silently dropped criterion would change the denominator of every number downstream without
    changing anything a reader can see.
    """
    annotations, corpus, fold = _load(curated_gold, hcy_dir)

    wide = {r.patient_id for r in curated_gold.cohort(annotations, fold, corpus=corpus)
            if r.in_cohort}
    narrow = {r.patient_id for r in curated_gold.cohort(annotations, fold, ("daphne",), corpus)
              if r.in_cohort}
    assert "SYN005" in wide and "SYN005" not in narrow

    with pytest.raises(ValueError, match="unknown cohort criteria"):
        curated_gold.cohort(annotations, fold, ("daphne", "vibes"))


def test_every_report_carries_its_admission_reason(curated_gold, hcy_dir, log):
    """The manifest has to say *which* signal admitted each report, not just that one did."""
    log.append("suggest", "SYN001", hpo_code="HP:0001903", hpo_name="Anemia",
               trigger_word="boy", segment_idx=0)
    result = _build(curated_gold, hcy_dir)
    by_id = {r.patient_id: r for r in result.reports}
    assert by_id["SYN001"].reason == "prior_annotation+daphne+suggestion"
    assert by_id["SYN005"].reason == "prior_annotation"
    assert by_id["SYN003"].reason == "no annotation from any source"


# ── the ground truth policy ──────────────────────────────────────────────────────────

def test_an_unruled_annotation_is_in_the_gold(curated_gold, hcy_dir):
    """Nobody has touched the fixture's log, so every survivor arrives on the `unruled` branch.

    SYN001 also exercises the fallback: the prior_annotation file carries HP:0007359, the confirmed pass
    does not, and under `adjudicated` that drop is an unconfirmed judgement, not a decision, so it survives into the ground truth.
    """
    result = _build(curated_gold, hcy_dir)
    assert result.gold["SYN001"] == {"HP:0001250", "HP:0001252", "HP:0007359"}
    assert {t.reason for t in result.terms
            if t.patient_id == "SYN001" and t.in_gold} == {"unruled"}


def test_an_unapproved_suggestion_is_in_the_gold(curated_gold, hcy_dir, log):
    """The second main rule: a proposal counts before anybody approves it."""
    log.append("suggest", "SYN001", hpo_code="HP:0001903", hpo_name="Anemia",
               trigger_word="markedly", segment_idx=2)
    result = _build(curated_gold, hcy_dir)
    assert "HP:0001903" in result.gold["SYN001"]
    row = next(t for t in result.terms if t.hpo_code == "HP:0001903")
    assert (row.source, row.status, row.reason) == ("new", "suggested", "suggested")


def test_a_suggestion_whose_trigger_is_not_in_its_segment_is_out(curated_gold, hcy_dir, log):
    """The evidence rule applies to this app's own rows too.

    The suggestion form validates the trigger against the segment, so this is a row whose segment
    was later repaired to a sentence the trigger is not in, and one rule for the files and a
    softer one for our own suggestions is how a ground truth ends up with terms nobody can point at.
    """
    log.append("suggest", "SYN001", hpo_code="HP:0001903", hpo_name="Anemia",
               trigger_word="pallor", segment_idx=0)
    result = _build(curated_gold, hcy_dir)
    assert "HP:0001903" not in result.gold["SYN001"]
    assert _reason(result, "SYN001", "HP:0001903") == "no_evidence"


def test_an_unsure_label_takes_a_term_out(curated_gold, hcy_dir, log):
    """Both `unsure` qualifiers exclude, whether they ride on a suggestion or an existing row."""
    from hpo_extraction.curation import store

    log.append("suggest", "SYN001", hpo_code="HP:0001903", hpo_name="Anemia",
               trigger_word="markedly", segment_idx=2, labels=["unsure_report"])
    log.append("label", "SYN001", target_key=store.target_key("daphne", "SYN001", "HP:0001250"),
               labels=["unsure_annotation"], hpo_code="HP:0001250")
    result = _build(curated_gold, hcy_dir)

    assert "HP:0001903" not in result.gold["SYN001"]
    assert "HP:0001250" not in result.gold["SYN001"]
    assert _reason(result, "SYN001", "HP:0001903") == "label:unsure_report"
    assert _reason(result, "SYN001", "HP:0001250") == "label:unsure_annotation"


def test_a_family_label_takes_a_term_out(curated_gold, hcy_dir, log):
    """A relative's finding leaves the ground truth.

    Unlike the doubt labels this is an editorial call, not a statement about evidence: the
    finding is real and the text supports it, so a system that extracts it is doing something
    defensible and counting it as a false positive would measure the annotation convention. Priced
    on its own by the ``keep_family`` variant, and shipped flagged, not deleted in the
    dataset export.
    """
    log.append("suggest", "SYN001", hpo_code="HP:0002014", hpo_name="Diarrhea",
               trigger_word="birth", segment_idx=2, labels=["family"])
    result = _build(curated_gold, hcy_dir)
    assert "HP:0002014" not in result.gold["SYN001"]
    assert _reason(result, "SYN001", "HP:0002014") == "label:family"


def test_a_label_excludes_an_existing_annotation_too(curated_gold, hcy_dir, log):
    """The exclusion is about the annotation, not about which file it arrived in."""
    from hpo_extraction.curation import store

    log.append("label", "SYN002", target_key=store.target_key("daphne", "SYN002", "HP:0011968"),
               labels=["family"], hpo_code="HP:0011968")
    result = _build(curated_gold, hcy_dir)
    assert "HP:0011968" not in result.gold["SYN002"]


def test_a_label_is_checked_before_the_evidence_rule(curated_gold, hcy_dir, log):
    """Reason ordering: a qualifier is a statement about the annotation, and it comes first.

    An unanchorable *and* doubtful term reported as `no_evidence` would say the curator never
    expressed an opinion about it, which is the opposite of what happened.
    """
    from hpo_extraction.curation import store

    log.append("label", "SYN001", target_key=store.target_key("prior_annotation", "SYN001", "HP:0001873"),
               labels=["family"], hpo_code="HP:0001873")
    result = _build(curated_gold, hcy_dir)
    assert _reason(result, "SYN001", "HP:0001873") == "label:family"


def test_an_unsure_label_beats_an_approval(curated_gold, hcy_dir, log):
    """The exclusion is checked first, on purpose.

    An approved-but-unsure term is a curator saying "this belongs in the set and I cannot tell
    whether the report supports it". Scoring a model against it measures the annotation.
    """
    from hpo_extraction.curation import store

    key = store.target_key("daphne", "SYN001", "HP:0001250")
    log.append("verdict", "SYN001", target_key=key, status="kept", hpo_code="HP:0001250")
    log.append("label", "SYN001", target_key=key, labels=["unsure_report"], hpo_code="HP:0001250")
    result = _build(curated_gold, hcy_dir)
    assert "HP:0001250" not in result.gold["SYN001"]


def test_removed_rejected_and_disputed_terms_are_out(curated_gold, hcy_dir, log):
    """The four lifecycle exclusions, each with its reason recorded.

    The user-facing rule names only the deletion proposal. The three verdicts are here because a
    curator who pressed *Remove* has said something stronger than silence, and dropping it would
    make Approve mode's buttons decorative.
    """
    from hpo_extraction.curation import store

    log.append("verdict", "SYN001", status="removed", hpo_code="HP:0001250",
               target_key=store.target_key("daphne", "SYN001", "HP:0001250"))
    log.append("suggest_delete", "SYN001", hpo_code="HP:0001252", text="wrong word",
               target_key=store.target_key("daphne", "SYN001", "HP:0001252"))
    event = log.append("suggest", "SYN002", hpo_code="HP:0001903", hpo_name="Anemia",
                       trigger_word="Feeding", segment_idx=1)
    log.append("reject", "SYN002", target_key=event["event_id"], hpo_code="HP:0001903")
    log.append("verdict", "SYN005", status="needs_work", hpo_code="HP:0004322",
               target_key=store.target_key("prior_annotation", "SYN005", "HP:0004322"))

    result = _build(curated_gold, hcy_dir)
    assert result.gold["SYN001"] == {"HP:0007359"}
    assert "HP:0001903" not in result.gold["SYN002"]
    assert "HP:0004322" not in result.gold["SYN005"]
    assert _reason(result, "SYN001", "HP:0001250") == "status:removed"
    assert _reason(result, "SYN001", "HP:0001252") == "status:delete_suggested"
    assert _reason(result, "SYN002", "HP:0001903") == "status:rejected"
    assert _reason(result, "SYN005", "HP:0004322") == "status:needs_work"


def test_a_withdrawn_suggestion_leaves_no_trace(curated_gold, hcy_dir, log):
    """`withdraw` deletes the row from the fold, so it is neither in the ground truth nor in the drop list."""
    event = log.append("suggest", "SYN001", hpo_code="HP:0001903", hpo_name="Anemia",
                       trigger_word="markedly", segment_idx=2)
    log.append("withdraw", "SYN001", target_key=event["event_id"])
    result = _build(curated_gold, hcy_dir)
    assert "HP:0001903" not in result.gold["SYN001"]
    assert not any(t.hpo_code == "HP:0001903" for t in result.terms)


def test_an_edited_code_is_the_one_scored(curated_gold, hcy_dir, log):
    """An `edit` is a repair, not a verdict, the repaired code is what the curator means.

    The evidence travels with it: the daphne row's own trigger word still evidence locations the annotation, so
    re-coding a term must not cost it its place in the ground truth.
    """
    from hpo_extraction.curation import store

    log.append("edit", "SYN001", target_key=store.target_key("daphne", "SYN001", "HP:0001250"),
               hpo_code="HP:0002133", hpo_name="Status epilepticus")
    result = _build(curated_gold, hcy_dir)
    assert "HP:0002133" in result.gold["SYN001"]
    assert "HP:0001250" not in result.gold["SYN001"]
    row = next(t for t in result.terms if t.hpo_code == "HP:0002133")
    assert row.edited and row.original_hpo_code == "HP:0001250" and row.anchored


# ── the variants ─────────────────────────────────────────────────────────────

def test_daphne_first_drops_a_prior_annotation_stand_in(curated_gold, hcy_dir):
    """The alternative reading: the confirmed pass dropping a term *is* the decision.

    SYN001's HP:0007359 is the case, prior_annotation has it, the confirmed pass does not.
    """
    result = _build(curated_gold, hcy_dir,
                    policy=curated_gold.GoldPolicy(prior_annotation_fallback="daphne_first"))
    assert result.gold["SYN001"] == {"HP:0001250", "HP:0001252"}


def test_daphne_first_still_falls_back_where_the_pass_never_ran(curated_gold, hcy_dir):
    """A patient with no confirmed row keeps its prior_annotation ground truth under either fallback.

    SYN005 has prior_annotation annotations and no daphne row at all, and `daphne_first` must not read "no
    daphne rows" as "no ground truth", or the report would sit in the cohort with nothing to score against.
    """
    result = _build(curated_gold, hcy_dir,
                    policy=curated_gold.GoldPolicy(prior_annotation_fallback="daphne_first"))
    assert result.gold["SYN005"] == {"HP:0001252", "HP:0003324", "HP:0004322"}


def test_approved_only_reproduces_the_apps_own_export(curated_gold, hcy_dir, log):
    """The restrictive floor: only adjudicated rows, which is what ``store.export_gold`` writes."""
    from hpo_extraction.curation import store

    log.append("verdict", "SYN001", status="kept", hpo_code="HP:0001250",
               target_key=store.target_key("daphne", "SYN001", "HP:0001250"))
    result = _build(curated_gold, hcy_dir, policy=curated_gold.VARIANTS["approved_only"])
    assert result.gold["SYN001"] == {"HP:0001250"}
    assert result.gold["SYN002"] == set()


def test_variants_share_one_cohort(curated_gold, hcy_dir):
    """Six ground truth definitions over one fixed report list, the only way the table is an ablation.

    A variant that also moved the cohort would mix a change of ground truth with a change of denominator,
    which is the confound this experiment exists to remove.
    """
    annotations, corpus, fold = _load(curated_gold, hcy_dir)
    built = curated_gold.variants(annotations, fold, corpus=corpus)

    assert set(built) == set(curated_gold.VARIANTS)
    cohorts = {name: sorted(result.gold) for name, result in built.items()}
    assert len(set(map(tuple, cohorts.values()))) == 1, cohorts


def test_keep_all_labels_keeps_what_default_drops(curated_gold, hcy_dir, log):
    """The variant that prices every label exclusion differs from default by those terms."""
    log.append("suggest", "SYN001", hpo_code="HP:0001903", trigger_word="markedly", segment_idx=2,
               labels=["unsure_annotation"])
    log.append("suggest", "SYN001", hpo_code="HP:0002014", trigger_word="birth", segment_idx=2,
               labels=["family"])
    annotations, corpus, fold = _load(curated_gold, hcy_dir)
    built = curated_gold.variants(annotations, fold, names=["default", "keep_family",
                                                            "keep_all_labels"], corpus=corpus)
    assert (built["keep_all_labels"].gold["SYN001"] - built["default"].gold["SYN001"]
            == {"HP:0001903", "HP:0002014"})
    # `keep_family` isolates the one editorial call: the relative's finding comes back, the
    # doubtful one does not.
    assert (built["keep_family"].gold["SYN001"] - built["default"].gold["SYN001"]
            == {"HP:0002014"})


def test_an_unknown_fallback_raises(curated_gold):
    with pytest.raises(ValueError, match="prior_annotation_fallback"):
        curated_gold.GoldPolicy(prior_annotation_fallback="whatever")


# ── the export ───────────────────────────────────────────────────────────────

def test_the_export_round_trips_through_the_repos_own_gold_loader(curated_gold, hcy_dir,
                                                                  tmp_path):
    """The CSV must be a drop-in for ``hcy_gt_path``, or none of the downstream reuse works."""
    from hpo_extraction.evaluation.datasets.hcy import HCYDataset

    result = _build(curated_gold, hcy_dir)
    path = tmp_path / "hcy_ground_truth_curated_eval.csv"
    curated_gold.write_gold_csv(path, result.gold)

    loaded = {str(k): {str(c).strip() for c in v if str(c).strip()}
              for k, v in HCYDataset(str(path), "").load_ground_truth().items()}
    assert loaded == result.gold


def test_the_export_omits_excluded_reports(curated_gold, hcy_dir, tmp_path):
    """No line at all for a report the filter refused, not a line with an empty cell."""
    result = _build(curated_gold, hcy_dir)
    path = tmp_path / "gold.csv"
    curated_gold.write_gold_csv(path, result.gold)
    assert "SYN003" not in path.read_text(encoding="utf-8")


def test_the_export_keeps_an_admitted_report_whose_gold_is_empty(curated_gold, hcy_dir, tmp_path):
    """The converse, and the reason the empty cell has to survive the round trip.

    An admitted report with no annotated term is the precision case: it belongs in the file with an
    empty ``hpo_codes`` cell so a scorer charges every prediction on it, and the repo's own loader
    has to read it back as an empty set, not as a missing report.
    """
    from hpo_extraction.evaluation.datasets.hcy import HCYDataset

    annotations, corpus, fold = _load(curated_gold, hcy_dir)
    result = curated_gold.build(annotations, fold,
                                criteria=curated_gold.DEFAULT_CRITERIA + ("original_gold",),
                                corpus=corpus, prior_gold_ids={"SYN003"})
    assert result.gold["SYN003"] == set()

    path = tmp_path / "gold.csv"
    curated_gold.write_gold_csv(path, result.gold)
    assert '"SYN003",""' in path.read_text(encoding="utf-8")

    loaded = HCYDataset(str(path), "").load_ground_truth()
    assert "SYN003" in loaded
    assert {str(c).strip() for c in loaded["SYN003"] if str(c).strip()} == set()


# ── the summary the results quote ────────────────────────────────────────────

def test_the_summary_reports_the_cohort_and_the_policy(curated_gold, hcy_dir, log):
    """Every number a reader needs to know a run's provenance is in one dict.

    The curation log keeps growing, so two runs of this experiment are comparable only when these
    agree, which makes them the thing to quote beside any figure taken from it.
    """
    log.append("suggest", "SYN005", hpo_code="HP:0004325", trigger_word="months", segment_idx=0,
               labels=["unsure_report"])
    result = _build(curated_gold, hcy_dir)
    summary = result.summary()

    assert summary["n_reports_in_cohort"] == 4
    assert summary["n_reports_seen"] == len(result.reports)
    assert summary["n_gold_pairs"] == sum(len(v) for v in result.gold.values())
    assert summary["n_terms_unanchored"] == sum(1 for t in result.terms if not t.anchored)
    assert summary["require_evidence"] is True
    assert summary["exclude_labels"] == list(curated_gold.DEFAULT_EXCLUDE_LABELS)
    assert summary["criteria"] == list(curated_gold.DEFAULT_CRITERIA)


def test_a_missing_curation_log_folds_to_empty(curated_gold, hcy_dir):
    """No log yet is a state, not an error: it is this experiment's own baseline."""
    annotations, corpus, _ = _load(curated_gold, hcy_dir)
    fold = curated_gold.load_curation(str(hcy_dir / "does_not_exist"))
    assert fold["rows"] == {} and fold["confirmed"] == set()
    assert curated_gold.build(annotations, fold, corpus=corpus).gold
