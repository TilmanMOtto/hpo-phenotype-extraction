from .pbert import process_text2phrases_pbert, set_stanza_dir


# PhenoRAG patch: upstream imported the benepar entity-extraction variants here, which pulls in
# ``benepar`` / ``spacy`` / ``torch-struct`` at package import. Those live in a SEPARATE environment
# (``autopcr_ee_venv``, src/AutoPCR/requirements_ee.txt): upstream pins transformers 4.49 and
# protobuf 6, which PhenoRAG_marc_env cannot also hold. The parse runs there once per cohort
# (extract_phrases.py) and leaves phrases_benepar.json / phrases_conjunct.json beside the corpus,
# which run_gsc_test_ner loads instead of importing this. The import is deferred so the linker
# run in the main env never needs the parser stack. See PATCHES.md.
def process_text2phrases_benepar(*args, **kwargs):
    from .benepar import process_text2phrases_benepar as _fn

    return _fn(*args, **kwargs)


def process_text2phrases_abbr(*args, **kwargs):
    from .benepar import process_text2phrases_abbr as _fn

    return _fn(*args, **kwargs)
