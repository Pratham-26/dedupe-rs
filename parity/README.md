# Parity harness

Generates golden vectors from the reference Python `dedupe` implementation and
stores them under `../dedupe-rs/tests/golden/` for the Rust integration tests.

## Setup

The Python reference must be importable (install it in editable mode from
`../dedupe` plus its dependencies), e.g.:

```bash
pip install -e ../dedupe affinegap categorical-distance doublemetaphone \
    highered simplecosine haversine BTrees==5.2 zope.index \
    dedupe_Levenshtein_search scikit-learn
```

## Generators

| Script | Output | Covers |
| --- | --- | --- |
| `gen_vectors.py` | `predicate_functions.json`, `comparators.json` | blocking predicate functions and field comparators |
| `gen_dm.py` | `double_metaphone.json` | Double Metaphone |
| `gen_model.py` | `datamodel.json`, `blocking.json`, `tfidf.json`, `clustering.json` | data model, fingerprinter, TF-IDF index, clustering |
| `gen_e2e.py` | `e2e.json` | a trained `Dedupe` model + its partitions |

Regenerate everything:

```bash
python gen_vectors.py
python gen_dm.py
python gen_model.py
python gen_e2e.py
```

## Profiling

`profile.py gen` writes the shared dataset + model used by
`dedupe-rs/examples/profile.rs`; `profile.py bench N` times the Python pipeline
with `N` cores.
