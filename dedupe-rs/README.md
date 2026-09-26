# dedupe (Rust)

A Rust port of the [`dedupe`](https://github.com/dedupeio/dedupe) Python library —
machine-learning-based fuzzy matching, de-duplication and entity resolution over
structured data.

The port mirrors the Python module layout so behaviour can be compared side by
side, and every algorithmic component is validated against the reference
implementation with generated golden vectors (see [Parity](#parity)).

## Status

| Area | Python module | Rust module | Parity |
| --- | --- | --- | --- |
| Character predicates | `cpredicates.pyx` | `cpredicates.rs` | exact |
| Blocking predicate functions | `predicate_functions.py` | `predicate_functions.rs` | exact (53,136 cases) |
| Double Metaphone | `doublemetaphone` (C++) | `double_metaphone.rs` | exact (133,778 cases) |
| Comparators | `affinegap`, `categorical-distance`, `simplecosine`, `haversine`, `higered` | `comparators.rs` | exact (2,622 cases) |
| Field types | `variables/*` | `variables.rs` | exact |
| Distance matrix | `datamodel.py` | `datamodel.rs` | exact (12 configs) |
| TF-IDF / Levenshtein indexes | `tfidf.py`, `canopy_index.py`, `levenshtein.py` | `index.rs` | exact membership |
| Fingerprinter | `blocking.py` | `blocking.rs` | exact (192 cases) |
| Clustering | `clustering.py` + SciPy | `clustering.rs` | exact (centroid/UPGMC + `fcluster`) |
| Canonicalization | `canonical.py` | `canonical.rs` | exact |
| Training (blocking rules) | `training.py` | `training.rs` | structural |
| Branch and bound | `branch_and_bound.py` | `branch_and_bound.rs` | exact |
| Active learning | `labeler.py` + scikit-learn | `labeler.rs` + `logistic.rs` | deterministic (seeded) |
| Scoring / core | `core.py` | `core.rs` | exact |
| API | `api.py` | `api.rs` | functional |
| Convenience | `convenience.py` | `convenience.rs` | functional |
| Serialization | `serializer.py` (JSON) | `serializer.rs` | exact wire format |

## Quick start

```rust
use dedupe::api::{Dedupe, JoinConstraint, RecordLink};
use dedupe::variables::{self, VariableDef};
use dedupe::{Data, Record, RecordId, Value};

let variables = vec![
    VariableDef::Field(variables::string("name", false)),
    VariableDef::Field(variables::string("age", false)),
];

let mut deduper = Dedupe::new(variables, 4, true)?;
deduper.prepare_training(&data)?;

// Ask for the most informative pair, label it, repeat...
let (a, b) = deduper.uncertain_pairs()?;
deduper.mark_pairs(&dedupe::serializer::TrainingData {
    match_: vec![(a, b)],
    distinct: vec![],
})?;

deduper.train(0.9, true)?;
let clusters = deduper.partition(&data, 0.5)?;
```

Settings can be persisted as inspectable JSON (rather than Python pickles):

```rust
use dedupe::api::{Settings, StaticDedupe};

let settings = Settings::from_matching(&deduper.matching);
let json = serde_json::to_string_pretty(&settings)?;
// ...
let static_dedupe = StaticDedupe::from_settings(&settings, 1, true)?;
let clusters = static_dedupe.partition(&data, 0.5)?;
```

## Design notes

* **Values.** Python's dynamically typed records are represented by the
  `Value` enum (`Null`, `Bool`, `Int`, `Float`, `Str`, `List`, `Tuple`, `Set`),
  with `py_str`/`py_repr` helpers that reproduce Python's `str()`/`repr()` for
  the values blocking predicates stringify.
* **Predicates.** `Predicate` is an object-safe trait; predicates are held in
  `Rc<dyn Predicate>` (`PredRef`) and keyed by their Python `repr`, exactly like
  the Python `__hash__`/`__eq__`.
* **Indexes.** The TF-IDF index is a faithful port of Zope's `CosineIndex`
  (including `mass_weightedUnion`'s balanced f32 merge and `byValue`'s
  f32-narrowed threshold), so membership matches. The Levenshtein index uses a
  BK-tree with exact distances.
* **Clustering.** SciPy's `fast_linkage` (fastcluster's *Generic Clustering
  Algorithm*), including its binary heap tie-breaking, and `fcluster`'s
  `get_max_dist_for_each_cluster` + `cluster_monocrit` are ported verbatim.
* **Classifier.** scikit-learn's `LogisticRegression` and `GridSearchCV` are
  replaced by an IRLS/Newton solver for the same L2-penalised objective plus a
  stratified k-fold F1 grid search.
* **Parallelism.** Scoring uses a `rayon` thread pool sized to `num_cores`.

## Deliberate deviations

* **Settings format.** Python pickles its settings; the port uses a documented
  JSON schema (`api::Settings`) so trained models are portable and inspectable.
  `Custom` variables require re-registering their comparator by label via
  `variables::register_custom_comparator`.
* **Randomness.** Active learning and random-forest candidate generation use a
  seeded, deterministic RNG instead of Python's global `random`/`numpy.random`.
  Results are therefore reproducible rather than byte-identical to a Python run.
* **TF-IDF tie order.** The membership of `search` matches exactly. For equal
  scores the *ordering* can differ (irrelevant to blocking, which uses sets).
* **Canopy assignment order.** Python iterates the predicate set in
  hash-randomised order; the port uses a deterministic order. This can change
  which record becomes a canopy centre and, on adversarial data, change a
  handful of candidate pairs (measured at ~0.008% on the profiling dataset).
* **Storage.** The Python API uses SQLite and `memmap` for blocking/scoring to
  bound memory; the port keeps these in memory (streaming can be added without
  changing the algorithms).

## Testing

* Unit tests live next to the code (`cargo test --lib`).
* Cross-language parity tests read golden vectors generated by
  [`../parity`](../parity):
  * `parity_predicate_functions` — 53,136 predicate-function cases.
  * `parity_double_metaphone` — 133,778 Double Metaphone cases.
  * `parity_comparators` — affine gap, CRF edit distance, categorical,
    cosine (text/set), haversine, price, exact.
  * `parity_datamodel` — distance-matrix layout and values.
  * `parity_blocking` — fingerprinter + index predicate coverage.
  * `parity_tfidf` — TF-IDF index membership.
  * `parity_clustering` — connected components, SciPy centroid linkage and
    `fcluster` cuts.
  * `parity_e2e` — end-to-end partition from a trained Python model.

Regenerate the golden vectors with:

```bash
cd parity
python gen_vectors.py && python gen_dm.py && python gen_model.py && python gen_e2e.py
```

## Profiling

See [PROFILING.md](PROFILING.md) for the head-to-head time and memory
benchmark against the Python implementation.

On a shared dataset with 5.3 M candidate pairs (8,000 records), the port is
**~5–8× faster** and uses **1.4–3.1× less peak memory** than the Python
implementation. Notable performance work:

* allocation-free affine-gap comparisons (ASCII fast path + thread-local
  workspace) and allocation-free scoring (`compare_into` / `distance_row_into`);
* a bitset pair generator that replaces ~50 M redundant hash insertions with
  bit sets when the `n²/2` bit matrix fits in memory;
* compact connected components and SciPy's `max_components` re-filtering, which
  bounds the O(n²) hierarchical clustering on giant components;
* `rayon` scoring pools sized exactly to `num_cores`.

Two examples are included:

* `cargo run --release --example profile [cores] [--json]` — end-to-end pipeline
  timings and peak RSS for a dataset written by `parity/profile.py gen`.
* `cargo run --release --example micro` — micro-benchmarks for the scoring hot
  path.
