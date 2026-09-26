# dedupe (Rust port)

An end-to-end Rust port of [`dedupeio/dedupe`](https://github.com/dedupeio/dedupe) —
the Python library for machine-learning-based fuzzy matching, de-duplication and
entity resolution over structured data.

The Python reference implementation is used as an executable specification: every
algorithm in the port is validated against generated golden vectors, and the two
implementations are benchmarked head to head for time and memory.

## Layout

| Path | Contents |
| --- | --- |
| `dedupe-rs/` | The Rust crate. See [`dedupe-rs/README.md`](dedupe-rs/README.md). |
| `dedupe-rs/src` | Port of `dedupe/*` (predicates, comparators, blocking, indexes, clustering, training, active learning, API). |
| `dedupe-rs/tests` | Unit tests, cross-language parity tests, and API smoke tests. |
| `dedupe-rs/tests/golden` | Golden vectors generated from the Python reference. |
| `dedupe-rs/PROFILING.md` | Time and memory benchmark against the Python implementation. |
| `parity/` | Python harness that generates the golden vectors and runs the benchmarks. |

## Quick start

```bash
cd dedupe-rs
cargo test            # unit + parity + smoke tests
cargo test --release  # same, faster

# End-to-end benchmark (needs the golden dataset, see below)
cargo run --release --example profile 8
```

## Parity

Every algorithmic component is checked against the reference implementation:

| Component | Generator | Cases |
| --- | --- | ---: |
| Double Metaphone | `parity/gen_dm.py` | 133,778 |
| Blocking predicate functions | `parity/gen_vectors.py` | 53,136 |
| Field comparators (affine gap, CRF, categorical, cosine, haversine, price, exact) | `parity/gen_vectors.py` | 2,622 |
| Distance matrix | `parity/gen_model.py` | 12 configs |
| Fingerprinter + index predicates | `parity/gen_model.py` | 192 |
| TF-IDF index membership | `parity/gen_model.py` | 104 |
| Clustering / SciPy linkage | `parity/gen_model.py` | 31 + 37 |
| Trained model end-to-end partition | `parity/gen_e2e.py` | 1 |

Regenerate the golden vectors (requires the Python reference and its
dependencies):

```bash
cd parity
python gen_vectors.py && python gen_dm.py && python gen_model.py && python gen_e2e.py
```

The reference implementation is expected at `../dedupe` (a clone of
`dedupeio/dedupe`); see [`parity/README.md`](parity/README.md) for setup.

## Performance

On a shared dataset (8,000 records, 5.3 M candidate pairs, 13 blocking
predicates) the port is **~5–8× faster** than Python and uses **1.4–3.1× less
peak memory**:

| Cores | Python total | Rust total | Speed-up | Python peak | Rust peak |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 129.4 s | 26.5 s | 4.9× | 2547 MB | 1301 MB |
| 8 | 80.1 s | 14.4 s | 5.6× | 2548 MB | 1763 MB |

See [`dedupe-rs/PROFILING.md`](dedupe-rs/PROFILING.md) for the full breakdown,
per-phase memory, and the optimization log. Reproduce with:

```bash
cd parity && python bench_all.py 2000 4000 8000 --cores 1,8
```

## License and attribution

This is a derivative work of [dedupe](https://github.com/dedupeio/dedupe),
which is distributed under the MIT License
(Copyright © 2014 Forest Gregg, Derek Eder, DataMade and Contributors).
The original license is reproduced in [`dedupe-rs/LICENSE`](dedupe-rs/LICENSE),
and this port is released under the same terms.

The Double Metaphone implementation in `dedupe-rs/src/double_metaphone.rs` is a
Rust port of the C++ implementation shipped with the `doublemetaphone` Python
package (Lawrence Philips' Double Metaphone, C++ by Maurice Aubrey). The
clustering module reproduces SciPy's `fast_linkage` / `fcluster` behaviour
(originally from Daniel Müllner's fastcluster), and the TF-IDF index reproduces
Zope's `CosineIndex` (`zope.index`, Zope Public License 2.1).
