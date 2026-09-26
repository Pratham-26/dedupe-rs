# Profiling: Python vs Rust

Both implementations run the same pipeline over the same dataset and the same
blocking predicates / classifier:

* `name` and `age` `String` fields, sizes 2,000 / 4,000 / 8,000 records.
* 13 blocking predicates (7 simple + 3 TF-IDF index + 2 Levenshtein index),
  representative of a trained dedupe model.
* Phases timed: `index_all`, blocking + pair generation, `score`, `cluster`.
* Peak resident memory via `PeakWorkingSetSize` (Rust) / `psutil.peak_wset`
  (Python).

Everything below was produced by a single run of `bench_all.py` so the two
implementations see identical machine conditions (absolute times drift by up to
~1.7x with CPU state; the ratios are stable).

## Harness

```bash
cd parity
python bench_all.py 2000 4000 8000 --cores 1,8
```

which drives:

```bash
python profile.py gen   1 8000     # shared dataset + model -> tests/golden/profile.json
python profile.py bench 1 8000 --json
cd ../dedupe-rs && cargo run --release --example profile 1 --json
```

## Wall-clock time and peak memory

| Impl | Records | Cores | Pairs | index (s) | blocking (s) | score (s) | cluster (s) | total (s) | peak RSS (MB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| python | 2000 | 1 | 528,145 | 0.16 | 1.00 | 6.06 | 2.46 | 10.74 | 388 |
| rust | 2000 | 1 | 529,100 | 0.04 | 0.54 | 0.75 | 0.24 | **1.58** | **124** |
| python | 2000 | 8 | 528,145 | 0.15 | 1.02 | 3.40 | 2.48 | 8.17 | 384 |
| rust | 2000 | 8 | 529,100 | 0.04 | 0.55 | 0.21 | 0.25 | **1.05** | **185** |
| python | 4000 | 1 | 2,855,520 | 0.34 | 8.63 | 31.69 | 14.83 | 63.55 | 1287 |
| rust | 4000 | 1 | 2,857,203 | 0.09 | 1.82 | 6.97 | 2.52 | **11.40** | **585** |
| python | 4000 | 8 | 2,855,520 | 0.34 | 8.93 | 8.17 | 14.65 | 40.26 | 1285 |
| rust | 4000 | 8 | 2,857,203 | 0.09 | 1.82 | 1.14 | 2.78 | **5.83** | **947** |
| python | 8000 | 1 | 5,342,873 | 0.67 | 20.51 | 62.40 | 30.04 | 129.37 | 2547 |
| rust | 8000 | 1 | 5,343,325 | 0.19 | 4.16 | 16.00 | 6.16 | **26.51** | **1301** |
| python | 8000 | 8 | 5,342,873 | 0.65 | 20.07 | 14.27 | 29.23 | 80.09 | 2548 |
| rust | 8000 | 8 | 5,343,325 | 0.18 | 4.18 | 3.75 | 6.26 | **14.37** | **1763** |

## Summary

| Records | Cores | Python total | Rust total | Speed-up | Python peak | Rust peak | Memory |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2000 | 1 | 10.74 s | 1.58 s | **6.8×** | 388 MB | 124 MB | **3.1× less** |
| 2000 | 8 | 8.17 s | 1.05 s | **7.8×** | 384 MB | 185 MB | **2.1× less** |
| 4000 | 1 | 63.55 s | 11.40 s | **5.6×** | 1287 MB | 585 MB | **2.2× less** |
| 4000 | 8 | 40.26 s | 5.83 s | **6.9×** | 1285 MB | 947 MB | **1.4× less** |
| 8000 | 1 | 129.37 s | 26.51 s | **4.9×** | 2547 MB | 1301 MB | **2.0× less** |
| 8000 | 8 | 80.09 s | 14.37 s | **5.6×** | 2548 MB | 1763 MB | **1.4× less** |

Rust is **4.9–7.8× faster** and uses **1.4–3.1× less peak memory**. The memory
advantage shrinks with thread count because each rayon worker keeps its own row
buffer and rayon collects per-thread result vectors.

## Where Rust memory goes (8,000 records, 1 core)

| Phase | Cumulative peak | Delta |
|---|---:|---:|
| after `index_all` | 50 MB | 50 MB |
| after blocking + pairs | 476 MB | 426 MB |
| after `score` | 718 MB | 242 MB |
| after `cluster` | 1297 MB | 579 MB |

The `cluster` spike is the centroid linkage over the large connected component
(~7,000 nodes → O(n²) ≈ 24 M `f64` condensed matrix, plus the working copy).
Python pays the same cost in SciPy; it is what makes the 8,000-record run peak
at 2.5 GB.

## Optimizations applied during this pass

1. **Affine-gap kernel** (`comparators.rs`): ASCII fast path (operate on bytes
   instead of collecting `Vec<char>`), a reusable thread-local workspace
   instead of four heap allocations per call, an early return for equal strings,
   and splitting the inner loop so the abbreviation branch disappears. This cut
   per-call cost from ~1.25 µs to ~0.63 µs and removed ~40 M allocations per
   million pairs.
2. **Allocation-free scoring** (`comparators.rs`, `datamodel.rs`, `core.rs`):
   `Comparator::compare_into`, `DistanceCalculator::distance_row_into` and
   `Classifier::predict_proba_f32` let the scorer reuse one row buffer per
   thread instead of allocating a `Vec<f32>` + `Vec<f64>` per pair.
3. **`FxHash`** for the hot internal maps/sets (index state, blocks, pair sets).
4. **Bitset pair generation** (`api.rs`): overlapping blocks produce ~50 M raw
   pairs for 5.3 M distinct pairs. When the upper-triangle bit matrix fits in
   256 MB, pairs are marked in a bitset instead of inserted into a hash set
   (falling back to a hash set for larger datasets). Pair generation dropped
   from ~5 s to effectively free.
5. **Compact connected components** (`clustering.rs`): `component_ranges`
   returns an edge-order array plus component offsets instead of one `Vec` per
   component (~5 M allocations for 5 M singleton components).
6. **`max_components` filtering** (`clustering.rs`): ported
   `_connected_components`'s recursive score-threshold re-filtering, which
   Python uses to bound hierarchical clustering on giant components. It was
   previously ignored by the port.
7. **In-place centroid linkage**: `linkage_centroid_in_place` reuses the
   condensed matrix as its working array, avoiding a second O(n²) allocation.
8. **Thread-count fidelity** (`core.rs`): scoring now builds a rayon pool sized
   exactly to `num_cores`, matching Python's `num_cores` semantics (previously
   `par_iter` used the global pool).

## Remaining head-room

* `score` is the dominant Rust phase (16 s of 26.5 s at 8,000 records) and is
  bound by the serial dependency chain in the affine-gap recurrence
  (~0.63 µs for a 9×9 comparison, ~7 ns/cell).
* Blocking (~4.2 s) is sequential because canopy predicates mutate shared
  state; parallelising it would need per-thread indices or a two-phase canopy.
* Fusing blocking and scoring (streaming pairs in chunks) would remove the
  ~340 MB `pairs` vector that is alive during scoring.
