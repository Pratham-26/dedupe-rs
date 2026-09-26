# dedupe-rs (Python)

Python bindings for the [Rust port of dedupe](../) — fuzzy matching,
de-duplication and entity resolution over structured data.

The public API follows the Python `dedupe` library, while the blocking,
indexing, distance computation, scoring, training and clustering run in Rust.

## Install

```bash
pip install dedupe-rs
```

Building from source requires a Rust toolchain and [maturin](https://maturin.rs):

```bash
cd python
maturin build --release -o dist
pip install dist/dedupe_rs-*.whl
```

or, for development:

```bash
maturin develop --release
```

## Usage

```python
import dedupe_rs
from dedupe_rs import variables as V

data = {
    0: {"name": "Bob",      "age": "51"},
    1: {"name": "Bob B.",   "age": "51"},
    2: {"name": "Tina",     "age": "15"},
}

deduper = dedupe_rs.Dedupe([V.String("name"), V.String("age")])
deduper.prepare_training(data)

while True:
    try:
        pair = deduper.uncertain_pairs()
    except IndexError:
        break
    # ... ask the user, then label the pair
    deduper.mark_pairs({"match": [(data[0], data[1])], "distinct": [(data[0], data[2])]})

deduper.train()
clusters = deduper.partition(data, threshold=0.5)
print(clusters)  # [((0, 1), (0.79, 0.79)), ((2,), (1.0,))]
```

Interactive labelling is available too:

```python
dedupe_rs.console_label(deduper)
```

### Record linkage

```python
linker = dedupe_rs.RecordLink([V.String("name")])
linker.prepare_training(data_1, data_2)
# ... label pairs ...
linker.train()
links = linker.join(data_1, data_2, threshold=0.5, constraint="one-to-one")
```

### Gazetteer

```python
gaz = dedupe_rs.Gazetteer([V.String("name")])
gaz.prepare_training(messy, canonical)
# ... label pairs ...
gaz.train()
gaz.index(canonical)
matches = gaz.search(messy, threshold=0.5, n_matches=2)
```

### Settings

Trained models are persisted as inspectable JSON (rather than a pickle):

```python
with open("settings.json", "w") as f:
    deduper.write_settings(f)

with open("settings.json") as f:
    static = dedupe_rs.StaticDedupe.from_settings(f)
clusters = static.partition(data, threshold=0.5)
```

## Variable types

`dedupe_rs.variables` provides `String`, `ShortString`, `Text`, `Exact`,
`Categorical`, `Exists`, `Price`, `LatLong`, `Set`, `Interaction` and `Custom`.

`Custom` takes a Python comparator `f(value_a, value_b) -> float`. Because it
calls back into Python it forces single-threaded scoring.

## Compatibility notes

* **Settings format** is JSON (`dedupe_rs` specific) rather than Python
  pickles; Python `dedupe` settings files are not interchangeable.
* **Randomness** in active learning and random-forest rule generation is
  seeded and deterministic. Pass `seed=` to `Dedupe`/`RecordLink`/`Gazetteer`.
* `partition` returns `(ids_tuple, scores_tuple)` pairs instead of tuples
  containing NumPy arrays.
* Records may use `int` or `str` ids; field values may be `str`, `int`,
  `float`, `None`, `list`, `tuple`, `set`/`frozenset`.

## Tests

```bash
python -m pytest tests -q
```
