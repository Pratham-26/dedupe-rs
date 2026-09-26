"""Tests for the dedupe_rs Python package.

Run with:  python -m pytest python/tests -q
"""

import io
import json

import pytest

import dedupe_rs
from dedupe_rs import variables as V

DATA = {
    0: {"name": "Bob", "age": "51"},
    1: {"name": "Linda", "age": "50"},
    2: {"name": "Gene", "age": "12"},
    3: {"name": "Tina", "age": "15"},
    4: {"name": "Bob B.", "age": "51"},
    5: {"name": "bob belcher", "age": "51"},
    6: {"name": "linda ", "age": "50"},
    7: {"name": "Gene Belcher", "age": "12"},
}

TRAIN = {
    "match": [
        (DATA[0], DATA[4]),
        (DATA[0], DATA[5]),
        (DATA[1], DATA[6]),
        (DATA[2], DATA[7]),
        (DATA[4], DATA[5]),
    ],
    "distinct": [
        (DATA[0], DATA[1]),
        (DATA[2], DATA[3]),
        (DATA[0], DATA[2]),
        (DATA[1], DATA[3]),
        (DATA[4], DATA[1]),
        (DATA[5], DATA[3]),
    ],
}


def names():
    return [V.String("name"), V.String("age")]


def trained() -> dedupe_rs.Dedupe:
    deduper = dedupe_rs.Dedupe(names(), num_cores=1, in_memory=True)
    deduper.prepare_training(DATA)
    deduper.mark_pairs(TRAIN)
    deduper.train(recall=0.8)
    return deduper


def clusters_of(clusters):
    return {frozenset(ids) for ids, _ in clusters}


def test_version_and_exports():
    assert isinstance(dedupe_rs.__version__, str)
    for name in ["Dedupe", "RecordLink", "Gazetteer", "StaticDedupe", "console_label", "canonicalize"]:
        assert hasattr(dedupe_rs, name)


def test_variable_config_roundtrip():
    cfg = V.String("name", name="n", has_missing=True).to_config()
    assert cfg["kind"] == "String"
    assert cfg["field"] == "name"
    assert cfg["name"] == "n"
    assert cfg["has_missing"] is True

    assert V.Categorical("t", ["a", "b"]).to_config()["categories"] == ["a", "b"]
    assert V.Interaction("a", "b").to_config()["interactions"] == ["a", "b"]


def test_dedupe_train_and_partition():
    deduper = trained()
    clusters = deduper.partition(DATA, threshold=0.5)
    groups = clusters_of(clusters)
    # Every record is accounted for exactly once.
    flat = [i for ids, _ in clusters for i in ids]
    assert sorted(flat) == list(range(8))
    # The known duplicates cluster together.  (With recall=0.8 the learner
    # keeps a single n-gram canopy predicate, matching the Python reference,
    # which links these pairs but leaves record 5 as a singleton.)
    assert any({0, 4} <= g for g in groups)
    assert any({1, 6} <= g for g in groups)
    assert any({2, 7} <= g for g in groups)
    # Scores are per-member floats in [0, 1].
    for _, scores in clusters:
        assert all(0.0 <= s <= 1.0 for s in scores)


def test_uncertain_pairs_and_index_error():
    deduper = dedupe_rs.Dedupe(names(), num_cores=1, in_memory=True)
    deduper.prepare_training(DATA)
    pair = deduper.uncertain_pairs()
    assert len(pair) == 2
    assert isinstance(pair[0], dict)
    # Drain the learner; exhaustion must raise IndexError.
    for _ in range(10_000):
        try:
            deduper.uncertain_pairs()
        except IndexError:
            break
    else:  # pragma: no cover
        pytest.fail("uncertain_pairs never raised IndexError")


def test_training_pairs_and_write_read_training():
    deduper = dedupe_rs.Dedupe(names(), num_cores=1, in_memory=True)
    deduper.prepare_training(DATA)
    deduper.mark_pairs(TRAIN)
    pairs = deduper.training_pairs
    assert len(pairs["match"]) == len(TRAIN["match"])
    assert len(pairs["distinct"]) == len(TRAIN["distinct"])

    buf = io.StringIO()
    dedupe_rs.write_training(TRAIN, buf)
    buf.seek(0)
    loaded = dedupe_rs.read_training(buf)
    assert loaded == TRAIN

    # The matcher-level read/write round trip.
    buf2 = io.StringIO()
    deduper.write_training(buf2)
    buf2.seek(0)
    other = dedupe_rs.Dedupe(names(), num_cores=1, in_memory=True)
    other.read_training(buf2)
    assert other.training_pairs == pairs


def test_settings_roundtrip_reproduces_partition():
    deduper = trained()
    expected = clusters_of(deduper.partition(DATA, 0.5))

    buf = io.StringIO()
    deduper.write_settings(buf)
    payload = json.loads(buf.getvalue())
    assert payload["version"] == 1
    assert isinstance(payload["predicates"], list)
    assert "coef" in payload["classifier"]

    buf.seek(0)
    stat = dedupe_rs.StaticDedupe.from_settings(buf, num_cores=1, in_memory=True)
    assert clusters_of(stat.partition(DATA, 0.5)) == expected

    # settings_to_dict exposes the same payload as a dict.
    buf.seek(0)
    as_dict = dedupe_rs.settings_to_dict(buf)
    assert as_dict["classifier"]["coef"] == payload["classifier"]["coef"]


def test_pairs_and_score():
    deduper = trained()
    pairs = deduper.pairs(DATA)
    assert pairs, "expected some blocking pairs"
    scores = deduper.score(pairs)
    assert scores
    for (a, b), score in scores:
        assert a <= b
        assert 0.0 <= score <= 1.0
    # Greedy one-to-one returns at most one link per id.
    links = deduper.one_to_one(scores, 0.5)
    left = [a for (a, _), _ in links]
    right = [b for (_, b), _ in links]
    assert len(left) == len(set(left))
    assert len(right) == len(set(right))


def test_record_link_join():
    data_1 = {k: v for k, v in DATA.items() if k % 2 == 0}
    data_2 = {k: v for k, v in DATA.items() if k % 2 == 1}
    linker = dedupe_rs.RecordLink(names(), num_cores=1, in_memory=True)
    linker.prepare_training(data_1, data_2)
    linker.mark_pairs(
        {
            "match": [(DATA[0], DATA[5])],
            "distinct": [(DATA[0], DATA[3])],
        }
    )
    linker.train(recall=0.8)
    links = linker.join(data_1, data_2, threshold=0.5, constraint="many-to-many")
    assert links
    for (a, b), score in links:
        assert a in data_1 and b in data_2


def test_gazetteer_search():
    gaz = dedupe_rs.Gazetteer(names(), num_cores=1, in_memory=True)
    gaz.prepare_training(DATA, DATA)
    gaz.mark_pairs({"match": [(DATA[0], DATA[5])], "distinct": [(DATA[0], DATA[3])]})
    gaz.train(recall=0.8)
    gaz.index(DATA)
    assert gaz.indexed_len == len(DATA)
    results = gaz.search({100: {"name": "bob belcher", "age": "51"}}, threshold=0.5, n_matches=2)
    assert 100 in results
    assert len(results[100]) <= 2
    gaz.unindex(DATA)
    assert gaz.indexed_len == 0


def test_canonicalize():
    records = [
        {"name": "mary crane", "address": "123 main st", "zip": "12345"},
        {"name": "mary crane east", "address": "123 main street", "zip": ""},
        {"name": "mary crane west", "address": "123 man st", "zip": ""},
    ]
    rep = dedupe_rs.canonicalize(records)
    assert rep == {"name": "mary crane", "address": "123 main street", "zip": "12345"}


def test_training_data_dedupe():
    data = {
        1: {"name": "bob", "key": "bob"},
        2: {"name": "bob", "key": "bob"},
        3: {"name": "sue", "key": "sue"},
    }
    training = dedupe_rs.training_data_dedupe(data, "key", training_size=100)
    assert ("match" in training) and ("distinct" in training)
    assert training["match"], "expected a matched pair sharing the common key"


def test_custom_variable():
    def comparator(a, b):
        return 1.0 if a == b else 0.0

    variables = [V.Custom("name", comparator), V.Exact("age")]
    deduper = dedupe_rs.Dedupe(variables, num_cores=1, in_memory=True)
    deduper.prepare_training(DATA)
    deduper.mark_pairs(
        {
            "match": [(DATA[0], DATA[5])],
            "distinct": [(DATA[0], DATA[3]), (DATA[1], DATA[2])],
        }
    )
    deduper.train(recall=0.8)
    clusters = deduper.partition(DATA, 0.5)
    flat = [i for ids, _ in clusters for i in ids]
    assert sorted(flat) == list(range(8))


def test_string_record_ids():
    data = {
        "a": {"name": "Bob", "age": "51"},
        "b": {"name": "Bob B.", "age": "51"},
        "c": {"name": "Tina", "age": "15"},
    }
    deduper = dedupe_rs.Dedupe(names(), num_cores=1, in_memory=True)
    deduper.prepare_training(data)
    deduper.mark_pairs({"match": [(data["a"], data["b"])], "distinct": [(data["a"], data["c"])]})
    deduper.train(recall=0.8)
    clusters = deduper.partition(data, 0.5)
    for ids, _ in clusters:
        assert all(isinstance(i, str) for i in ids)


def test_value_types_roundtrip():
    # Sets, ints, floats and None must survive a training round trip.
    data = {
        0: {"tags": ["red", "blue"], "price": 10, "loc": (42.5, -5.0), "note": None},
        1: {"tags": ["red", "green"], "price": 10, "loc": (42.5, -5.0), "note": None},
    }
    variables = [
        V.Set("tags"),
        V.Price("price"),
        V.LatLong("loc"),
        V.Exact("note"),
    ]
    deduper = dedupe_rs.Dedupe(variables, num_cores=1, in_memory=True)
    deduper.prepare_training(data)
    deduper.mark_pairs({"match": [(data[0], data[1])], "distinct": []})
    buf = io.StringIO()
    deduper.write_training(buf)
    buf.seek(0)
    loaded = dedupe_rs.read_training(buf)
    assert loaded["match"][0][0]["tags"] == ["red", "blue"]


def test_categorical_interaction_data_model():
    variables = [
        V.Categorical("type", ["a", "b"], name="type"),
        V.Interaction("type", "name"),
        V.Exact("name", name="name"),
    ]
    deduper = dedupe_rs.Dedupe(variables, num_cores=1, in_memory=True)
    assert deduper.data_model_len == 5
