#!/usr/bin/env python
"""Generate golden vectors for the data model, predicates/indexes and TF-IDF."""
import json
import math
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "dedupe-rs", "tests", "golden")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, os.path.join(HERE, "..", "dedupe"))

import dedupe.variables as V  # noqa: E402
from dedupe import blocking, predicates  # noqa: E402
from dedupe.datamodel import DataModel  # noqa: E402
from dedupe.tfidf import TfIdfIndex  # noqa: E402


def enc(v):
    if v is None:
        return {"t": "n"}
    if isinstance(v, bool):
        return {"t": "b", "v": v}
    if isinstance(v, str):
        return {"t": "s", "v": v}
    if isinstance(v, int):
        return {"t": "i", "v": v}
    if isinstance(v, float):
        return {"t": "f", "v": v}
    if isinstance(v, tuple):
        return {"t": "tup", "v": [enc(x) for x in v]}
    if isinstance(v, list):
        return {"t": "l", "v": [enc(x) for x in v]}
    if isinstance(v, (set, frozenset)):
        return {"t": "set", "v": [enc(x) for x in v]}
    if isinstance(v, dict):
        return {"t": "d", "v": {k: enc(x) for k, x in v.items()}}
    raise TypeError(type(v))


def sanitize(v):
    if isinstance(v, float):
        return None if (v != v or math.isinf(v)) else v
    if isinstance(v, dict):
        return {k: sanitize(x) for k, x in v.items()}
    if isinstance(v, list):
        return [sanitize(x) for x in v]
    return v


def build_vars(defs):
    out = []
    for d in defs:
        t = d["type"]
        kw = {}
        if "name" in d and d["name"] is not None:
            kw["name"] = d["name"]
        if d.get("has_missing"):
            kw["has_missing"] = True
        if t == "String":
            out.append(V.String(d["field"], crf=d.get("crf", False), **kw))
        elif t == "ShortString":
            out.append(V.ShortString(d["field"], **kw))
        elif t == "Text":
            out.append(V.Text(d["field"], corpus=d.get("corpus", []), **kw))
        elif t == "Exact":
            out.append(V.Exact(d["field"], **kw))
        elif t == "Categorical":
            out.append(V.Categorical(d["field"], categories=d["categories"], **kw))
        elif t == "Exists":
            out.append(V.Exists(d["field"], **kw))
        elif t == "Price":
            out.append(V.Price(d["field"], **kw))
        elif t == "LatLong":
            out.append(V.LatLong(d["field"], **kw))
        elif t == "Set":
            out.append(V.Set(d["field"], corpus=d.get("corpus", []), **kw))
        elif t == "Interaction":
            out.append(V.Interaction(*d["interactions"]))
        else:
            raise ValueError(t)
    return out


CONFIGS = [
    [{"type": "Exact", "field": "name"}],
    [{"type": "String", "field": "name"}],
    [{"type": "ShortString", "field": "name"}],
    [{"type": "String", "field": "name", "has_missing": True}],
    [{"type": "Categorical", "field": "type", "categories": ["a", "b", "c"]}],
    [{"type": "Exact", "field": "id"}, {"type": "Exists", "field": "flag"}],
    [{"type": "Price", "field": "amount"}],
    [{"type": "LatLong", "field": "loc"}],
    [{"type": "Exact", "field": "id"}, {"type": "Set", "field": "tags"}],
    [{"type": "Categorical", "field": "type", "categories": ["a", "b"], "name": "type"},
     {"type": "Interaction", "interactions": ["type", "name"]},
     {"type": "Exact", "field": "name", "name": "name"}],
    [{"type": "String", "field": "name", "has_missing": True},
     {"type": "String", "field": "age"},
     {"type": "Interaction", "interactions": ["(name: String)", "(age: String)"]}],
    [{"type": "Text", "field": "desc"}],
]


def gen_datamodel():
    rnd = random.Random(1234)
    out = []
    for defs in CONFIGS:
        variables = build_vars(defs)
        dm = DataModel(variables)
        # record pairs
        pairs = []
        for _ in range(40):
            r1, r2 = {}, {}
            for d in defs:
                t = d["type"]
                if t == "Interaction":
                    continue
                f = d["field"]
                if t == "Exact":
                    r1[f] = rnd.choice(["x", "y", "z"])
                    r2[f] = rnd.choice(["x", "y", "z"])
                elif t in ("String", "ShortString", "Text"):
                    r1[f] = rnd.choice(["bob", "bobby", "robert", None])
                    r2[f] = rnd.choice(["bob", "bobby", "robert", None])
                elif t == "Categorical":
                    r1[f] = rnd.choice(d["categories"])
                    r2[f] = rnd.choice(d["categories"])
                elif t == "Exists":
                    r1[f] = rnd.choice([None, "", 0, 1, "x"])
                    r2[f] = rnd.choice([None, "", 0, 1, "x"])
                elif t == "Price":
                    r1[f] = rnd.choice([1, 10, 100.5, None])
                    r2[f] = rnd.choice([1, 10, 100.5, None])
                elif t == "LatLong":
                    r1[f] = (round(rnd.uniform(-90, 90), 3), round(rnd.uniform(-180, 180), 3))
                    r2[f] = (round(rnd.uniform(-90, 90), 3), round(rnd.uniform(-180, 180), 3))
                elif t == "Set":
                    r1[f] = tuple(rnd.sample(["a", "b", "c", "d"], rnd.randint(0, 3)))
                    r2[f] = tuple(rnd.sample(["a", "b", "c", "d"], rnd.randint(0, 3)))
            # ensure interaction-referenced fields exist
            for d in defs:
                if d["type"] == "Interaction":
                    for ref in d["interactions"]:
                        pass
            pairs.append((r1, r2))

        # For interaction configs the referenced field names must exist; the
        # generator above already created them for field types.
        distances = dm.distances(pairs)
        meta = {
            "defs": defs,
            "derived_start": dm._derived_start,
            "missing_field_indices": list(dm._missing_field_indices),
            "interaction_indices": [list(x) for x in dm._interaction_indices],
            "len": len(dm),
            "predicates": sorted(repr(p) for p in dm.predicates),
            "field_variables": [
                {"field": fv.field, "name": fv.name, "len": len(fv)} for fv in dm.field_variables
            ],
        }
        out.append({
            "meta": meta,
            "pairs": [[enc(a), enc(b)] for a, b in pairs],
            "distances": [[float(x) for x in row] for row in distances],
        })
    json.dump(sanitize(out), open(os.path.join(OUT, "datamodel.json"), "w"))
    print("datamodel:", len(out), "configs")


DATA = {
    100: {"name": "Bob", "age": "50", "dataset": 0},
    105: {"name": "Charlie", "age": "75", "dataset": 1},
    110: {"name": "Meredith", "age": "40", "dataset": 1},
    115: {"name": "Sue", "age": "10", "dataset": 0},
    120: {"name": "Jimbo", "age": "21", "dataset": 0},
    125: {"name": "Jimbo", "age": "21", "dataset": 0},
    130: {"name": "Willy", "age": "35", "dataset": 0},
    135: {"name": "Willy", "age": "35", "dataset": 1},
    140: {"name": "Martha", "age": "19", "dataset": 1},
    145: {"name": "Kyle", "age": "27", "dataset": 0},
}


def gen_predicate_outputs():
    defs = [{"type": "String", "field": "name"}, {"type": "String", "field": "age"}]
    variables = build_vars(defs)
    dm = DataModel(variables)
    fp = blocking.Fingerprinter(dm.predicates)
    fp.index_all(DATA)

    preds = sorted(dm.predicates, key=repr)
    cases = []
    for p in preds:
        for target in (False, True):
            blocks = {}
            for rid in sorted(DATA):
                record = DATA[rid]
                try:
                    keys = p(record, target=target)
                except Exception as e:  # pragma: no cover
                    keys = {"__error__:" + type(e).__name__}
                for k in keys:
                    blocks.setdefault(k, set()).add(rid)
            groups = sorted([sorted(v) for v in blocks.values()])
            cases.append({
                "predicate": repr(p),
                "target": target,
                "expected_groups": groups,
            })
    json.dump(sanitize({
        "defs": defs,
        "data": {str(k): enc(v) for k, v in DATA.items()},
        "cases": cases,
    }), open(os.path.join(OUT, "blocking.json"), "w"))
    print("blocking:", len(cases), "cases")


def gen_tfidf():
    rnd = random.Random(77)
    docs = [tuple(rnd.sample(["alpha", "beta", "gamma", "delta", "epsilon", "zeta"], rnd.randint(1, 4)))
            for _ in range(60)]
    docs += [("AND", "OR", "EOF", "NOT"), ("And", "Or", "Eof", "Not"), (r"f\o",), ("f*",)]
    index = TfIdfIndex()
    for d in docs:
        index.index(d)
    index._index.initSearch()

    queries = docs[:20] + [("alpha",), ("alpha", "beta"), (), ("zeta", "epsilon"),
                           ("AND", "OR", "EOF", "NOT"), ("xxx",)]
    cases = []
    for q in queries:
        for threshold in (0.0, 0.2, 0.5, 0.8):
            results = index.search(q, threshold)
            cases.append({"query": [enc(x) for x in q], "threshold": threshold,
                          "expected": [int(r) for r in results]})
    json.dump(sanitize({"docs": [[enc(x) for x in d] for d in docs], "cases": cases}),
              open(os.path.join(OUT, "tfidf.json"), "w"))
    print("tfidf:", len(cases), "cases")



def gen_clustering():
    import numpy
    import dedupe.clustering as clustering

    def enc_pair(p, score):
        return {"a": enc(p[0]), "b": enc(p[1]), "score": float(score)}

    def run(pairs, threshold):
        arr = numpy.array(list(pairs), dtype=[("pairs", "i4", 2), ("score", "f4")])
        out = []
        for ids, scores in clustering.cluster(arr, threshold):
            out.append({"ids": [int(i) for i in ids], "scores": [float(s) for s in scores]})
        return out

    rnd = random.Random(5150)
    cases = []

    dupes = [
        ((1, 2), 0.86), ((1, 3), 0.72), ((1, 4), 0.2), ((1, 5), 0.6),
        ((2, 3), 0.86), ((2, 4), 0.2), ((2, 5), 0.72), ((3, 4), 0.3),
        ((3, 5), 0.5), ((4, 5), 0.72), ((10, 11), 0.9),
    ]
    for t in (0.0, 0.5, 1.0):
        cases.append({"pairs": [enc_pair(p, s) for p, s in dupes], "threshold": t,
                      "expected": run(dupes, t), "_raw": dupes})

    for _ in range(30):
        n = rnd.randint(2, 14)
        ids = rnd.sample(range(1, 40), n)
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                if rnd.random() < 0.5:
                    a, b = sorted((ids[i], ids[j]))
                    pairs.append(((a, b), round(rnd.uniform(0.0, 1.0), 3)))
        if not pairs:
            continue
        t = rnd.choice([0.0, 0.3, 0.5, 0.7, 0.9])
        cases.append({"pairs": [enc_pair(p, s) for p, s in pairs], "threshold": t,
                      "expected": run(pairs, t), "_raw": pairs})

    # connected components, with int and str ids
    G = [
        ((1, 2), 0.1), ((2, 3), 0.2), ((4, 5), 0.2), ((4, 6), 0.2),
        ((7, 9), 0.2), ((8, 9), 0.2), ((10, 11), 0.2), ((12, 13), 0.2),
        ((12, 14), 0.5), ((11, 12), 0.2),
    ]
    arr = numpy.array(G, dtype=[("pairs", "i4", 2), ("score", "f4")])
    comps = [
        sorted([tuple(int(x) for x in edge) for edge, _ in c])
        for c in clustering.connected_components(arr, 30000)
    ]
    comps = sorted(comps)

    # Per-component condensed matrices + scipy linkages for direct comparison.
    import scipy.cluster.hierarchy as sch
    linkage_cases = []
    all_case_pairs = [dupes] + [c["_raw"] for c in cases if "_raw" in c]
    for raw in all_case_pairs:
        arr = numpy.array(list(raw), dtype=[("pairs", "i4", 2), ("score", "f4")])
        for comp in clustering.connected_components(arr, 30000):
            i_to_id, cd, N = clustering.condensedDistance(comp)
            Z = sch.linkage(cd, method="centroid")
            linkage_cases.append({
                "ids": [int(i_to_id[i]) for i in range(N)],
                "condensed": [float(x) for x in cd],
                "n": N,
                "Z": [[float(v) for v in row] for row in Z],
            })

    json.dump(sanitize({
        "cases": cases,
        "components": {"pairs": [enc_pair(p, s) for p, s in G], "expected": comps},
        "linkage": linkage_cases,
    }), open(os.path.join(OUT, "clustering.json"), "w"))
    print("clustering:", len(cases), "cases,", len(linkage_cases), "linkage components")

if __name__ == "__main__":
    gen_datamodel()
    gen_predicate_outputs()
    gen_tfidf()
    gen_clustering()
