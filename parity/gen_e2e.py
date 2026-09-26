#!/usr/bin/env python
"""End-to-end parity: train a Dedupe in Python, serialize the trained state,
and record its partition/join output for the Rust port to reproduce."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "dedupe-rs", "tests", "golden")
sys.path.insert(0, os.path.join(HERE, "..", "dedupe"))

import dedupe  # noqa: E402
import dedupe.predicates as P  # noqa: E402


def enc_record(r):
    return {k: v for k, v in r.items()}


def enc_pred(p):
    if isinstance(p, P.CompoundPredicate):
        return {"Compound": {"parts": [enc_pred(x) for x in p]}}
    if isinstance(p, P.ExistsPredicate):
        return {"Exists": {"field": p.field}}
    if isinstance(p, P.StringPredicate):
        return {"Simple": {"func": p.func.__name__, "field": p.field, "string_mode": True}}
    if isinstance(p, P.SimplePredicate):
        return {"Simple": {"func": p.func.__name__, "field": p.field, "string_mode": False}}
    if hasattr(p, "index"):
        return {
            "Index": {
                "type_name": p.type,
                "threshold": float(p.threshold),
                "threshold_repr": str(p.threshold),
                "field": p.field,
            }
        }
    raise TypeError(type(p))


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

def main():
    variables = [dedupe.variables.String("name"), dedupe.variables.String("age")]
    deduper = dedupe.Dedupe(variables, num_cores=1, in_memory=True)
    deduper.prepare_training(DATA)
    deduper.mark_pairs(TRAIN)
    deduper.train(recall=0.8, index_predicates=True)

    predicates = list(deduper.predicates)
    clf = deduper.classifier.best_estimator_
    coef = [float(x) for x in clf.coef_[0]]
    intercept = float(clf.intercept_[0])
    c = float(deduper.classifier.best_params_["C"])

    partitions = {}
    for threshold in (0.3, 0.5, 0.7):
        clusters = deduper.partition(DATA, threshold=threshold)
        partitions[str(threshold)] = [
            {
                "ids": [int(i) for i in ids],
                "scores": [round(float(s), 6) for s in scores],
            }
            for ids, scores in clusters
        ]

    settings = {
        "version": 1,
        "variables": [
            {"kind": "String", "field": "name", "name": None, "has_missing": False,
             "crf": False, "categories": [], "corpus": [], "interactions": [],
             "comparator_label": None},
            {"kind": "String", "field": "age", "name": None, "has_missing": False,
             "crf": False, "categories": [], "corpus": [], "interactions": [],
             "comparator_label": None},
        ],
        "predicates": [enc_pred(p) for p in predicates],
        "classifier": {"coef": coef, "intercept": intercept, "c": c},
    }

    json.dump(
        {
            "settings": settings,
            "data": {str(k): v for k, v in DATA.items()},
            "partitions": partitions,
            "n_features": len(coef),
        },
        open(os.path.join(OUT, "e2e.json"), "w"),
    )
    print("e2e:", len(predicates), "predicates,", len(coef), "features")


if __name__ == "__main__":
    main()
