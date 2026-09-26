#!/usr/bin/env python
"""Profiling harness: generate a shared dataset + trained model, then time the
Python pipeline.  Run from the `parity` directory:

    python profile.py gen      # write dedupe-rs/tests/golden/profile.json
    python profile.py bench    # time the Python pipeline
"""
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "dedupe-rs", "tests", "golden")
sys.path.insert(0, os.path.join(HERE, "..", "dedupe"))

import dedupe  # noqa: E402
import dedupe.clustering as clustering  # noqa: E402
import dedupe.core as core  # noqa: E402
import dedupe.predicates as P  # noqa: E402
import numpy  # noqa: E402

WANTED_PREDICATES = {
    "SimplePredicate: (wholeFieldPredicate, name)",
    "SimplePredicate: (firstTokenPredicate, name)",
    "SimplePredicate: (commonTwoTokens, name)",
    "SimplePredicate: (sortedAcronym, name)",
    "SimplePredicate: (sameThreeCharStartPredicate, name)",
    "SimplePredicate: (commonFourGram, name)",
    "SimplePredicate: (firstTwoTokensPredicate, name)",
    "TfidfNGramCanopyPredicate: (0.4, name)",
    "TfidfNGramSearchPredicate: (0.4, name)",
    "TfidfTextCanopyPredicate: (0.4, name)",
    "TfidfTextSearchPredicate: (0.4, name)",
    "LevenshteinCanopyPredicate: (1, name)",
    "LevenshteinSearchPredicate: (1, name)",
}


def select_predicates(data_model):
    return [p for p in data_model.predicates if repr(p) in WANTED_PREDICATES]


SYL1 = ["ba", "be", "bi", "bo", "ca", "ce", "ci", "da", "de", "fa", "fi", "ga",
        "ha", "ja", "ka", "la", "ma", "na", "pa", "ra", "sa", "ta", "va", "wa"]
SYL2 = ["ker", "son", "ley", "ton", "man", "ford", "berg", "stein", "well", "croft",
        "shaw", "ridge", "dale", "wick", "more", "hart"]
SYL3 = ["an", "el", "in", "or", "ur", "ad", "ed", "id", "od", "al", "il", "ol"]
SYL4 = ["derson", "son", "sky", "ova", "etti", "ez", "man", "and", "er", "ing"]


def make_name(rnd):
    return (rnd.choice(SYL1) + rnd.choice(SYL2) + " " +
            rnd.choice(SYL1) + rnd.choice(SYL3) + rnd.choice(SYL4))


def mutate(name, rnd):
    r = rnd.random()
    if r < 0.2:
        return name.upper()
    if r < 0.35:
        return name.replace(" ", "")
    if r < 0.5 and len(name) > 3:
        i = rnd.randrange(len(name))
        return name[:i] + name[i + 1:]
    if r < 0.65:
        return name + " " + rnd.choice(["jr", "sr", "ii", "iii"])
    if r < 0.8:
        return name.replace("e", "i")
    return name


def gen_dataset(n=8000, seed=2024):
    rnd = random.Random(seed)
    n_entities = n * 3 // 4
    data = {}
    rid = 0
    for _ in range(n_entities):
        name = make_name(rnd)
        age = str(rnd.randint(18, 90))
        copies = rnd.choice([1, 1, 1, 2, 3])
        for c in range(copies):
            if rid >= n:
                break
            data[rid] = {"name": name if c == 0 else mutate(name, rnd), "age": age}
            rid += 1
        if rid >= n:
            break
    return data


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
        return {"Index": {"type_name": p.type, "threshold": float(p.threshold),
                          "threshold_repr": str(p.threshold), "field": p.field}}
    raise TypeError(type(p))


def write_profile(data):
    variables = dedupe.Dedupe(
        [dedupe.variables.String("name"), dedupe.variables.String("age")],
        num_cores=1, in_memory=True,
    )
    predicates = select_predicates(variables.data_model)
    n_features = len(variables.data_model)
    rnd = random.Random(7)
    coef = [rnd.uniform(-1, 1) for _ in range(n_features)]
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
        "classifier": {"coef": coef, "intercept": 0.5, "c": 1.0},
    }
    json.dump(
        {"settings": settings, "data": {str(k): v for k, v in data.items()},
         "n_features": n_features},
        open(os.path.join(OUT, "profile.json"), "w"),
    )
    print(f"profile.json: {len(data)} records, {len(predicates)} predicates, {n_features} features")


def peak_rss_mb():
    try:
        import psutil
        m = psutil.Process().memory_info()
        peak = getattr(m, "peak_wset", None)
        if peak is None:
            peak = getattr(m, "rss", 0)
        return peak / (1024 * 1024)
    except Exception:
        return 0.0


def bench(data, cores=1):
    from dedupe.blocking import Fingerprinter
    import sklearn.linear_model

    deduper = dedupe.Dedupe(
        [dedupe.variables.String("name"), dedupe.variables.String("age")],
        num_cores=1, in_memory=True,
    )
    predicates = select_predicates(deduper.data_model)
    fingerprinter = Fingerprinter(predicates)

    t0 = time.perf_counter()
    fingerprinter.index_all(data)
    t1 = time.perf_counter()
    rss1 = peak_rss_mb()
    blocks = {}
    for block_key, record_id in fingerprinter(data.items()):
        blocks.setdefault(block_key, []).append(record_id)
    pairs = set()
    for ids in blocks.values():
        ids.sort()
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                pairs.add((ids[i], ids[j]))
    t2 = time.perf_counter()
    rss2 = peak_rss_mb()
    pair_list = [((a, data[a]), (b, data[b])) for a, b in pairs]
    n = len(pair_list)

    n_features = len(deduper.data_model)
    clf = sklearn.linear_model.LogisticRegression()
    xs = numpy.random.RandomState(0).rand(50, n_features)
    ys = numpy.array([0, 1] * 25)
    clf.fit(xs, ys)
    t3 = time.perf_counter()
    rss3pre = peak_rss_mb()
    scores = core.scoreDuplicates(iter(pair_list), deduper.data_model.distances, clf, cores)
    t4 = time.perf_counter()
    rss3 = peak_rss_mb()
    clusters = list(clustering.cluster(scores, 0.5))
    t5 = time.perf_counter()
    rss4 = peak_rss_mb()

    result = {
        "impl": "python",
        "cores": cores,
        "records": len(data),
        "pairs": n,
        "scored": len(scores),
        "clusters": len(clusters),
        "index_all": t1 - t0,
        "blocking": t2 - t1,
        "score": t4 - t3,
        "cluster": t5 - t4,
        "total": t5 - t0,
        "peak_mb": rss4,
        "peak_index_mb": rss1,
        "peak_pairs_mb": rss2,
        "peak_score_mb": rss3,
        "peak_cluster_mb": rss4,
        "_rss3pre": rss3pre,
    }
    return result


def print_result(r):
    print(f"Python pipeline ({r['records']} records, {r['cores']} core(s)):")
    print(f"  index_all      : {r['index_all']:8.3f}s")
    print(f"  blocking+pairs : {r['blocking']:8.3f}s ({r['pairs']} pairs)")
    print(f"  score          : {r['score']:8.3f}s ({r['scored']} scored)")
    print(f"  cluster        : {r['cluster']:8.3f}s ({r['clusters']} clusters)")
    print(f"  TOTAL          : {r['total']:8.3f}s")
    print(f"  peak RSS       : {r['peak_mb']:8.1f} MB")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "bench"
    cores = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    n = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3].isdigit() else 8000
    json_mode = "--json" in sys.argv
    data = gen_dataset(n)
    if mode == "gen":
        write_profile(data)
    else:
        result = bench(data, cores)
        if json_mode:
            print(json.dumps(result))
        else:
            print_result(result)
