#!/usr/bin/env python
"""Generate golden parity vectors from the Python reference implementation.

Writes JSON files consumed by the Rust integration tests under
``dedupe-rs/tests/golden``.  Run from the ``parity`` directory::

    python gen_vectors.py
"""
import json
import os
import random
import string
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "dedupe-rs", "tests", "golden")
os.makedirs(OUT, exist_ok=True)

sys.path.insert(0, os.path.join(HERE, "..", "dedupe"))


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
    raise TypeError(type(v))


def sanitize(v):
    """JSON has no NaN; represent it as null."""
    if isinstance(v, float):
        return None if v != v else v
    if isinstance(v, dict):
        return {k: sanitize(x) for k, x in v.items()}
    if isinstance(v, list):
        return [sanitize(x) for x in v]
    return v


def rand_str(n):
    alphabet = string.ascii_letters + string.digits + " -.,/'&"
    return "".join(random.choice(alphabet) for _ in range(n))


def gen_predicate_functions():
    from dedupe import predicate_functions as fn

    string_funcs = [
        "wholeFieldPredicate",
        "tokenFieldPredicate",
        "firstTokenPredicate",
        "firstTwoTokensPredicate",
        "commonIntegerPredicate",
        "alphaNumericPredicate",
        "nearIntegersPredicate",
        "hundredIntegerPredicate",
        "hundredIntegersOddPredicate",
        "firstIntegerPredicate",
        "commonTwoTokens",
        "commonThreeTokens",
        "fingerprint",
        "oneGramFingerprint",
        "twoGramFingerprint",
        "commonFourGram",
        "commonSixGram",
        "sameThreeCharStartPredicate",
        "sameFiveCharStartPredicate",
        "sameSevenCharStartPredicate",
        "suffixArray",
        "sortedAcronym",
        "doubleMetaphone",
        "metaphoneToken",
    ]
    numeric_funcs = ["orderOfMagnitude", "roundTo1"]
    tuple_funcs = ["latLongGridPredicate"]
    set_funcs = [
        "commonSetElementPredicate",
        "commonTwoElementsPredicate",
        "commonThreeElementsPredicate",
        "lastSetElementPredicate",
        "firstSetElementPredicate",
        "magnitudeOfCardinality",
    ]

    cases = []
    rnd = random.Random(20240926)

    fixed_strings = [
        "donald", "go-of,y  ", " cip ciop ", "do\nal d", "don ald", "g00fy  ",
        " c1p   c10p ", "don4ld", "donald 1992", "d on 456 ld", " c1p   c10p  c100p",
        "don 456 ld ", " g00fy  ", " c11p   c10p ", "mississippi", "don4ld",
        "c1p\nc10p ", "don 4l d", "donald 19 92", "g 00f y  ", " c1p   c10p ",
        "9301 S. State St. ", "STT", "i", "time sandwich", "sandwich time",
        "7", "a", "1", "foo bar", "foo", "123 16th st", "oneword", "123 456/",
        "foo", "1foo", "f1oo", "", " ", "12", "1.5", "3.14159", "-42",
    ]
    strings = list(fixed_strings)
    for _ in range(2000):
        strings.append(rand_str(rnd.randint(0, 20)))
    for s in strings:
        for name in string_funcs:
            try:
                expected = sorted(getattr(fn, name)(s))
            except Exception as e:  # pragma: no cover
                expected = {"error": type(e).__name__}
            cases.append({"fn": name, "input": enc(s), "expected": expected})

    numbers = [0, 1, 2, 9, 10, 42, 98, 99, 100, 1230, -2, 22315, -22315,
               10 ** 6, 3.14, 0.001, 999.99, 10500, 25000, 15000]
    for _ in range(500):
        numbers.append(rnd.randint(-(10 ** 9), 10 ** 9))
        numbers.append(round(rnd.uniform(-1e6, 1e6), rnd.randint(0, 4)))
    numbers = [n for n in numbers if not isinstance(n, float) or n == n]
    for n in numbers:
        for name in numeric_funcs:
            try:
                expected = sorted(getattr(fn, name)(n))
            except ValueError:
                continue
            cases.append({"fn": name, "input": enc(n), "expected": expected})

    tuples = [(1.11, 2.22), (1.11, 2.27), (1.18, 2.22), (1.19, 2.29),
              (42.535, -5.012), (0, 0), (0.0, 0.0), (1, 2), (-90.0, 180.0)]
    for _ in range(300):
        tuples.append((round(rnd.uniform(-90, 90), 4), round(rnd.uniform(-180, 180), 4)))
    for t in tuples:
        for name in tuple_funcs:
            expected = sorted(getattr(fn, name)(t))
            cases.append({"fn": name, "input": enc(t), "expected": expected})

    sets = [{"red", "blue", "green"}, set(), {"i"}, {"donald"},
            (1, 2, 3), (1,), ("a", "b", "c"), ("c", "b", "a"), (1, 2, 3, 4, 5)]
    for _ in range(300):
        sets.append(tuple(rnd.sample(range(1000), rnd.randint(0, 6))))
    for s in sets:
        for name in set_funcs:
            try:
                expected = sorted(getattr(fn, name)(s))
            except (ValueError, TypeError):
                continue
            cases.append({"fn": name, "input": enc(s), "expected": expected})

    with open(os.path.join(OUT, "predicate_functions.json"), "w") as f:
        json.dump(sanitize(cases), f)
    print("predicate_functions:", len(cases), "cases")


def gen_comparators():
    from affinegap import normalizedAffineGapDistance as affine
    from categorical import CategoricalComparator
    from simplecosine.cosine import CosineTextSimilarity, CosineSetSimilarity
    from haversine import haversine
    from dedupe.variables.price import PriceType
    from highered import CRFEditDistance

    rnd = random.Random(4242)
    cases = []

    # affine gap
    pairs = [("", "abc"), ("a", "a"), ("abc", "abd"), ("kitten", "sitting"),
             ("mary crane", "mary crane center"), ("", ""), ("ab", "ba")]
    for _ in range(1500):
        pairs.append((rand_str(rnd.randint(0, 12)), rand_str(rnd.randint(0, 12))))
    for a, b in pairs:
        if a == "" and b == "":
            continue
        cases.append({"cmp": "affinegap", "a": enc(a), "b": enc(b),
                      "expected": [float(affine(a, b))]})

    # crf edit distance
    crf = CRFEditDistance()
    pairs = [("kitten", "sitting"), ("abc", "abc"), ("abc", "abd"), ("a", "b")]
    for _ in range(400):
        pairs.append((rand_str(rnd.randint(1, 8)), rand_str(rnd.randint(1, 8))))
    for a, b in pairs:
        cases.append({"cmp": "crf", "a": enc(a), "b": enc(b),
                      "expected": [float(crf(a, b))]})

    # categorical
    cats = ["a", "b", "c"]
    cc = CategoricalComparator(cats)
    for a in cats:
        for b in cats:
            cases.append({"cmp": "categorical", "a": enc(a), "b": enc(b),
                          "expected": [float(x) for x in cc(a, b)]})

    # price
    for a, b in [(1, 10), (10, 1), (0, 5), (-1, 5), (3.5, 7.25)]:
        cases.append({"cmp": "price", "a": enc(a), "b": enc(b),
                      "expected": [float(PriceType.comparator(a, b))]})

    # latlong
    for a, b in [((42.535, -5.012), (42.6, -5.1)), ((0, 0), (0, 0)),
                 ((1.0, 2.0), (3.0, 4.0))]:
        import math
        expected = math.sqrt(haversine(a, b))
        cases.append({"cmp": "latlong", "a": enc(tuple(float(x) for x in a)),
                      "b": enc(tuple(float(x) for x in b)), "expected": [expected]})

    # cosine text
    corpus = [rand_str(rnd.randint(0, 10)) for _ in range(60)]
    ct = CosineTextSimilarity(corpus)
    items = corpus[:20] + ["", "hello world", "hello there"]
    for _ in range(400):
        a = rnd.choice(items)
        b = rnd.choice(items)
        expected = float(ct(a, b))
        cases.append({"cmp": "cosine_text", "a": enc(a), "b": enc(b),
                      "expected": [expected], "corpus": [enc(c) for c in corpus]})

    # cosine set
    set_corpus = [tuple(rnd.sample(range(30), rnd.randint(0, 6))) for _ in range(40)]
    cs = CosineSetSimilarity(set_corpus)
    set_items = set_corpus[:12] + [(), (0, 1, 2)]
    for _ in range(300):
        a = rnd.choice(set_items)
        b = rnd.choice(set_items)
        expected = float(cs(a, b))
        cases.append({"cmp": "cosine_set", "a": enc(tuple(a)), "b": enc(tuple(b)),
                      "expected": [expected],
                      "corpus": [enc(tuple(c)) for c in set_corpus]})

    # exact
    for a, b in [("a", "a"), ("a", "b"), (1, 1), (1, 2)]:
        from dedupe.variables.exact import ExactType
        cases.append({"cmp": "exact", "a": enc(a), "b": enc(b),
                      "expected": [float(ExactType.comparator(a, b))]})

    with open(os.path.join(OUT, "comparators.json"), "w") as f:
        json.dump(sanitize(cases), f)
    print("comparators:", len(cases), "cases")


if __name__ == "__main__":
    gen_predicate_functions()
    gen_comparators()
