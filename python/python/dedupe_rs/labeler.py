"""Interactive labelling, mirroring ``dedupe.console_label``."""

from __future__ import annotations

import sys
from typing import Any, Dict, List, Tuple

__all__ = ["console_label"]


def _print(*args: Any) -> None:
    print(*args, file=sys.stderr)


def _mark_pair(deduper: Any, labeled_pair: Tuple[Any, str]) -> None:
    record_pair, label = labeled_pair
    examples: Dict[str, List[Any]] = {"distinct": [], "match": []}
    if label == "unsure":
        # See https://github.com/dedupeio/dedupe/issues/984 for reasoning
        examples["match"].append(record_pair)
        examples["distinct"].append(record_pair)
    else:
        examples[label].append(record_pair)
    deduper.mark_pairs(examples)


def console_label(deduper: Any) -> None:
    """Train a matcher (Dedupe, RecordLink or Gazetteer) from the command line.

    Example::

        deduper = dedupe_rs.Dedupe(variables)
        deduper.prepare_training(data)
        dedupe_rs.console_label(deduper)
    """
    finished = False
    use_previous = False
    fields = list(deduper.field_names)

    buffer_len = 1  # Max number of previous operations
    unlabeled: List[Any] = []
    labeled: List[Tuple[Any, str]] = []

    training = deduper.training_pairs
    n_match = len(training["match"])
    n_distinct = len(training["distinct"])

    while not finished:
        if use_previous:
            record_pair, label = labeled.pop(0)
            if label == "match":
                n_match -= 1
            elif label == "distinct":
                n_distinct -= 1
            use_previous = False
        else:
            try:
                if not unlabeled:
                    unlabeled = list(deduper.uncertain_pairs())
                record_pair = unlabeled.pop()
            except IndexError:
                break

        for record in record_pair:
            for field in fields:
                _print(f"{field} : {record[field]}")
            _print()
        _print(f"{n_match}/10 positive, {n_distinct}/10 negative")
        _print("Do these records refer to the same thing?")

        valid_response = False
        user_input = ""
        while not valid_response:
            if labeled:
                _print("(y)es / (n)o / (u)nsure / (f)inished / (p)revious")
                valid_responses = {"y", "n", "u", "f", "p"}
            else:
                _print("(y)es / (n)o / (u)nsure / (f)inished")
                valid_responses = {"y", "n", "u", "f"}
            user_input = input()
            if user_input in valid_responses:
                valid_response = True

        if user_input == "y":
            labeled.insert(0, (record_pair, "match"))
            n_match += 1
        elif user_input == "n":
            labeled.insert(0, (record_pair, "distinct"))
            n_distinct += 1
        elif user_input == "u":
            labeled.insert(0, (record_pair, "unsure"))
        elif user_input == "f":
            _print("Finished labeling")
            finished = True
        elif user_input == "p":
            use_previous = True
            unlabeled.append(record_pair)

        while len(labeled) > buffer_len:
            _mark_pair(deduper, labeled.pop())

    for labeled_pair in labeled:
        _mark_pair(deduper, labeled_pair)
