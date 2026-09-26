"""A Rust implementation of `dedupe <https://github.com/dedupeio/dedupe>`_.

Fuzzy matching, de-duplication and entity resolution over structured data,
with the same workflow as the Python library::

    import dedupe_rs

    variables = [
        dedupe_rs.variables.String("name"),
        dedupe_rs.variables.String("age"),
    ]
    deduper = dedupe_rs.Dedupe(variables)
    deduper.prepare_training(data)

    while True:
        try:
            pair = deduper.uncertain_pairs()
        except IndexError:
            break
        deduper.mark_pairs({"match": [pair], "distinct": []})

    deduper.train()
    clusters = deduper.partition(data, threshold=0.5)

The heavy lifting (blocking, indices, distance computation, scoring, training
and clustering) is implemented in Rust.
"""

from __future__ import annotations

from ._native import (
    Dedupe,
    Gazetteer,
    RecordLink,
    StaticDedupe,
    StaticGazetteer,
    StaticRecordLink,
    canonicalize,
    read_training,
    settings_to_dict,
    training_data_dedupe,
    training_data_link,
    version,
    write_training,
)
from . import variables
from .labeler import console_label
from .variables import (
    Categorical,
    Custom,
    Exact,
    Exists,
    Interaction,
    LatLong,
    Price,
    Set,
    ShortString,
    String,
    Text,
)

__version__ = version()

__all__ = [
    "Dedupe",
    "RecordLink",
    "Gazetteer",
    "StaticDedupe",
    "StaticRecordLink",
    "StaticGazetteer",
    "variables",
    "String",
    "ShortString",
    "Text",
    "Exact",
    "Categorical",
    "Exists",
    "Price",
    "LatLong",
    "Set",
    "Interaction",
    "Custom",
    "console_label",
    "canonicalize",
    "write_training",
    "read_training",
    "training_data_link",
    "training_data_dedupe",
    "settings_to_dict",
    "__version__",
]
