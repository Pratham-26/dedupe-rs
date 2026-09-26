"""Variable definitions for :mod:`dedupe_rs`.

These mirror ``dedupe.variables`` from the Python library.  Each class is a
lightweight value object exposing a ``to_config()`` dict that the Rust
extension turns into a field type, its blocking predicates and its comparator.
"""

from __future__ import annotations

from dataclasses import dataclass, field as _field
from typing import Any, Callable, Iterable, Optional, Sequence

__all__ = [
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
]


@dataclass
class String:
    """A fuzzy string field (affine-gap distance by default)."""

    field: str
    name: Optional[str] = None
    crf: bool = False
    has_missing: bool = False

    def to_config(self) -> dict:
        return {
            "kind": "String",
            "field": self.field,
            "name": self.name,
            "crf": self.crf,
            "has_missing": self.has_missing,
        }


@dataclass
class ShortString(String):
    """A short string field (e.g. a name) with n-gram predicates only."""

    def to_config(self) -> dict:
        cfg = super().to_config()
        cfg["kind"] = "ShortString"
        return cfg


@dataclass
class Text:
    """A long free-text field compared with cosine similarity."""

    field: str
    corpus: Optional[Iterable[str]] = None
    name: Optional[str] = None
    has_missing: bool = False

    def to_config(self) -> dict:
        return {
            "kind": "Text",
            "field": self.field,
            "name": self.name,
            "has_missing": self.has_missing,
            "corpus": list(self.corpus) if self.corpus is not None else [],
        }


@dataclass
class Exact:
    """An exact-match field."""

    field: str
    name: Optional[str] = None
    has_missing: bool = False

    def to_config(self) -> dict:
        return {
            "kind": "Exact",
            "field": self.field,
            "name": self.name,
            "has_missing": self.has_missing,
        }


@dataclass
class Categorical:
    """A categorical field expanded into dummy indicator columns."""

    field: str
    categories: Sequence[str]
    name: Optional[str] = None
    has_missing: bool = False

    def to_config(self) -> dict:
        return {
            "kind": "Categorical",
            "field": self.field,
            "name": self.name,
            "has_missing": self.has_missing,
            "categories": list(self.categories),
        }


@dataclass
class Exists:
    """Whether a field is present."""

    field: str
    name: Optional[str] = None
    has_missing: bool = False

    def to_config(self) -> dict:
        return {
            "kind": "Exists",
            "field": self.field,
            "name": self.name,
            "has_missing": self.has_missing,
        }


@dataclass
class Price:
    """A numeric price compared on a log scale."""

    field: str
    name: Optional[str] = None
    has_missing: bool = False

    def to_config(self) -> dict:
        return {
            "kind": "Price",
            "field": self.field,
            "name": self.name,
            "has_missing": self.has_missing,
        }


@dataclass
class LatLong:
    """A ``(latitude, longitude)`` pair compared by haversine distance."""

    field: str
    name: Optional[str] = None
    has_missing: bool = False

    def to_config(self) -> dict:
        return {
            "kind": "LatLong",
            "field": self.field,
            "name": self.name,
            "has_missing": self.has_missing,
        }


@dataclass
class Set:
    """A set-valued field compared with cosine similarity."""

    field: str
    corpus: Optional[Iterable[Sequence[str]]] = None
    name: Optional[str] = None
    has_missing: bool = False

    def to_config(self) -> dict:
        return {
            "kind": "Set",
            "field": self.field,
            "name": self.name,
            "has_missing": self.has_missing,
            "corpus": [list(doc) for doc in self.corpus] if self.corpus is not None else [],
        }


class Interaction:
    """A derived interaction of two or more other variables."""

    def __init__(self, *interactions: str, name: Optional[str] = None, has_missing: bool = False):
        self.interactions = list(interactions)
        self.name = name
        self.has_missing = has_missing

    def to_config(self) -> dict:
        return {
            "kind": "Interaction",
            "interactions": self.interactions,
            "name": self.name,
            "has_missing": self.has_missing,
        }

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Interaction({', '.join(map(repr, self.interactions))})"


@dataclass
class Custom:
    """A user-defined field with a Python comparator ``f(value_a, value_b) -> float``."""

    field: str
    comparator: Callable[[Any, Any], float]
    name: Optional[str] = None
    has_missing: bool = False

    def to_config(self) -> dict:
        label = self.name or getattr(self.comparator, "__name__", "custom")
        return {
            "kind": "Custom",
            "field": self.field,
            "name": self.name,
            "has_missing": self.has_missing,
            "comparator": self.comparator,
            "comparator_label": label,
        }
