"""Supplier name normalization helpers.

This module provides utilities to guess the most likely normalized supplier
name by combining lightweight string cleaning, optional rule-based
replacement, and fuzzy matching against a known list of supplier names.
"""

from __future__ import annotations

import csv
from difflib import SequenceMatcher
from typing import Iterable, List, Optional, Sequence

from utils_data_cleansing import apply_rule, rough_clean


def load_supplier_names(path: str) -> List[str]:
    """Load supplier names from a semicolon-separated CSV file.

    The repository's ``CONSIGNEE_NAME.csv`` file uses a semicolon delimiter
    where the first column is the normalized supplier name and the second
    column contains a count. This helper returns the list of names from the
    first column while ignoring empty rows.
    """

    supplier_names: List[str] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=";")
        # Skip header if present
        header = next(reader, None)
        if header and len(header) == 1 and header[0].lower() == "count":
            # file starts with an empty first column in the header
            pass
        for row in reader:
            if not row:
                continue
            supplier_name = row[0].strip()
            if supplier_name:
                supplier_names.append(supplier_name)
    return supplier_names


def guess_supplier_name(
    name: Optional[str],
    known_suppliers: Sequence[str],
    rules: Optional[Sequence[dict]] = None,
    min_score: float = 0.7,
) -> Optional[str]:
    """Return the most likely normalized supplier name.

    Args:
        name: Raw supplier name to normalize.
        known_suppliers: Iterable of canonical supplier names to match
            against.
        rules: Optional list of rule dictionaries compatible with
            :func:`utils_data_cleansing.apply_rule`. When provided, the raw
            name is first normalized using these rules.
        min_score: Minimum similarity score (0-1) required to accept a fuzzy
            match. Defaults to ``0.7``.

    Returns:
        The best matching supplier name from ``known_suppliers`` when the
        similarity score exceeds ``min_score``. If no match is found, the
        cleaned version of ``name`` is returned. ``None`` is returned when
        ``name`` is ``None``.
    """

    if name is None:
        return None

    known_list: List[str] = list(known_suppliers)
    if not known_list:
        return None

    if rules:
        cleaned = apply_rule(name, rules)
    else:
        cleaned = rough_clean(name).title()

    # Exact match after cleaning
    if cleaned in known_list:
        return cleaned

    # Lightweight fuzzy matching based on sequence similarity avoids the heavy
    # scikit-learn dependency while still providing stable results for short
    # supplier names.
    best_score = 0.0
    best_match: Optional[str] = None
    for candidate in known_list:
        score = SequenceMatcher(None, cleaned.lower(), candidate.lower()).ratio()
        if score > best_score:
            best_score = score
            best_match = candidate

    return best_match if best_match and best_score >= min_score else cleaned
