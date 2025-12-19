"""Supplier name normalization helpers.

This module provides utilities to guess the most likely normalized supplier
name by combining lightweight string cleaning, optional rule-based
replacement, and fuzzy matching against a known list of supplier names.
"""

from __future__ import annotations

import csv
from typing import List, Optional, Sequence

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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

    The fuzzy-matching step is powered by a character-level TF-IDF
    vectorizer from scikit-learn with cosine similarity, allowing robust
    matching even when names include extra tokens or minor spelling issues.

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

    target = rough_clean(cleaned)
    normalized_candidates = [rough_clean(candidate) for candidate in known_list]

    if not target.strip() or not any(normalized_candidates):
        return cleaned

    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
    tfidf_matrix = vectorizer.fit_transform([target] + normalized_candidates)

    similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).ravel()
    best_index = int(similarities.argmax())
    best_score = float(similarities[best_index]) if similarities.size else 0.0

    return known_list[best_index] if best_score >= min_score else cleaned


def guess_supplier_name_from_priority(
    names: Sequence[Optional[str]],
    known_suppliers: Sequence[str],
    rules: Optional[Sequence[dict]] = None,
    min_score: float = 0.7,
) -> Optional[str]:
    """Return the first resolved supplier name from a prioritized list.

    Args:
        names: Ordered raw supplier names to evaluate. Empty strings and
            ``None`` values are skipped.
        known_suppliers: Iterable of canonical supplier names to match
            against.
        rules: Optional list of rule dictionaries compatible with
            :func:`utils_data_cleansing.apply_rule`. When provided, each raw
            name is first normalized using these rules.
        min_score: Minimum similarity score (0-1) required to accept a fuzzy
            match. Defaults to ``0.7``.

    Returns:
        The first non-empty ``names`` entry that successfully resolves through
        :func:`guess_supplier_name`. ``None`` when no values are provided or
        ``known_suppliers`` is empty.
    """

    if not known_suppliers:
        return None

    for candidate in names:
        if candidate is None:
            continue

        candidate = candidate.strip()
        if not candidate:
            continue

        resolved = guess_supplier_name(
            candidate, known_suppliers, rules=rules, min_score=min_score
        )
        if resolved is not None:
            return resolved

    return None
