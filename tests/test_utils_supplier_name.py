from utils_supplier_name import (
    batch_guess_supplier_names,
    fuzzy_guess_supplier_name,
    guess_supplier_name,
    guess_supplier_name_from_priority,
    load_supplier_names,
)


def test_load_supplier_names_reads_first_column(tmp_path):
    sample = tmp_path / "suppliers.csv"
    sample.write_text(";count\nAlpha;10\nBeta;5\n", encoding="utf-8")

    names = load_supplier_names(sample)

    assert names == ["Alpha", "Beta"]


def test_guess_supplier_name_exact_match_with_rules():
    rules = [
        {"type": "equals", "pattern": "huebner gmbh", "replacement": "Hubner"}
    ]
    known = ["Hubner", "Alstom"]

    result = guess_supplier_name("Huebner GmbH", known, rules=rules)

    assert result == "Hubner"


def test_guess_supplier_name_returns_best_fuzzy_match():
    known = ["Siemens", "Alstom", "Knorr Bremse"]

    result = guess_supplier_name("Siemen", known, rules=None, min_score=0.6)

    assert result == "Siemens"


def test_guess_supplier_name_handles_longer_inputs_with_cosine_similarity():
    known = ["Knorr Bremse", "Siemens"]

    result = guess_supplier_name(
        "Knorr Bremsee Rail Systems", known, rules=None, min_score=0.3
    )

    assert result == "Knorr Bremse"


def test_guess_supplier_name_handles_missing_values():
    assert guess_supplier_name(None, ["Alpha"]) is None
    assert guess_supplier_name("test", [], []) is None


def test_guess_supplier_name_from_priority_returns_first_valid_match():
    known = ["Siemens", "Alstom", "Knorr Bremse"]
    names = [None, "", "Siemen", "Fallback"]

    result = guess_supplier_name_from_priority(names, known, rules=None, min_score=0.6)

    assert result == "Siemens"


def test_guess_supplier_name_from_priority_returns_none_when_no_candidates():
    known = ["Alpha"]

    result = guess_supplier_name_from_priority([], known)

    assert result is None


def test_batch_guess_supplier_names_reuses_vectorizer_for_multiple_inputs():
    known = ["Siemens", "Knorr Bremse"]
    raw_names = ["Siemen", "Knorr Bremsee Rail Systems", None]

    results = batch_guess_supplier_names(raw_names, known, min_score=0.5)

    assert results == ["Siemens", "Knorr Bremse", None]


def test_fuzzy_guess_supplier_name_uses_ratio_matching():
    known = ["Knorr Bremse", "Alstom"]

    result = fuzzy_guess_supplier_name("Knor Bremse GmbH", known, min_ratio=0.6)

    assert result == "Knorr Bremse"
