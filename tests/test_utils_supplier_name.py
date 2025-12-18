import os
from utils_supplier_name import guess_supplier_name, load_supplier_names


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


def test_guess_supplier_name_handles_missing_values():
    assert guess_supplier_name(None, ["Alpha"]) is None
    assert guess_supplier_name("test", [], []) is None
