import logging
import sys
from pathlib import Path
from typing import Any, Dict

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils_geoloc import get_geoloc, load_data


def test_load_data_returns_mapping_and_warnings(tmp_path):
    yaml_content = "Supplier A: [12.34, 56.78]\nSupplier B: [90.12, 34.56]\n"
    mapping_file = tmp_path / "geoloc.yml"
    mapping_file.write_text(yaml_content, encoding="utf-8")

    mapping, warnings = load_data(str(mapping_file))

    expected: Dict[str, Any] = {
        "SUPPLIER A": [12.34, 56.78],
        "SUPPLIER B": [90.12, 34.56],
    }
    assert mapping == expected
    assert warnings == []


def test_load_data_warns_on_missing_file(tmp_path, caplog):
    missing_file = tmp_path / "missing.yml"
    with caplog.at_level(logging.WARNING):
        mapping, warnings = load_data(str(missing_file))

    assert mapping == {}
    assert len(warnings) == 1
    assert "Geolocation mapping file not found" in warnings[0]
    assert warnings[0] in caplog.text


def test_get_geoloc_returns_value_when_present():
    rules = {"SUPPLIER X": [1.0, 2.0]}

    result = get_geoloc("Supplier X", rules)

    assert result == [1.0, 2.0]


def test_get_geoloc_uses_default_for_missing():
    rules = {}
    default_value = [0.0, 0.0]

    result = get_geoloc("Unknown", rules, default_return=default_value)

    assert result == default_value


def test_get_geoloc_raises_for_missing_when_configured():
    rules = {}
    with pytest.raises(KeyError):
        get_geoloc("Missing", rules, raise_on_missing=True)
