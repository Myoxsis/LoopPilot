#%%

import ast
import logging
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

try:  # Optional dependency
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - exercised implicitly when PyYAML is missing
    yaml = None

#%%

def load_data(path: str) -> Tuple[Dict[str, Any], List[str]]:
    """Load supplier name mapping from a YAML file.

    Returns both the mapping and a list of warnings encountered while reading
    the file so callers can decide how to surface issues.
    """
    warnings: List[str] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            file_contents = f.read()
        raw_map = (
            (yaml.safe_load(file_contents) if yaml else _parse_yaml_like(file_contents))
            or {}
        )
        supplier_name_map = {str(k).upper(): v for k, v in raw_map.items()}
    except FileNotFoundError:
        message = f"Geolocation mapping file not found: {path}"
        logging.warning(message)
        warnings.append(message)
        supplier_name_map = {}
    return supplier_name_map, warnings


def get_geoloc(
    name: str,
    rules: Mapping[str, Any],
    default_return: Optional[Sequence[float]] = None,
    raise_on_missing: bool = False,
) -> Any:
    """Return the geolocation for a supplier name.

    Args:
        name: Supplier name to look up.
        rules: Mapping of supplier names to geolocation values.
        default_return: Value to return when the supplier name is missing.
            Defaults to ``None``.
        raise_on_missing: When True, raise a KeyError if the name is missing
            instead of returning ``default_return``.
    """
    normalized_name = str(name).upper()
    if normalized_name in rules:
        return rules[normalized_name]
    if raise_on_missing:
        raise KeyError(f"Supplier name '{name}' not found in geolocation rules.")
    return default_return


def _parse_yaml_like(content: str) -> Dict[str, Any]:
    """Parse a minimal YAML-like mapping without requiring PyYAML.

    This supports the simple ``key: [value, value]`` structure used in
    ``geoloc.yml`` and serves as a fallback when PyYAML is unavailable.
    """
    cleaned_lines: List[str] = []
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        key, value = stripped.split(":", maxsplit=1)
        normalized_key = key.strip().strip('"').strip("'")
        cleaned_lines.append(f"{normalized_key!r}:{value}")
    mapping_literal = "{\n" + ",\n".join(cleaned_lines) + "\n}"
    return ast.literal_eval(mapping_literal)
