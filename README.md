# LoopPilot

Utility helpers and data files for normalizing supplier names and mapping them to geographic coordinates.

## Supplier name guessing utility (`utils_supplier_name.py`)

`load_supplier_names(path)` reads supplier names from a semicolon-separated CSV file (e.g., `CONSIGNEE_NAME.csv`). `guess_supplier_name(name, known_suppliers, rules=None, min_score=0.7)` cleans the raw value, optionally applies normalization rules from `utils_data_cleansing`, and fuzzy matches it against a list of known supplier names. When you have multiple candidate columns for a supplier name, `guess_supplier_name_from_priority(names, known_suppliers, rules=None, min_score=0.7)` evaluates them in order and returns the first resolved match.

For performance-sensitive workloads you can pre-fit a TF-IDF matcher once with `build_tfidf_matcher(known_suppliers, analyzer="char_wb", ngram_range=(2, 4))` and then call `batch_guess_supplier_names(names, known_suppliers, rules=None, min_score=0.7)` to cleanse many names using the shared model. A lightweight alternative that avoids scikit-learn is `fuzzy_guess_supplier_name(name, known_suppliers, rules=None, min_ratio=0.85)`, which uses Python's built-in sequence matching ratio.

Example usage:

```python
from utils_data_cleansing import load_rules
from utils_supplier_name import guess_supplier_name, load_supplier_names

rules = load_rules("rules.yml")
known_suppliers = load_supplier_names("CONSIGNEE_NAME.csv")

normalized = guess_supplier_name("Huebner GmbH", known_suppliers, rules=rules)
print(normalized)  # Hubner
```

## Data cleansing utility (`utils_data_cleansing.py`)

`load_rules(path)` reads normalization rules from a YAML file, returning a list of dictionaries that include the `type`, `pattern`, and `replacement` keys found in the source file. The helper `apply_rule(text, rules)` lowercases and strips punctuation/stopwords via `rough_clean`, then applies the first matching rule. Supported rule types are:

- **equals**: match the fully cleaned string exactly.
- **regex**: case-insensitive regular expression match.
- **startswith**: check the cleaned string prefix.
- **contains**: substring search on the cleaned string.

Example usage:

```python
from utils_data_cleansing import load_rules, apply_rule

rules = load_rules("rules.yml")
normalized = apply_rule("Hübner GmbH", rules)
print(normalized)  # Hubner
```

## Geolocation utility (`utils_geoloc.py`)

`load_data(path)` reads a YAML mapping of place names to coordinates, returning an uppercase-keyed dictionary. `get_geoloc(name, rules)` performs a lookup using the uppercase name and returns `[lat, lon]` when present or `[0.0, 0.0]` when unknown.

Example usage:

```python
from utils_geoloc import load_data, get_geoloc

geoloc_rules = load_data("geoloc.yml")
coords = get_geoloc("Amsterdam", geoloc_rules)
print(coords)  # [52.3667, 4.9]
```

## Data file formats

- **`rules.yml`**: YAML with a top-level `rules` list. Each item should define `type` (`equals`, `regex`, `startswith`, or `contains`), a `pattern` string used for matching, and the `replacement` value to return. Example:

  ```yaml
  rules:
    - type: startswith
      pattern: "alstom"
      replacement: "Alstom"
    - type: regex
      pattern: "h(?:ue|ü)bner"
      replacement: "Hubner"
  ```

- **`geoloc.yml`**: YAML mapping of place names to `[latitude, longitude]` pairs. Keys are freeform strings; values must be two-element numeric lists. Example:

  ```yaml
  "Amsterdam": [52.3667, 4.9000]
  "Aachen": [50.7753, 6.0839]
  ```

- **`CONSIGNEE_NAME.csv`**: Semicolon-separated CSV summarizing name counts. Columns: `;count` header, where the first column is the normalized consignee name and the second column is the count. Example rows:

  ```csv
  ;count
  Alstom;110436
  Deutsche Bahn;7935
  ```

## Running analysis

- **Notebook**: Open `analysis.ipynb` in Jupyter (e.g., `jupyter notebook` or `jupyter lab`) to explore the cleansing/geolocation logic interactively. The notebook expects the helper modules and data files in the repository root.
- **Standalone scripts**: Import the utilities as shown above or copy the snippets into a Python REPL to test specific names and coordinates.

## Testing

No automated test suite is configured. Run the usage snippets above (or cells within `analysis.ipynb`) to validate changes locally.
