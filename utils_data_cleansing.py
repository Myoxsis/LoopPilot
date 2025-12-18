#%%

import re
try:  # Optional dependency for loading YAML rule files
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - only hit when PyYAML is missing
    yaml = None
from dataclasses import dataclass
from typing import List, Optional

#%%

def load_rules(path):
    if yaml is None:
        raise ImportError(
            "PyYAML is required to load rule files. Install it or provide rules manually."
        )

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    rules = []
    for item in raw.get('rules', []):
        # Only pass known fields
        kwargs = {
        k: item[k] for k in (
        "type", "pattern", "replacement"
        ) if k in item
        }
        rules.append(kwargs)
    return rules

def rough_clean(text):
    t = str(text).lower().lstrip(' ')

    punct_to_delete = [".", '"', "#", ",", ";"]
    punct_to_space = ["/", '-', "  ", "   "]
    stopwords_to_delete = ["gmbh & cokg", "gmbh & co kg", "gmbh & co", "gmbh", 'gmb', 'gmh', "ltd", "sp z o o", "sa", "sas", 'ab', 'nv', "ag", "bv", "sp", "spz"]
    
    for w in punct_to_delete:
        t = t.replace(w, '')
    for w in punct_to_space:
        t = t.replace(w, ' ')

    t = t.split(" ")
    t = [i for i in t if i not in stopwords_to_delete]
    t = " ".join(t).rstrip(' ')
    return t


def apply_rule(text, rules):
    t = rough_clean(text)
    for r in rules:
        if r['type'] == "equals":
            if r['pattern'].lower() == t.lower():
                return r['replacement']
        if r['type'] == "regex":
            rx = re.compile(r['pattern'])
            if rx.search(t.lower()):
                return r['replacement']
        if r['type'] == "startswith":
            if t.startswith(r['pattern'].lower()):
                return r['replacement']
        if r['type'] == "contains":
            if r['pattern'] in t:
                return r['replacement']     
    return t.title()

#%%

#rules = load_rules("rules.yml")

#for text in ['Hübner', "Huebner"]:
#    t = rough_clean(text)
#    print(t)
#    x = apply_rule(t, rules)
#    print(x)
