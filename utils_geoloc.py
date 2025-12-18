#%%

import yaml

#%%

def load_data(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            _map = yaml.safe_load(f) or {}
        SUPPLIER_NAME_MAP = {str(k).upper(): v for k, v in _map.items()}
    except FileNotFoundError:
        SUPPLIER_NAME_MAP = {}
    return SUPPLIER_NAME_MAP

def get_geoloc(x, rules):
    t = str(x).upper()
    if t in rules:
        return rules[t]
    else:
        return [0.0, 0.0]
