# src/utils.py
from __future__ import annotations
import json, os, random, time, pathlib
import numpy as np

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch  # optional
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

def ensure_dir(p: str | os.PathLike) -> None:
    pathlib.Path(p).mkdir(parents=True, exist_ok=True)

def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")

def dump_json(obj, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
