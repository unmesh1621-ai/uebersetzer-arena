"""Shared helpers: paths, seeding, configs, run logs, parameter counting, text I/O."""
import json
import platform
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
PATHS = {
    name: ROOT / rel
    for name, rel in {
        "data": "data",
        "results": "results",
        "predictions": "results/predictions",
        "timing": "results/timing",
        "logs": "results/logs",
        "checkpoints": "checkpoints",
        "plots": "plots",
        "samples": "samples",
    }.items()
}
for _p in PATHS.values():
    _p.mkdir(parents=True, exist_ok=True)

SEED = 42
ROWS = ["transformer", "mt5", "llama_prompt", "llama_lora", "gemma_prompt", "gemma_lora"]


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path) -> dict:
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["row"] = path.stem
    return cfg


def count_params(model) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def read_lines(path) -> list[str]:
    with open(path, encoding="utf-8") as f:
        content = f.read()
    if not content:
        return []
    if content.endswith("\n"):
        content = content[:-1]
    return content.split("\n")


def write_lines(path, lines) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for line in lines:
            f.write(line.replace("\n", " ") + "\n")


def _versions() -> dict:
    out = {"python": platform.python_version(), "torch": torch.__version__}
    for mod in ("transformers", "peft", "sacrebleu", "sentencepiece"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:
            pass
    return out


class RunLog:
    """Context manager that records status, timings and arbitrary values to results/logs/<name>.json."""

    def __init__(self, name: str):
        self.name = name
        self.data: dict = {}

    def set(self, **kv) -> None:
        self.data.update(kv)

    def append(self, key: str, value) -> None:
        self.data.setdefault(key, []).append(value)

    def __enter__(self):
        self._t0 = time.time()
        self.data.update(
            start=datetime.now().isoformat(timespec="seconds"),
            gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            versions=_versions(),
        )
        return self

    def __exit__(self, exc_type, exc, tb):
        self.data.update(
            status="failed" if exc_type else "ok",
            end=datetime.now().isoformat(timespec="seconds"),
            seconds=round(time.time() - self._t0, 2),
        )
        if exc_type:
            self.data["error"] = f"{exc_type.__name__}: {exc}"
        with open(PATHS["logs"] / f"{self.name}.json", "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)
        return False
