import json

import pytest
import torch.nn as nn

from src.common import PATHS, RunLog, count_params, read_lines, write_lines


def test_count_params():
    layer = nn.Linear(4, 2)
    layer.bias.requires_grad_(False)
    assert count_params(layer) == (10, 8)


def test_runlog_failed():
    with pytest.raises(RuntimeError):
        with RunLog("pytest_failed") as log:
            log.set(foo=1)
            raise RuntimeError("boom")
    data = json.loads((PATHS["logs"] / "pytest_failed.json").read_text(encoding="utf-8"))
    assert data["status"] == "failed"
    assert data["foo"] == 1
    (PATHS["logs"] / "pytest_failed.json").unlink()


def test_runlog_append():
    with RunLog("pytest_append") as log:
        log.append("loss_history", [1, 2.5])
        log.append("loss_history", [2, 2.0])
    data = json.loads((PATHS["logs"] / "pytest_append.json").read_text(encoding="utf-8"))
    assert data["status"] == "ok"
    assert data["loss_history"] == [[1, 2.5], [2, 2.0]]
    (PATHS["logs"] / "pytest_append.json").unlink()


def test_write_read_roundtrip(tmp_path):
    lines = ["Grüße aus München.", "Straße"]
    path = tmp_path / "x.de"
    write_lines(path, lines)
    assert read_lines(path) == lines
