import json

import pytest

from src.common import read_lines, write_lines
from src.translate import load_system, run_translation

pytestmark = pytest.mark.slow
TRICKY = ["Hello.", "", "Line one.\nLine two."]


@pytest.fixture(scope="module")
def llama_prompt():
    return load_system("llama_prompt")


def test_decoder_uses_left_padding(llama_prompt):
    assert llama_prompt.tok.padding_side == "left"


def test_one_clean_line_per_input(llama_prompt):
    out = llama_prompt.translate(TRICKY, max_new_tokens=32)
    assert len(out) == 3
    assert all("\n" not in o for o in out)
    assert out[1] == ""


def test_prediction_file_line_count(tmp_path):
    src = tmp_path / "in.en"
    write_lines(src, ["Good morning.", "", "The weather is nice today."])
    pred = run_translation("transformer", src, out_dir=tmp_path, max_new_tokens=16)
    assert len(read_lines(pred)) == 3
    timing = json.loads((tmp_path / "transformer.json").read_text(encoding="utf-8"))
    assert timing["n"] == 3 and timing["total_s"] > 0
