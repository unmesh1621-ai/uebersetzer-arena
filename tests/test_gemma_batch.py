import pytest

from src.common import PATHS, read_lines
from src.translate import load_system

pytestmark = pytest.mark.slow


def test_gemma_batched_greedy_stays_german():
    # Regression: with SDPA attention, this length-sorted batch of 16 FLORES sentences (positions 304-319)
    # came back as French text starting with "majeur" for 14 of 16 rows; eager attention fixes it.
    src = [" ".join(s.split()) for s in read_lines(PATHS["data"] / "flores.en")]
    order = sorted(range(len(src)), key=lambda i: -len(src[i]))
    batch = [src[i] for i in order[304:320]]
    tr = load_system("gemma_lora")
    assert tr.model.config._attn_implementation == "eager"
    out = tr.translate(batch, batch_size=16)
    assert not [o for o in out if o.startswith("majeur")]
