import pytest
from transformers import AutoTokenizer

from src.common import load_config
from src.train_lora import encode_example

END_OF_TURN = {"llama": "<|eot_id|>", "gemma": "<end_of_turn>"}


@pytest.fixture(scope="module", params=["llama", "gemma"])
def tok_and_name(request):
    cfg = load_config(f"configs/{request.param}_lora.yaml")
    return AutoTokenizer.from_pretrained(cfg["model_name"]), request.param


def test_only_answer_is_labelled(tok_and_name):
    tok, name = tok_and_name
    ex = encode_example(tok, "Hello.", "Hallo.", max_len=256)
    labelled = [t for t, l in zip(ex["input_ids"], ex["labels"]) if l != -100]
    text = tok.decode(labelled)
    assert text.startswith("Hallo.")
    assert END_OF_TURN[name] in text
    first = next(i for i, l in enumerate(ex["labels"]) if l != -100)
    assert all(l == -100 for l in ex["labels"][:first])
    assert ex["labels"][first:] == ex["input_ids"][first:]
    assert "Translate the following English sentence" in tok.decode(ex["input_ids"][:first])


def test_truncated_to_max_len(tok_and_name):
    tok, _ = tok_and_name
    ex = encode_example(tok, "word " * 300, "Wort " * 300, max_len=256)
    assert len(ex["input_ids"]) <= 256
    assert len(ex["labels"]) == len(ex["input_ids"]) == len(ex["attention_mask"])
