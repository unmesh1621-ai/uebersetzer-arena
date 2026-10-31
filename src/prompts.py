"""Prompt shared by LoRA fine-tuning and few-shot prompting of the decoder-only models."""
import re

from src.common import PATHS, read_lines

INSTRUCTION = "Translate the following English sentence into German. Output only the German translation.\nEnglish: {src}"

_LABEL = re.compile(r"^(german|deutsch|translation|übersetzung)\s*:\s*", re.IGNORECASE)
_QUOTES = "\"'“”„«»"


def few_shot_examples(n: int = 5) -> list[tuple[str, str]]:
    """First n validation pairs whose English side has 5-25 words (fixed, so every run sees the same shots)."""
    en, de = read_lines(PATHS["data"] / "val.en"), read_lines(PATHS["data"] / "val.de")
    return [(e, d) for e, d in zip(en, de) if 5 <= len(e.split()) <= 25][:n]


def build_messages(src: str, shots=None, tgt=None) -> list[dict]:
    msgs = []
    for s_en, s_de in shots or []:
        msgs += [{"role": "user", "content": INSTRUCTION.format(src=s_en)}, {"role": "assistant", "content": s_de}]
    msgs.append({"role": "user", "content": INSTRUCTION.format(src=src)})
    if tgt is not None:
        msgs.append({"role": "assistant", "content": tgt})
    return msgs


def postprocess(text: str) -> tuple[str, bool]:
    """Keep the first non-empty line, drop a 'German:' label and wrapping quotes."""
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    out = lines[0] if lines else ""
    out = _LABEL.sub("", out).strip()
    if len(out) >= 2 and out[0] in _QUOTES and out[-1] in _QUOTES:
        out = out[1:-1].strip()
    return out, out != text
