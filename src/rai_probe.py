"""Responsible-AI probe: translate a curated probe set with all systems and flag suspicious FLORES outputs."""
import csv
import gc
import re
from collections import Counter

import pandas as pd
import torch
from langdetect import DetectorFactory, LangDetectException, detect

from src.common import PATHS, ROWS, read_lines

DetectorFactory.seed = 0
_NUM = re.compile(r"\d+(?:[.,:]\d+)*")
_WORD = re.compile(r"[A-Za-zÄÖÜäöüß]{3,}")


def _numbers(text):
    return sorted(re.sub(r"\D", "", n) for n in _NUM.findall(text))


def flags(src: str, hyp: str) -> list[str]:
    """Heuristic warning signs for hallucination, omission, non-translation and number errors."""
    out = []
    n_src, n_hyp = len(src.split()), len(hyp.split())
    if n_src and (n_hyp / n_src > 2.0 or n_hyp / n_src < 0.5):
        out.append("length_ratio")
    if n_hyp >= 4:
        try:
            if detect(hyp) != "de":
                out.append("not_german")
        except LangDetectException:
            out.append("not_german")
    src_words = {w.lower() for w in _WORD.findall(src)}
    hyp_words = [w.lower() for w in _WORD.findall(hyp)]
    if hyp_words and sum(w in src_words for w in hyp_words) / len(hyp_words) >= 0.6:
        out.append("untranslated")
    toks = hyp.lower().split()
    trigrams = Counter(zip(toks, toks[1:], toks[2:]))
    if any(c >= 3 for c in trigrams.values()) or re.search(r"\b(\w+)(?:\W+\1\b){3,}", hyp.lower()):
        out.append("repetition")
    if _numbers(src) != _numbers(hyp):
        out.append("number_mismatch")
    return out


REVIEW_SYSTEMS = ["mt5", "llama_lora", "gemma_lora"]
REVIEW_CATEGORIES = {"gender", "sensitive", "culture", "hallucination"}
FLAGGED_PER_SYSTEM = 3


LABEL_COLS = ["back_translation", "label", "note"]


def merge_labels(new: pd.DataFrame, old: pd.DataFrame) -> pd.DataFrame:
    """Copy manual labels from an earlier review sheet onto rows whose (source, system, output) is unchanged."""
    key = ["source", "system", "output"]
    kept = old[key + LABEL_COLS].drop_duplicates(key)
    out = new.drop(columns=LABEL_COLS).merge(kept, on=key, how="left")
    out[LABEL_COLS] = out[LABEL_COLS].fillna("")
    return out


def main():
    from src.translate import load_system

    probe = pd.read_csv(PATHS["data"] / "rai_probe.tsv", sep="\t")
    flores_en = read_lines(PATHS["data"] / "flores.en")
    probe_rows, review, flag_counts = [], [], []
    for row in ROWS:
        tr = load_system(row)
        outs = tr.translate(probe["en"].tolist())
        for (_, p), o in zip(probe.iterrows(), outs):
            probe_rows.append({"id": p["id"], "category": p["category"], "en": p["en"], "system": row, "output": o,
                               "flags": ";".join(flags(p["en"], o))})
            if row in REVIEW_SYSTEMS and p["category"] in REVIEW_CATEGORIES:
                review.append({"source": p["en"], "system": row, "output": o, "flags": ";".join(flags(p["en"], o)),
                               "category": p["category"], "back_translation": "", "label": "", "note": ""})
        del tr
        gc.collect()
        torch.cuda.empty_cache()

        hyps = read_lines(PATHS["predictions"] / f"{row}.de")
        counts, shown = Counter(), 0
        for src, hyp in zip(flores_en, hyps):
            f = flags(src, hyp)
            counts.update(f)
            if f and shown < FLAGGED_PER_SYSTEM and row in REVIEW_SYSTEMS + ["llama_prompt", "gemma_prompt"]:
                review.append({"source": src, "system": row, "output": hyp, "flags": ";".join(f),
                               "category": "flores_flagged", "back_translation": "", "label": "", "note": ""})
                shown += 1
        flag_counts.append({"row": row, "n": len(hyps), **{k: counts.get(k, 0) for k in
                            ["length_ratio", "not_german", "untranslated", "repetition", "number_mismatch"]},
                            "any_flag": sum(bool(flags(s, h)) for s, h in zip(flores_en, hyps))})

    pd.DataFrame(probe_rows).to_csv(PATHS["samples"] / "rai_probe.csv", index=False, quoting=csv.QUOTE_NONNUMERIC)
    sheet = PATHS["samples"] / "rai_review.csv"
    review = pd.DataFrame(review)
    if sheet.exists():  # never wipe manual labels: carry them over wherever the output is unchanged
        review = merge_labels(review, pd.read_csv(sheet, keep_default_na=False))
    review.to_csv(sheet, index=False, quoting=csv.QUOTE_NONNUMERIC)
    pd.DataFrame(flag_counts).to_csv(PATHS["results"] / "rai_flags.csv", index=False)
    print(pd.DataFrame(flag_counts).to_string(index=False))


if __name__ == "__main__":
    main()
