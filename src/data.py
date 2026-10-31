"""Download WMT14 en-de + FLORES+ devtest, clean, split and write plain-text files to data/."""
import json
import unicodedata

import numpy as np
from datasets import load_dataset
from langdetect import DetectorFactory, LangDetectException, detect

from src.common import PATHS, SEED, write_lines

DetectorFactory.seed = 0

N_TRAIN, N_VAL, N_TEST = 30_000, 2_000, 1_000
MAX_WORDS = 100
MAX_RATIO = 3.0
CHUNK = 20_000


def _is_german(text: str) -> bool:
    try:
        return detect(text) == "de"
    except LangDetectException:
        return False


def clean_pairs(pairs, blocklist):
    """Apply the spec's filters in order; return kept pairs and per-filter drop counts."""
    stats = dict.fromkeys(["empty", "duplicate", "too_long", "ratio", "langid", "leak"], 0)
    seen, kept = set(), []
    for en, de in pairs:
        en = " ".join(unicodedata.normalize("NFC", en).split())
        de = " ".join(unicodedata.normalize("NFC", de).split())
        if not en or not de:
            stats["empty"] += 1
            continue
        if (en, de) in seen:
            stats["duplicate"] += 1
            continue
        seen.add((en, de))
        n_en, n_de = len(en.split()), len(de.split())
        if n_en > MAX_WORDS or n_de > MAX_WORDS:
            stats["too_long"] += 1
            continue
        if max(n_en, n_de) / min(n_en, n_de) > MAX_RATIO:
            stats["ratio"] += 1
            continue
        if en in blocklist or de in blocklist:
            stats["leak"] += 1
            continue
        if not _is_german(de):
            stats["langid"] += 1
            continue
        kept.append((en, de))
    return kept, stats


def _length_stats(lines):
    lens = np.array([len(s.split()) for s in lines])
    return {"mean_words": round(float(lens.mean()), 2), "p95_words": int(np.percentile(lens, 95))}


def main():
    flores = {}
    for lang in ("eng_Latn", "deu_Latn"):
        for split in ("dev", "devtest"):
            ds = load_dataset("openlanguagedata/flores_plus", lang, split=split)
            flores[(lang, split)] = {r["id"]: r["text"] for r in ds}
    ids = sorted(flores[("eng_Latn", "devtest")])
    flores_en = [flores[("eng_Latn", "devtest")][i] for i in ids]
    flores_de = [flores[("deu_Latn", "devtest")][i] for i in ids]
    blocklist = {unicodedata.normalize("NFC", t).strip() for d in flores.values() for t in d.values()}

    wmt = load_dataset("wmt/wmt14", "de-en", split="train")
    order = np.random.default_rng(SEED).permutation(len(wmt))
    need = N_TRAIN + N_VAL + N_TEST
    kept, total_stats, seen_global, n_candidates = [], None, set(), 0
    for start in range(0, len(order), CHUNK):
        rows = wmt.select(order[start:start + CHUNK].tolist())["translation"]
        n_candidates += len(rows)
        chunk_kept, stats = clean_pairs([(r["en"], r["de"]) for r in rows], blocklist)
        for p in chunk_kept:  # dedupe across chunks too
            if p in seen_global:
                stats["duplicate"] += 1
            else:
                seen_global.add(p)
                kept.append(p)
        total_stats = stats if total_stats is None else {k: total_stats[k] + v for k, v in stats.items()}
        print(f"candidates {n_candidates}: kept {len(kept)}")
        if len(kept) >= need:
            break
    kept = kept[:need]
    splits = {
        "train": kept[:N_TRAIN],
        "val": kept[N_TRAIN:N_TRAIN + N_VAL],
        "test": kept[N_TRAIN + N_VAL:],
    }
    for name, pairs in splits.items():
        write_lines(PATHS["data"] / f"{name}.en", [p[0] for p in pairs])
        write_lines(PATHS["data"] / f"{name}.de", [p[1] for p in pairs])
    write_lines(PATHS["data"] / "flores.en", flores_en)
    write_lines(PATHS["data"] / "flores.de", flores_de)

    stats = {
        "source": {"train_pool": "wmt/wmt14 de-en train", "test": "openlanguagedata/flores_plus devtest"},
        "candidates_sampled": n_candidates,
        "dropped": total_stats,
        "counts": {name: len(p) for name, p in splits.items()} | {"flores": len(flores_en)},
        "lengths": {
            f"{name}.{lang}": _length_stats([p[i] for p in pairs])
            for name, pairs in splits.items()
            for i, lang in ((0, "en"), (1, "de"))
        }
        | {"flores.en": _length_stats(flores_en), "flores.de": _length_stats(flores_de)},
    }
    with open(PATHS["data"] / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats["counts"]))


if __name__ == "__main__":
    main()
