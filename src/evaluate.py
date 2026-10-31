"""Score every system on FLORES+ devtest and merge size, training-time and inference-time figures into metrics.csv."""
import json

import pandas as pd
from sacrebleu.metrics import BLEU, CHRF

from src.common import PATHS, ROWS, load_config, read_lines

_BLEU = BLEU(tokenize="13a")
_CHRF = CHRF(char_order=6, word_order=0)


def score(hyps: list[str], refs: list[str]) -> dict:
    if len(hyps) != len(refs):
        raise ValueError(f"{len(hyps)} hypotheses vs {len(refs)} references")
    bleu = _BLEU.corpus_score(hyps, [refs])
    chrf = _CHRF.corpus_score(hyps, [refs])
    return {"bleu": round(bleu.score, 2), "chrf": round(chrf.score, 2),
            "bleu_sig": str(_BLEU.get_signature()), "chrf_sig": str(_CHRF.get_signature())}


def tokenizer_fertility() -> pd.DataFrame:
    """Mean tokens per FLORES devtest sentence for each tokenizer (no special tokens)."""
    import sentencepiece as spm
    from transformers import AutoTokenizer

    en, de = read_lines(PATHS["data"] / "flores.en"), read_lines(PATHS["data"] / "flores.de")
    sp = spm.SentencePieceProcessor(model_file=str(PATHS["checkpoints"] / "transformer" / "spm.model"))
    encoders = {"transformer (SentencePiece 16k)": lambda s: sp.encode(s)}
    for label, cfg in [("mT5", "mt5"), ("LLaMA-3.2", "llama_lora"), ("Gemma-3", "gemma_lora")]:
        tok = AutoTokenizer.from_pretrained(load_config(f"configs/{cfg}.yaml")["model_name"])
        encoders[f"{label} ({len(tok):,} vocab)"] = lambda s, t=tok: t(s, add_special_tokens=False)["input_ids"]
    rows = []
    for name, enc in encoders.items():
        n_en = sum(len(enc(s)) for s in en) / len(en)
        n_de = sum(len(enc(s)) for s in de) / len(de)
        rows.append({"tokenizer": name, "tokens_per_sent_en": round(n_en, 1), "tokens_per_sent_de": round(n_de, 1),
                     "words_per_sent_de": round(sum(len(s.split()) for s in de) / len(de), 1)})
    return pd.DataFrame(rows)


def _json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _train_log(row):
    family = row.split("_")[0]
    if row in ("transformer", "mt5"):
        return _json(PATHS["logs"] / f"train_{row}.json")
    lora = _json(PATHS["logs"] / f"train_{family}_lora.json")
    if row.endswith("_lora"):
        return lora
    # Prompting uses the frozen base model: same total size as the LoRA run minus the adapter, nothing trained.
    return {"params_total": lora.get("params_total", 0) - lora.get("params_trainable", 0), "params_trainable": 0,
            "train_seconds": 0.0}


def main():
    refs = read_lines(PATHS["data"] / "flores.de")
    records, sigs = [], {}
    for row in ROWS:
        cfg = load_config(f"configs/{row}.yaml")
        hyps = read_lines(PATHS["predictions"] / f"{row}.de")
        s = score(hyps, refs)
        sigs = {"bleu": s["bleu_sig"], "chrf": s["chrf_sig"]}
        log, timing = _train_log(row), _json(PATHS["timing"] / f"{row}.json")
        rec = {
            "row": row, "arch": cfg["arch"], "adaptation": cfg["adaptation"], "bleu": s["bleu"], "chrf": s["chrf"],
            "bleu_beam4": None, "chrf_beam4": None,
            "params_total": log.get("params_total"), "params_trainable": log.get("params_trainable"),
            "train_seconds": log.get("train_seconds"), "infer_total_s": timing.get("total_s"),
            "ms_per_sent": timing.get("ms_per_sent"), "tok_per_s": timing.get("tok_per_s"),
            "peak_vram_mb": timing.get("peak_vram_mb"),
            "postproc_pct": round(100 * timing.get("postproc_changed", 0) / max(timing.get("n", 1), 1), 1),
        }
        beam_path = PATHS["predictions"] / f"{row}_beam4.de"
        if beam_path.exists():
            b = score(read_lines(beam_path), refs)
            rec["bleu_beam4"], rec["chrf_beam4"] = b["bleu"], b["chrf"]
        records.append(rec)
    df = pd.DataFrame(records)
    df.to_csv(PATHS["results"] / "metrics.csv", index=False)
    (PATHS["results"] / "signatures.json").write_text(json.dumps(sigs, indent=2), encoding="utf-8")
    tokenizer_fertility().to_csv(PATHS["results"] / "fertility.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
