"""Sampling-temperature analysis: BLEU/chrF for every system at T in {0.3, 0.7, 1.0, 1.3} on 300 FLORES sentences.

Results are merged into results/temp_sweep.csv by (row, temperature), so rows can be (re)run separately with --rows.
"""
import argparse
import gc
import traceback

import pandas as pd
import torch

from src.common import PATHS, ROWS, read_lines
from src.evaluate import score
from src.translate import load_system, run_translation

TEMPS = [0.3, 0.7, 1.0, 1.3]
N = 300
OUT = PATHS["results"] / "temp_sweep.csv"


def _save(records):
    new = pd.DataFrame(records)
    if OUT.exists():
        old = pd.read_csv(OUT)
        keys = set(zip(new.row, new.temperature))
        old = old[[(r, t) not in keys for r, t in zip(old.row, old.temperature)]]
        new = pd.concat([old, new], ignore_index=True)
    new.sort_values(["row", "temperature"]).to_csv(OUT, index=False)


def sweep_row(row, refs):
    greedy = read_lines(PATHS["predictions"] / f"{row}.de")[:N]  # T = 0 point, same 300 sentences
    s0 = score(greedy, refs)
    records = [{"row": row, "temperature": 0.0, "bleu": s0["bleu"], "chrf": s0["chrf"]}]
    tr = load_system(row)
    for t in TEMPS:
        pred = run_translation(row, temperature=t, subset=N, translator=tr)
        s = score(read_lines(pred), refs)
        records.append({"row": row, "temperature": t, "bleu": s["bleu"], "chrf": s["chrf"]})
        print(row, t, s["bleu"], s["chrf"], flush=True)
    del tr
    gc.collect()
    torch.cuda.empty_cache()
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", nargs="+", default=ROWS, choices=ROWS)
    args = ap.parse_args()
    refs = read_lines(PATHS["data"] / "flores.de")[:N]
    failed = []
    for row in args.rows:
        try:
            _save(sweep_row(row, refs))
        except Exception:
            traceback.print_exc()
            failed.append(row)
    if failed:
        raise SystemExit(f"temperature sweep failed for: {failed}")


if __name__ == "__main__":
    main()
