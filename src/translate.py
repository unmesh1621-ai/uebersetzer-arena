"""Unified loading, batched translation and timing for all six systems.

CLI: python -m src.translate --system <row> [--temp T] [--beam 4] [--subset N] [--input data/flores.en]
"""
import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

from src.common import PATHS, ROWS, SEED, load_config, read_lines, write_lines
from src.prompts import build_messages, few_shot_examples, postprocess
from src.train_lora import render
from src.transformer_scratch.decode import beam_decode, greedy_decode, sample_decode
from src.transformer_scratch.train import BOS, EOS, load_transformer, pad

END_OF_TURN = {"llama": "<|eot_id|>", "gemma": "<end_of_turn>"}


class Translator:
    """Common interface; subclasses implement _batch(srcs, temperature, beam, max_new_tokens) -> (outputs, n_tokens)."""

    def __init__(self):
        self.gen_tokens = 0
        self.postproc_changed = 0

    def translate(self, srcs, temperature=0.0, beam=1, batch_size=16, max_new_tokens=256, seed=SEED):
        torch.manual_seed(seed)
        self.gen_tokens = self.postproc_changed = 0
        clean = [" ".join(s.split()) for s in srcs]
        out = [""] * len(clean)
        todo = sorted((i for i, s in enumerate(clean) if s), key=lambda i: -len(clean[i]))
        for k in range(0, len(todo), batch_size):
            idx = todo[k : k + batch_size]
            hyps, n_tok = self._batch([clean[i] for i in idx], temperature, beam, max_new_tokens)
            self.gen_tokens += n_tok
            for i, h in zip(idx, hyps):
                out[i] = " ".join(h.split())
        return out


class TransformerTranslator(Translator):
    def __init__(self):
        super().__init__()
        self.model, self.sp = load_transformer()

    def _batch(self, srcs, temperature, beam, max_new_tokens):
        src = pad([self.sp.encode(s)[:127] + [EOS] for s in srcs], "cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if beam > 1:
                ids = beam_decode(self.model, src, BOS, EOS, max_new_tokens, beam=beam)
            elif temperature > 0:
                gen = torch.Generator(device="cuda").manual_seed(torch.initial_seed() % 2**63)
                ids = sample_decode(self.model, src, BOS, EOS, max_new_tokens, temperature, gen)
            else:
                ids = greedy_decode(self.model, src, BOS, EOS, max_new_tokens)
        return [self.sp.decode(x) for x in ids], sum(len(x) for x in ids)


def _gen_kwargs(temperature, beam, max_new_tokens):
    kw = {"max_new_tokens": max_new_tokens, "num_beams": beam}
    if temperature > 0:
        kw.update(do_sample=True, temperature=temperature, top_p=1.0, top_k=0)
    else:
        kw.update(do_sample=False, temperature=None, top_p=None, top_k=None)
    return kw


def _count_new(rows, stop_ids):
    n = 0
    for row in rows:
        for t in row:
            if t in stop_ids:
                break
            n += 1
    return n


class MT5Translator(Translator):
    def __init__(self):
        super().__init__()
        self.cfg = load_config("configs/mt5.yaml")
        path = PATHS["checkpoints"] / "mt5" / "best"
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(path, torch_dtype=torch.bfloat16).to("cuda").eval()

    @torch.no_grad()
    def _batch(self, srcs, temperature, beam, max_new_tokens):
        enc = self.tok([self.cfg["prefix"] + s for s in srcs], return_tensors="pt", padding=True, truncation=True,
                       max_length=self.cfg["max_len"]).to("cuda")
        out = self.model.generate(**enc, **_gen_kwargs(temperature, beam, max_new_tokens))
        new = out[:, 1:].tolist()  # drop decoder start token
        n = _count_new(new, {self.tok.eos_token_id, self.tok.pad_token_id})
        return self.tok.batch_decode(out, skip_special_tokens=True), n


class DecoderTranslator(Translator):
    def __init__(self, row):
        super().__init__()
        family, mode = row.split("_")
        self.cfg = load_config(f"configs/{row}.yaml")
        self.tok = AutoTokenizer.from_pretrained(self.cfg["model_name"])
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        # Gemma-3 must use eager attention (as in training): with SDPA, batched left-padded greedy generation
        # turned whole batches into unrelated French text (tests/test_gemma_batch.py).
        attn = "eager" if family == "gemma" else "sdpa"
        model = AutoModelForCausalLM.from_pretrained(self.cfg["model_name"], torch_dtype=torch.bfloat16,
                                                     attn_implementation=attn).to("cuda")
        if mode == "lora":
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, PATHS["checkpoints"] / row).merge_and_unload()
        self.model = model.eval()
        self.shots = few_shot_examples(self.cfg["shots"]) if mode == "prompt" else None
        self.stop_ids = {self.tok.eos_token_id, self.tok.convert_tokens_to_ids(END_OF_TURN[family])}

    @torch.no_grad()
    def _batch(self, srcs, temperature, beam, max_new_tokens):
        prompts = [render(self.tok, build_messages(s, shots=self.shots), add_generation_prompt=True) for s in srcs]
        enc = self.tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        out = self.model.generate(**enc, **_gen_kwargs(temperature, beam, max_new_tokens),
                                  eos_token_id=list(self.stop_ids), pad_token_id=self.tok.pad_token_id)
        new = out[:, enc["input_ids"].shape[1]:]
        texts = []
        for row in self.tok.batch_decode(new, skip_special_tokens=True):
            clean, changed = postprocess(row)
            self.postproc_changed += changed
            texts.append(clean)
        return texts, _count_new(new.tolist(), self.stop_ids)


def load_system(row: str) -> Translator:
    if row not in ROWS:
        raise ValueError(f"unknown system {row}; expected one of {ROWS}")
    if row == "transformer":
        return TransformerTranslator()
    if row == "mt5":
        return MT5Translator()
    return DecoderTranslator(row)


def suffix_for(temperature: float, beam: int) -> str:
    if beam > 1:
        return f"_beam{beam}"
    return f"_t{temperature}" if temperature > 0 else ""


def run_translation(system, input_path=PATHS["data"] / "flores.en", temperature=0.0, beam=1, subset=None,
                    batch_size=16, max_new_tokens=256, out_dir=None, translator=None) -> Path:
    srcs = read_lines(input_path)[:subset] if subset else read_lines(input_path)
    tr = translator or load_system(system)
    tr.translate(srcs[:batch_size], temperature, beam, batch_size, max_new_tokens)  # warm-up, untimed
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    hyps = tr.translate(srcs, temperature, beam, batch_size, max_new_tokens)
    torch.cuda.synchronize()
    total = time.perf_counter() - t0
    name = system + suffix_for(temperature, beam)
    pred_path = Path(out_dir or PATHS["predictions"]) / f"{name}.de"
    write_lines(pred_path, hyps)
    timing = {
        "system": system, "input": str(input_path), "n": len(srcs), "temperature": temperature, "beam": beam,
        "batch_size": batch_size, "max_new_tokens": max_new_tokens, "total_s": round(total, 3),
        "ms_per_sent": round(1000 * total / max(len(srcs), 1), 2), "gen_tokens": tr.gen_tokens,
        "tok_per_s": round(tr.gen_tokens / total, 1), "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 2**20),
        "postproc_changed": tr.postproc_changed,
    }
    timing_path = Path(out_dir or PATHS["timing"]) / f"{name}.json"
    timing_path.write_text(json.dumps(timing, indent=2), encoding="utf-8")
    print(json.dumps(timing))
    return pred_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True, choices=ROWS)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--beam", type=int, default=1)
    ap.add_argument("--subset", type=int)
    ap.add_argument("--input", default=str(PATHS["data"] / "flores.en"))
    args = ap.parse_args()
    run_translation(args.system, args.input, args.temp, args.beam, args.subset)


if __name__ == "__main__":
    main()
