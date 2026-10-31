"""Train the from-scratch Transformer: SentencePiece vocab, token-bucketed batches, Noam schedule, early stopping."""
import argparse
import random
import time

import sacrebleu
import sentencepiece as spm
import torch
import torch.nn as nn

from src.common import PATHS, SEED, RunLog, count_params, load_config, read_lines, set_seed
from src.transformer_scratch.decode import greedy_decode
from src.transformer_scratch.model import Seq2SeqTransformer

PAD, UNK, BOS, EOS = 0, 1, 2, 3
CKPT = PATHS["checkpoints"] / "transformer"


def train_spm(texts, vocab_size):
    CKPT.mkdir(parents=True, exist_ok=True)
    corpus = CKPT / "spm_corpus.txt"
    corpus.write_text("\n".join(texts) + "\n", encoding="utf-8")
    spm.SentencePieceTrainer.train(
        input=str(corpus), model_prefix=str(CKPT / "spm"), vocab_size=vocab_size, model_type="unigram",
        character_coverage=1.0, pad_id=PAD, unk_id=UNK, bos_id=BOS, eos_id=EOS,
        hard_vocab_limit=False, minloglevel=2,
    )
    corpus.unlink()
    return spm.SentencePieceProcessor(model_file=str(CKPT / "spm.model"))


def make_batches(src_ids, tgt_ids, batch_tokens):
    order = sorted(range(len(src_ids)), key=lambda i: (len(src_ids[i]), len(tgt_ids[i])))
    batches, cur, cur_max = [], [], 0
    for i in order:
        longest = max(cur_max, len(src_ids[i]), len(tgt_ids[i]))
        if cur and longest * (len(cur) + 1) > batch_tokens:
            batches.append(cur)
            cur, longest = [], max(len(src_ids[i]), len(tgt_ids[i]))
        cur.append(i)
        cur_max = longest
    if cur:
        batches.append(cur)
    return batches


def pad(seqs, device):
    width = max(len(s) for s in seqs)
    return torch.tensor([s + [PAD] * (width - len(s)) for s in seqs], device=device)


def build_model(cfg, vocab_size):
    return Seq2SeqTransformer(vocab_size, cfg["d_model"], cfg["nhead"], cfg["num_layers"], cfg["dim_ff"], cfg["dropout"], PAD)


def load_transformer(ckpt_dir=CKPT, device="cuda"):
    state = torch.load(ckpt_dir / "best.pt", map_location=device)
    sp = spm.SentencePieceProcessor(model_file=str(ckpt_dir / "spm.model"))
    model = build_model(state["cfg"], state["vocab_size"]).to(device)
    model.load_state_dict(state["model"])
    return model.eval(), sp


def translate_batch(model, sp, sentences, max_len, device, batch_size=64):
    out = []
    for i in range(0, len(sentences), batch_size):
        src = pad([sp.encode(s)[: max_len - 1] + [EOS] for s in sentences[i : i + batch_size]], device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out += [sp.decode(ids) for ids in greedy_decode(model, src, BOS, EOS, max_len)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/transformer.yaml")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(SEED)
    device = "cuda"

    train_en, train_de = read_lines(PATHS["data"] / "train.en"), read_lines(PATHS["data"] / "train.de")
    val_en, val_de = read_lines(PATHS["data"] / "val.en"), read_lines(PATHS["data"] / "val.de")
    max_steps = None
    if args.smoke:
        train_en, train_de, val_en, val_de, max_steps = train_en[:200], train_de[:200], val_en[:32], val_de[:32], 20

    with RunLog("train_transformer") as log:
        sp = train_spm(train_en + train_de, cfg["vocab_size"] if not args.smoke else 1000)
        L = cfg["max_len"]
        src_ids = [sp.encode(s)[: L - 1] + [EOS] for s in train_en]
        tgt_ids = [[BOS] + sp.encode(s)[: L - 2] + [EOS] for s in train_de]
        batches = make_batches(src_ids, tgt_ids, cfg["batch_tokens"])

        model = build_model(cfg, sp.get_piece_size()).to(device)
        total, trainable = count_params(model)
        log.set(params_total=total, params_trainable=trainable, vocab_size=sp.get_piece_size(), smoke=args.smoke)
        opt = torch.optim.Adam(model.parameters(), lr=1.0, betas=tuple(cfg["betas"]), eps=1e-9)
        d, w = cfg["d_model"], cfg["warmup"]
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: cfg["lr_factor"] * d ** -0.5 * min((s + 1) ** -0.5, (s + 1) * w ** -1.5)
        )
        loss_fn = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=cfg["label_smoothing"])

        best, bad_epochs, step, t0 = -1.0, 0, 0, time.time()
        rng = random.Random(SEED)
        for epoch in range(1, cfg["epochs"] + 1):
            model.train()
            rng.shuffle(batches)
            for b in batches:
                src = pad([src_ids[i] for i in b], device)
                tgt = pad([tgt_ids[i] for i in b], device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(src, tgt[:, :-1])
                loss = loss_fn(logits.float().reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                step += 1
                if step % 50 == 0 or max_steps:
                    log.append("loss_history", [step, round(loss.item(), 4)])
                if max_steps and step >= max_steps:
                    break
            model.eval()
            hyps = translate_batch(model, sp, val_en, L, device)
            chrf = sacrebleu.corpus_chrf(hyps, [val_de]).score
            log.append("val_history", [epoch, round(chrf, 2)])
            print(f"epoch {epoch} step {step} loss {loss.item():.3f} val chrF {chrf:.2f}", flush=True)
            if chrf > best:
                best, bad_epochs = chrf, 0
                torch.save({"model": model.state_dict(), "cfg": cfg, "vocab_size": sp.get_piece_size()}, CKPT / "best.pt")
            else:
                bad_epochs += 1
            if bad_epochs >= cfg["patience"] or max_steps:
                break
        log.set(train_seconds=round(time.time() - t0, 1), best_val_chrf=round(best, 2), epochs_run=epoch, steps=step,
                peak_vram_mb=round(torch.cuda.max_memory_allocated() / 2**20))


if __name__ == "__main__":
    main()
