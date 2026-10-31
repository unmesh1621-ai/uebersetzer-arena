"""Fine-tune mT5-small on the EN->DE training split with Seq2SeqTrainer."""
import argparse
import time

import numpy as np
import sacrebleu
import torch
from datasets import Dataset
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainer,
                          Seq2SeqTrainingArguments)

from src.common import PATHS, SEED, RunLog, count_params, load_config, read_lines, set_seed

CKPT = PATHS["checkpoints"] / "mt5"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/mt5.yaml")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", help="Trainer checkpoint dir to resume from")
    ap.add_argument("--prior-seconds", type=float, default=0.0,
                    help="training+eval wall-clock already spent before --resume (added to train_seconds)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(SEED)

    data = {s: (read_lines(PATHS["data"] / f"{s}.en"), read_lines(PATHS["data"] / f"{s}.de")) for s in ("train", "val")}
    if args.smoke:
        data = {"train": (data["train"][0][:200], data["train"][1][:200]), "val": (data["val"][0][:32], data["val"][1][:32])}

    tok = AutoTokenizer.from_pretrained(cfg["model_name"])
    L = cfg["max_len"]

    def encode(batch):
        enc = tok([cfg["prefix"] + s for s in batch["en"]], max_length=L, truncation=True)
        enc["labels"] = tok(text_target=batch["de"], max_length=L, truncation=True)["input_ids"]
        return enc

    ds = {s: Dataset.from_dict({"en": en, "de": de}).map(encode, batched=True, remove_columns=["en", "de"])
          for s, (en, de) in data.items()}

    def compute_metrics(pred):
        preds = np.where(pred.predictions < 0, tok.pad_token_id, pred.predictions)
        hyps = tok.batch_decode(preds, skip_special_tokens=True)
        return {"chrf": sacrebleu.corpus_chrf([h.strip() for h in hyps], [data["val"][1]]).score}

    with RunLog("train_mt5") as log:
        model = AutoModelForSeq2SeqLM.from_pretrained(cfg["model_name"], torch_dtype=torch.float32)
        total, trainable = count_params(model)
        log.set(params_total=total, params_trainable=trainable, smoke=args.smoke)
        targs = Seq2SeqTrainingArguments(
            output_dir=str(CKPT / "runs"),
            per_device_train_batch_size=cfg["batch_size"],
            per_device_eval_batch_size=16,
            gradient_accumulation_steps=cfg["grad_accum"],
            learning_rate=cfg["lr"],
            lr_scheduler_type="linear",
            optim=cfg.get("optim", "adamw_torch"),
            num_train_epochs=cfg["epochs"],
            max_steps=20 if args.smoke else -1,
            bf16=True,
            eval_strategy="steps" if args.smoke else "epoch",
            eval_steps=20 if args.smoke else None,
            save_strategy="steps" if args.smoke else "epoch",
            save_steps=20 if args.smoke else None,
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="chrf",
            greater_is_better=True,
            predict_with_generate=True,
            generation_max_length=L,
            generation_num_beams=1,
            logging_steps=5 if args.smoke else 50,
            report_to=[],
            seed=SEED,
            dataloader_num_workers=0,
        )
        trainer = Seq2SeqTrainer(
            model=model, args=targs, train_dataset=ds["train"], eval_dataset=ds["val"], processing_class=tok,
            data_collator=DataCollatorForSeq2Seq(tok, model=model), compute_metrics=compute_metrics,
        )
        t0 = time.time()
        trainer.train(resume_from_checkpoint=args.resume)
        train_seconds = time.time() - t0 + args.prior_seconds
        log.set(resumed_from=args.resume, prior_seconds=args.prior_seconds)
        trainer.save_model(str(CKPT / "best"))
        tok.save_pretrained(str(CKPT / "best"))
        for h in trainer.state.log_history:
            if "loss" in h:
                log.append("loss_history", [h["step"], round(h["loss"], 4)])
            if "eval_chrf" in h:
                log.append("val_history", [h.get("epoch"), round(h["eval_chrf"], 2)])
        log.set(train_seconds=round(train_seconds, 1), best_val_chrf=round(trainer.state.best_metric or 0, 2),
                epochs_run=cfg["epochs"], steps=trainer.state.global_step,
                peak_vram_mb=round(torch.cuda.max_memory_allocated() / 2**20))


if __name__ == "__main__":
    main()
