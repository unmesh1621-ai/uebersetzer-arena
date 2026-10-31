"""LoRA supervised fine-tuning of a decoder-only chat model (LLaMA or Gemma) for EN->DE translation."""
import argparse
import time

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from src.common import PATHS, SEED, RunLog, count_params, load_config, read_lines, set_seed
from src.prompts import build_messages

# Llama 3.x templates insert today's date; pin it so training and inference prompts are identical.
TEMPLATE_KWARGS = {"date_string": "26 Jul 2024"}


def render(tok, messages, add_generation_prompt: bool) -> str:
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt, **TEMPLATE_KWARGS)


def encode_example(tok, src: str, tgt: str, max_len: int) -> dict:
    """Token ids for prompt+answer; labels are -100 everywhere except the answer and its end-of-turn token."""
    prompt = render(tok, build_messages(src), add_generation_prompt=True)
    full = render(tok, build_messages(src, tgt=tgt), add_generation_prompt=False)
    assert full.startswith(prompt), "chat template prompt is not a prefix of the full conversation"
    prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    answer_ids = tok(full[len(prompt):], add_special_tokens=False)["input_ids"]
    input_ids = (prompt_ids + answer_ids)[:max_len]
    labels = ([-100] * len(prompt_ids) + answer_ids)[:max_len]
    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}


def collate(tok):
    def fn(batch):
        width = max(len(b["input_ids"]) for b in batch)
        out = {"input_ids": [], "attention_mask": [], "labels": []}
        for b in batch:
            pad = width - len(b["input_ids"])
            out["input_ids"].append(b["input_ids"] + [tok.pad_token_id] * pad)
            out["attention_mask"].append(b["attention_mask"] + [0] * pad)
            out["labels"].append(b["labels"] + [-100] * pad)
        return {k: torch.tensor(v) for k, v in out.items()}

    return fn


def load_tokenizer(name):
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["llama", "gemma"], required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = load_config(f"configs/{args.model}_lora.yaml")
    set_seed(SEED)
    out_dir = PATHS["checkpoints"] / f"{args.model}_lora"

    tr_en, tr_de = read_lines(PATHS["data"] / "train.en"), read_lines(PATHS["data"] / "train.de")
    va_en, va_de = read_lines(PATHS["data"] / "val.en")[:500], read_lines(PATHS["data"] / "val.de")[:500]
    if args.smoke:
        tr_en, tr_de, va_en, va_de = tr_en[:200], tr_de[:200], va_en[:16], va_de[:16]

    tok = load_tokenizer(cfg["model_name"])

    def to_ds(en, de):
        rows = [encode_example(tok, s, t, cfg["max_len"]) for s, t in zip(en, de)]
        return Dataset.from_list([r for r in rows if any(l != -100 for l in r["labels"])])

    train_ds, val_ds = to_ds(tr_en, tr_de), to_ds(va_en, va_de)

    with RunLog(f"train_{args.model}_lora") as log:
        model = AutoModelForCausalLM.from_pretrained(cfg["model_name"], torch_dtype=torch.bfloat16, attn_implementation="eager"
                                                     if args.model == "gemma" else "sdpa").to("cuda")
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        lc = cfg["lora"]
        model = get_peft_model(model, LoraConfig(r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"],
                                                 target_modules=lc["targets"], task_type="CAUSAL_LM"))
        total, trainable = count_params(model)
        log.set(params_total=total, params_trainable=trainable, train_examples=len(train_ds), smoke=args.smoke,
                model_name=cfg["model_name"])
        targs = TrainingArguments(
            output_dir=str(out_dir / "runs"),
            per_device_train_batch_size=cfg["batch_size"],
            per_device_eval_batch_size=cfg["batch_size"],
            gradient_accumulation_steps=cfg["grad_accum"],
            learning_rate=cfg["lr"],
            lr_scheduler_type="cosine",
            warmup_steps=cfg["warmup_ratio"],  # v5: a float in [0,1) is a ratio of total steps
            num_train_epochs=cfg["epochs"],
            max_steps=20 if args.smoke else -1,
            bf16=True,
            logging_steps=5 if args.smoke else 20,
            eval_strategy="no",
            save_strategy="no",
            report_to=[],
            seed=SEED,
            dataloader_num_workers=0,
            remove_unused_columns=False,
        )
        if args.smoke:
            targs.gradient_accumulation_steps = 1
        trainer = Trainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds, data_collator=collate(tok))
        t0 = time.time()
        result = trainer.train()
        train_seconds = time.time() - t0
        val_loss = trainer.evaluate()["eval_loss"]
        model.save_pretrained(str(out_dir))
        tok.save_pretrained(str(out_dir))
        for h in trainer.state.log_history:
            if "loss" in h:
                log.append("loss_history", [h["step"], round(h["loss"], 4)])
        log.set(train_seconds=round(train_seconds, 1), final_train_loss=round(result.training_loss, 4),
                val_loss=round(val_loss, 4), steps=trainer.state.global_step,
                peak_vram_mb=round(torch.cuda.max_memory_allocated() / 2**20))


if __name__ == "__main__":
    main()
