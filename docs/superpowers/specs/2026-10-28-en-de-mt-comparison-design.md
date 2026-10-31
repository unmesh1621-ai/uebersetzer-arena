# Design Spec — English→German MT: Transformer vs mT5 vs LLaMA vs Gemma

- **Course:** Foundations of Generative AI (ICT4403), Assignment 3
- **Date:** 2026-10-28 · **Deadline:** 2 days (finish by 2026-10-30)
- **Status:** Approved design, pending written-spec review
- **Supersedes:** language/time assumptions in `PLAN.md` (Telugu, 1 week)

---

## 1. Goal and success criteria

Build four machine-translation systems for **English → German**, translate the same test set with all of them under comparable settings, and deliver a GitHub repo + README + submission docx that satisfy every item of the brief.

**Done when:**
1. All 6 result rows exist in `results/metrics.csv` with BLEU, chrF, total/trainable params, training time, inference time.
2. Temperature sweep results + plot exist for all 6 systems.
3. Responsible-AI probe outputs are labelled and the ethics analysis is written.
4. README contains all 8 required sections; report notebook runs top-to-bottom from `results/` without training.
5. `YourName_RegNum_FGAI_Assignment3.docx` + `.pdf` are generated (user supplies name/reg no.).
6. Repo is committed locally and ready for the user to push to a **public** GitHub repo.

**Out of scope:** models >1.3B params, hyper-parameter searches, human evaluation beyond the RAI review, LMS upload and GitHub push (user does these).

## 2. Constraints

| Constraint | Value |
|---|---|
| Hardware | RTX 3080 Laptop, 8 GB VRAM, bf16 capable; Windows 11 |
| Python | 3.13 available; use a venv (fallback 3.12 if a wheel is missing) |
| Time | 2 days; GPU training runs overnight via a queue |
| Gated assets | Llama-3.2-1B-Instruct, Gemma-3-1B-it, FLORES+ — user accepts licences and runs `hf auth login` themselves |
| Seed | 42 everywhere |

## 3. Systems (result rows)

| Row id | Model | Architecture | Adaptation |
|---|---|---|---|
| `transformer` | Hand-written PyTorch Transformer (~15M) | Encoder–decoder | Trained from scratch |
| `mt5` | `google/mt5-small` (~300M) | Encoder–decoder | Full fine-tune |
| `llama_prompt` | `meta-llama/Llama-3.2-1B-Instruct` (1.24B) | Decoder-only | 5-shot prompting |
| `llama_lora` | same | Decoder-only | LoRA SFT |
| `gemma_prompt` | `google/gemma-3-1b-it` (1.0B) | Decoder-only | 5-shot prompting |
| `gemma_lora` | same | Decoder-only | LoRA SFT |

Extra column for encoder–decoder rows: beam-4 scores.

## 4. Data

- **Train/val pool:** WMT14 en–de (`wmt/wmt14`, config `de-en`), training split.
- **Cleaning:** NFC normalise; strip whitespace; drop empty/duplicate pairs; drop if either side >100 words; drop if word-length ratio outside [1/3, 3]; drop if German side fails language-ID (`langdetect`, seeded); drop any sentence present in FLORES dev/devtest (leakage guard).
- **Splits (seeded random sample after cleaning):** train **30,000**, val **2,000**, held-out test **1,000** (saved, not scored — satisfies "divide into train/val/test").
- **Official test set:** FLORES+ devtest `eng_Latn`→`deu_Latn` (1,012 sentences, `openlanguagedata/flores_plus`).
- **Temperature subset:** first 300 FLORES devtest sentences.
- **Files:** `data/{train,val,test,flores}.{en,de}` one sentence per line; `data/stats.json` (counts after each filter, length stats, tokens/sentence per tokenizer).

## 5. Components

Each unit communicates only through files, so every step is re-runnable on its own.

| Unit | Responsibility | Inputs → Outputs |
|---|---|---|
| `src/common.py` | seeding, paths, YAML config loading, run-log writer (GPU, versions, timings), param counting | — |
| `src/data.py` | download, clean, split, stats | HF datasets → `data/*` |
| `src/transformer_scratch/model.py` | sinusoidal PE, encoder–decoder (`nn.TransformerEncoderLayer/DecoderLayer`), masks, tied embeddings | — |
| `src/transformer_scratch/train.py` | SentencePiece (shared unigram 16k) training, batching by tokens, label-smoothed loss, Noam schedule, early stopping on val chrF | `data/` → `checkpoints/transformer/`, `results/logs/transformer.json` |
| `src/transformer_scratch/decode.py` | greedy, beam search, temperature sampling | — |
| `src/train_mt5.py` | Seq2SeqTrainer fine-tune | `data/` → `checkpoints/mt5/`, log |
| `src/train_lora.py --model {llama,gemma}` | PEFT LoRA SFT with chat template, loss on assistant tokens only | `data/` → `checkpoints/{model}_lora/`, log |
| `src/prompts.py` | shared instruction + 5 fixed few-shot examples (from val) + chat-template builder + output post-processing | — |
| `src/translate.py --system <row> [--temp T] [--beam 4] [--subset N]` | unified loading, batched generation, timing | `data/flores.en` → `results/predictions/<row>[_suffix].de`, `results/timing/<row>[_suffix].json` |
| `src/evaluate.py` | sacrebleu BLEU (13a) + chrF with signatures; merges params, train time, timing, postproc rate | → `results/metrics.csv` |
| `src/temperature_sweep.py` | runs translate for T ∈ {0.3,0.7,1.0,1.3} on 300-sentence subset, scores | → `results/temp_sweep.csv` |
| `src/rai_probe.py` | translates probe set with all systems; computes automatic flags on FLORES outputs; writes the review sheet (back-translation + labels are filled in during the review step, not by a model) | → `samples/rai_probe.csv`, `samples/rai_review.csv` |
| `src/plots.py` | all figures | → `plots/*.png` |
| `run_all.ps1` | overnight queue: each training step runs independently, logs to `results/logs/`, continues on failure | — |
| `notebooks/report.ipynb` | read-only: tables, plots, samples, analysis | `results/`, `plots/`, `samples/` |

`configs/<row>.yaml` holds every hyper-parameter; scripts read nothing hard-coded.

## 6. Training settings

| | Transformer | mT5-small | LoRA (LLaMA & Gemma, identical) |
|---|---|---|---|
| Size | 4 enc + 4 dec layers, d=256, 4 heads, FFN 1024, dropout 0.3 | as released | r=16, α=32, dropout 0.05, targets q,k,v,o,gate,up,down |
| Optimiser | Adam β(0.9,0.98), Noam warmup 4k | AdamW lr 5e-4, linear decay | AdamW lr 2e-4, cosine, warmup 3% |
| Batch | ~4k tokens | effective 32 | effective 32 (micro 4–8 × accumulation) |
| Length | ≤128 tokens | ≤128 | ≤256 (prompt+answer) |
| Epochs | ≤30, early stop patience 5 (val chrF) | 3 | 1 |
| Other | label smoothing 0.1, bf16 | bf16, prefix `translate English to German: ` | bf16 base (no quantisation), gradient checkpointing |
| Est. time | ~45 min | ~1 h | ~45–60 min each |

**Prompt (shared by LoRA SFT and prompting):** user turn `Translate the following English sentence into German. Output only the German translation.\nEnglish: {src}` → assistant turn `{tgt}`. Prompting rows prepend 5 fixed val examples as prior user/assistant turns.

**OOM fallback order:** lower micro-batch → raise accumulation → 4-bit QLoRA (documented in README).

## 7. Inference and evaluation

- **Main setting (all rows):** greedy, `max_new_tokens=256`, batch 16, same GPU, models in bf16.
- **Enc–dec extra:** beam 4.
- **Decoder post-processing:** keep first non-empty line, strip `German:` echo / quotes; record % of outputs changed.
- **Timing:** one warm-up batch excluded; `torch.cuda.synchronize()` around timed loop; report total s, ms/sentence, generated tokens/s, peak VRAM.
- **Metrics:** sacrebleu BLEU (`13a`) and chrF (char order 6, word order 0), corpus-level, signatures saved.
- **Params:** total and trainable (`requires_grad`) per row; prompting rows trainable = 0.
- **Training time:** wall-clock from run logs; prompting rows = 0.
- **Tokenizer analysis:** mean tokens per FLORES sentence (EN and DE) for SentencePiece-16k, mT5, LLaMA, Gemma tokenizers.

## 8. Temperature analysis

300 FLORES sentences × T ∈ {0.3, 0.7, 1.0, 1.3} (greedy = T 0 point), `top_p=1.0`, `top_k=0`, seed 42, all 6 rows. Outputs: `results/temp_sweep.csv`, `plots/temperature.png` (BLEU & chrF vs T), and 3 sentences shown across temperatures in the README.

## 9. Responsible-AI analysis

- **Probe set (~40 hand-written EN sentences, `data/rai_probe.en`):** occupational gender (doctor, nurse, engineer, secretary, with/without pronouns), formal vs informal address (Sie/du), idioms & culture-specific terms, named entities/numbers/dates/units, long compounds & verb-final clauses, mildly offensive/sensitive phrasing.
- **Automatic flags on FLORES outputs:** length ratio outliers, non-German language-ID, untranslated English spans, repetition loops, number mismatch source vs output.
- **Review:** Claude labels ~30 flagged + probe outputs in `samples/rai_review.csv` (columns: source, system, output, back-translation, category, label, note); user spot-checks. README states that review was done by the author with AI assistance.
- **Write-up:** per category — observed examples per system, which architecture/adaptation is more prone, mitigations (gender-aware prompting/rewriting, constrained decoding & NE/number checks, QE-based filtering, human-in-the-loop for sensitive domains, data documentation, user-facing disclaimers).

## 10. Deliverables and repo layout

The repo root is this `IA3/` folder (pushed as GitHub repo `fgai-ia3-mt-en-de`); the assignment brief PDF and `.remember/` are git-ignored.

```
fgai-ia3-mt-en-de/
├── README.md            # 8 required sections
├── requirements.txt     # pinned versions
├── run_all.ps1
├── configs/  src/  notebooks/report.ipynb
├── data/                # small text splits committed (<10 MB); script regenerates
├── results/  plots/  samples/
└── docs/                # spec, workflow plan, research notes
```
Checkpoints and adapters are **git-ignored** (size); README says how to regenerate.

README sections: problem & objective · dataset + source links (WMT14, FLORES+) · models · execution steps & dependencies · results table · comparison & evaluation (enc–dec vs dec-only: size, pretraining, tokenization, adaptation) · sampling-temperature analysis · responsible AI · conclusion naming the best model.

Submission doc: `YourName_RegNum_FGAI_Assignment3.docx` → PDF with name, reg no., assignment title, system title ("Comparative English–German Machine Translation: Transformer, mT5, LLaMA and Gemma"), dataset names + links, repo link.

## 11. Error handling and verification

- Every script fails loudly on missing inputs and writes a run log with status.
- **Smoke test before overnight queue:** each training script runs 20 steps on 200 pairs and each system translates 16 sentences; all must succeed.
- `run_all.ps1` continues past a failed step and records the failure; morning check reads `results/logs/*.json`.
- Sanity checks before reporting: prediction file line count == 1,012; no empty-output rate >5% without explanation; BLEU recomputed from saved files matches `metrics.csv`.

## 12. Risks

| Risk | Mitigation |
|---|---|
| Licence approval delay | Request all three at the very start of Day 1; prompting rows can run as soon as access lands |
| OOM on 8 GB | fallback order in §6 |
| Windows package issues (bitsandbytes, torch for 3.13) | bf16 LoRA needs no bitsandbytes; venv on 3.12 if needed |
| Scratch Transformer scores very low on 30k pairs | expected; discussed as data-hunger of from-scratch models |
| Decoder models add chatter | post-processing + reported rate |
