# EN→DE MT Comparison (Transformer / mT5 / LLaMA / Gemma) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, run and compare six English→German MT systems on FLORES+ devtest and package results into a GitHub-ready repo, README and submission docx within 2 days.

**Architecture:** File-based pipeline — each `src/` script reads `data/` or `results/` and writes files, so any step re-runs alone. A hand-written PyTorch Transformer plus Hugging Face models (mT5 fine-tune, LoRA/prompting for LLaMA & Gemma) all feed one `translate.py` and one `evaluate.py`, guaranteeing comparable settings. Training runs overnight through `run_all.ps1`.

**Tech Stack:** Python 3.13 (fallback 3.12) venv, PyTorch (CUDA), transformers, peft, datasets, sentencepiece, sacrebleu, langdetect, pyyaml, pandas, matplotlib, pytest, jupyter, python-docx.

**Spec:** `docs/superpowers/specs/2026-10-28-en-de-mt-comparison-design.md`

## Global Constraints

- Language pair English→German; test set FLORES+ devtest `eng_Latn`→`deu_Latn` (1,012 lines).
- Train/val/test from WMT14 `de-en`: 30,000 / 2,000 / 1,000 after cleaning; seed 42 everywhere.
- Hardware: RTX 3080 Laptop 8 GB, bf16; Windows 11 (PowerShell for the queue).
- Row ids exactly: `transformer`, `mt5`, `llama_prompt`, `llama_lora`, `gemma_prompt`, `gemma_lora`.
- Models: `google/mt5-small`, `meta-llama/Llama-3.2-1B-Instruct`, `google/gemma-3-1b-it`.
- Main decoding: greedy, `max_new_tokens=256`, batch 16; extra beam 4 for `transformer`, `mt5`.
- Temperatures {0.3, 0.7, 1.0, 1.3}, `top_p=1.0`, `top_k=0`, first 300 FLORES lines.
- BLEU tokenizer `13a`; chrF char order 6, word order 0.
- All hyper-parameters live in `configs/<row>.yaml`; nothing hard-coded in scripts.
- Checkpoints/adapters git-ignored; data text files committed.
- User does: HF licences + `hf auth login`, GitHub push, LMS upload, name/reg no. Claude never handles tokens.

## Review Focus

1. **Decoder models emit chatter/echo or multiple lines** → `postprocess()` must return one clean German line; tested in Task 5.
2. **A prediction file with fewer/more lines than the source** (empty outputs, newlines inside outputs) → `translate.py` must write exactly one line per input; tested in Task 8.
3. **Test-set leakage** (FLORES sentence present in WMT sample) → removed by `clean_pairs`; tested in Task 2.
4. **Padding side wrong for decoder-only batched generation** (right-padding corrupts outputs) → left padding asserted in Task 8 test.
5. **LoRA loss computed on prompt tokens** → label mask must be -100 on everything except assistant answer; tested in Task 7.

---

## Workflow & Procedures (applies to every task)

**Two-day timeline**

| When | Tasks | Gate to pass before moving on |
|---|---|---|
| Day 1 AM | 0 Environment, 1 Common, 2 Data | `pytest` green; `data/stats.json` shows 30000/2000/1000/1012 |
| Day 1 PM | 3–8 Models, prompts, LoRA, translate | every unit test green; smoke run of every training script + 16-line translate per row |
| Day 1 night | 9 Overnight queue | `run_all.ps1` launched; first 5 min of logs checked |
| Day 2 AM | 10 Evaluate, 11 Temperature sweep | 6 rows in `metrics.csv`, line-count checks pass |
| Day 2 midday | 12 RAI probe + review | user spot-check done (~20 min) |
| Day 2 PM | 13 Plots, 14 Notebook, 15 README, 16 Docx | notebook runs top-to-bottom; README has all 8 sections |

**Per-task procedure**
1. Read the task's Interfaces block; touch only listed files.
2. TDD for pure logic: failing test → run (FAIL) → implement → run (PASS).
3. GPU scripts: run the `--smoke` variant before any full run.
4. Every run writes `results/logs/<name>.json` with `status`, timings, GPU name, package versions.
5. Commit after each task: `git commit -m "<type>: <task summary>"` ending with the attribution line.
6. Never report a number that is not read from a file written by a run (verification-before-completion).

**Failure procedure:** OOM → lower micro-batch, raise grad accumulation, then 4-bit QLoRA (log it). Any other failure → superpowers:systematic-debugging; do not edit configs to "make numbers look right".

**Commands** (from repo root, venv active): tests `python -m pytest -q`; scripts `python -m src.<module> --config configs/<row>.yaml [--smoke]`.

---

### Task 0: Environment

**Files:**
- Create: `requirements.txt`, `tests/test_env.py`

- [ ] **Step 1:** Create venv `python -m venv .venv`; activate (`.venv\Scripts\Activate.ps1`).
- [ ] **Step 2:** Install torch CUDA wheel from the official index (`pip install torch --index-url https://download.pytorch.org/whl/cu126` or newest CUDA listed for Python 3.13); if no wheel, recreate venv with Python 3.12.
- [ ] **Step 3:** Write `requirements.txt` (transformers, peft, datasets, sentencepiece, sacrebleu, langdetect, pyyaml, pandas, matplotlib, pytest, jupyter, python-docx, accelerate) and `pip install -r requirements.txt`; then `pip freeze > requirements.lock.txt`.
- [ ] **Step 4:** Test `tests/test_env.py::test_cuda` asserts `torch.cuda.is_available()` and `torch.cuda.is_bf16_supported()`. Run `python -m pytest tests/test_env.py -q` → PASS.
- [ ] **Step 5 (USER):** Accept licences on the HF pages of `meta-llama/Llama-3.2-1B-Instruct`, `google/gemma-3-1b-it`, `openlanguagedata/flores_plus`; run `hf auth login`. Verify: `hf auth whoami` prints the username.
- [ ] **Step 6:** Commit `requirements*.txt tests/test_env.py`.

### Task 1: Common utilities + configs

**Files:**
- Create: `src/__init__.py`, `src/common.py`, `configs/{transformer,mt5,llama_prompt,llama_lora,gemma_prompt,gemma_lora}.yaml`, `tests/test_common.py`

**Interfaces — Produces:**
- `set_seed(seed: int) -> None`
- `load_config(path: str) -> dict` (YAML → dict; adds key `row` = file stem)
- `count_params(model) -> tuple[int, int]` (total, trainable)
- `RunLog(name: str)` context manager; `.set(**kv)`; `.append(key, value)` (used by every training script for `loss_history` as `[step, loss]` pairs and `val_history` as `[epoch_or_step, metric]`, read by Task 13); on exit writes `results/logs/<name>.json` with `status` (`ok`/`failed`), `start`, `end`, `seconds`, `gpu`, `versions`, plus set keys; re-raises exceptions.
- `PATHS`: dict with `data`, `results`, `predictions`, `timing`, `logs`, `checkpoints`, `plots`, `samples` (all `pathlib.Path`, created on import).
- `read_lines(path) -> list[str]`, `write_lines(path, lines) -> None` (UTF-8, `\n`, one per line).

- [ ] **Step 1:** Tests: `test_count_params` (a `nn.Linear(4,2)` with bias frozen → `(10, 8)`); `test_runlog_failed` (raise inside → JSON has `status == "failed"` and exception re-raised); `test_write_read_roundtrip` (lines with umlauts survive).
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement `src/common.py`. **Step 4:** Run → PASS.
- [ ] **Step 5:** Write the six YAMLs with the values from spec §6 (keys: `model_name`, `max_len`, `lr`, `epochs`, `batch_size`, `grad_accum`, `lora:{r,alpha,dropout,targets}`, `prefix`, `shots`). Prompt rows hold only `model_name` and `shots: 5`.
- [ ] **Step 6:** Commit.

### Task 2: Data pipeline

**Files:**
- Create: `src/data.py`, `tests/test_data.py`

**Interfaces — Produces:**
- `clean_pairs(pairs: list[tuple[str,str]], blocklist: set[str]) -> tuple[list[tuple[str,str]], dict]` — returns kept pairs and per-filter drop counts (keys `empty`, `duplicate`, `too_long`, `ratio`, `langid`, `leak`).
- CLI `python -m src.data` → `data/{train,val,test}.{en,de}`, `data/flores.{en,de}`, `data/stats.json`.

- [ ] **Step 1:** Tests on `clean_pairs`: drops `("", "Hallo")`; drops exact duplicate; drops a 101-word source; drops ratio 1:4; drops German side `"This is English text clearly"` (langid); drops pair whose `en` is in `blocklist`; keeps `("Good morning.", "Guten Morgen.")`; NFC-normalises decomposed `"ü"` to `"ü"`.
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement: stream WMT14 train (`load_dataset("wmt/wmt14","de-en",split="train",streaming=True)`), take first 300k, shuffle seed 42, clean with FLORES dev+devtest English as blocklist (`langdetect` with `DetectorFactory.seed = 0`), slice 30000/2000/1000. Load FLORES+ devtest `eng_Latn` and `deu_Latn` aligned by `id`. Stats: counts, drop counts, mean/95th-pct word lengths. **Step 4:** Run tests → PASS.
- [ ] **Step 5:** Run `python -m src.data`. Verify: `stats.json` counts 30000/2000/1000/1012; `wc -l data/flores.de` = 1012.
- [ ] **Step 6:** Commit code + `data/` text files.

### Task 3: Scratch Transformer model + decoding

**Files:**
- Create: `src/transformer_scratch/{__init__,model,decode}.py`, `tests/test_transformer.py`

**Interfaces — Produces:**
- `Seq2SeqTransformer(vocab_size:int, d_model=256, nhead=4, num_layers=4, dim_ff=1024, dropout=0.3, pad_id=0)`; `forward(src: LongTensor[B,S], tgt_in: LongTensor[B,T]) -> logits[B,T,V]`; `encode(src) -> memory`, `decode_step(memory, src_mask, ys) -> logits[B,V]`. Shared, tied embeddings × √d_model; sinusoidal PE; `batch_first=True`; causal + padding masks.
- `greedy_decode(model, src, bos, eos, max_len) -> list[list[int]]`
- `beam_decode(model, src, bos, eos, max_len, beam=4, alpha=0.6) -> list[list[int]]` (length-normalised)
- `sample_decode(model, src, bos, eos, max_len, temperature: float, generator) -> list[list[int]]`

- [ ] **Step 1:** Tests: output shape `[2,5,V]`; changing token at position t+1 of `tgt_in` does not change logits at ≤t (causality); padding src tokens does not change logits of non-pad batch item; overfit test: 200 Adam steps on 4 copy-pairs → `greedy_decode` reproduces them; `beam_decode(beam=1)` == `greedy_decode`; outputs stop at `eos` and exclude it.
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Run → PASS. **Step 5:** Commit.

### Task 4: Scratch Transformer training

**Files:**
- Create: `src/transformer_scratch/train.py`

**Interfaces:**
- Consumes: Task 1 `load_config`, `RunLog`, `count_params`; Task 3 model/decoders.
- Produces: `checkpoints/transformer/{best.pt,spm.model}`; `load_transformer(ckpt_dir) -> (model, sp)` used by Task 8; log `results/logs/train_transformer.json` with `train_seconds`, `best_val_chrf`, `epochs_run`, `params_total`, `params_trainable`.

- [ ] **Step 1:** Implement: SentencePiece unigram 16k shared vocab on train en+de (`pad=0,unk=1,bos=2,eos=3`); token-budget batching (~4k tokens, sorted by length); label-smoothed CE (0.1, ignore pad); Adam β(0.9,0.98) with Noam schedule warmup 4000; bf16 autocast; per-epoch val chrF via greedy on val; save best; early stop patience 5; max 30 epochs; `--smoke` = 200 pairs, 20 steps, 1 val batch.
- [ ] **Step 2:** Run `python -m src.transformer_scratch.train --config configs/transformer.yaml --smoke`. Expected: completes, log `status: ok`, `best.pt` exists.
- [ ] **Step 3:** Commit.

### Task 5: Shared prompts

**Files:**
- Create: `src/prompts.py`, `tests/test_prompts.py`

**Interfaces — Produces:**
- `INSTRUCTION = "Translate the following English sentence into German. Output only the German translation.\nEnglish: {src}"`
- `few_shot_examples(n=5) -> list[tuple[str,str]]` (first 5 val pairs with 5–25 words)
- `build_messages(src: str, shots: list[tuple[str,str]] | None = None, tgt: str | None = None) -> list[dict]` — alternating user/assistant turns; answer appended as final assistant turn when `tgt` given.
- `postprocess(text: str) -> tuple[str, bool]` — returns cleaned line and whether it changed.

- [ ] **Step 1:** Tests: `build_messages("Hi", shots=[("a","b")])` has 3 messages, roles user/assistant/user, last content == `INSTRUCTION.format(src="Hi")`; with `tgt="Hallo"` → 2 messages, last is assistant `"Hallo"`; `postprocess("German: Guten Tag.\n\nNote: ...")` → `("Guten Tag.", True)`; `postprocess('"Hallo Welt"')` → `("Hallo Welt", True)`; `postprocess("Hallo")` → `("Hallo", False)`; `postprocess("")` → `("", False)`.
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Run → PASS. **Step 5:** Commit.

### Task 6: mT5 fine-tuning

**Files:**
- Create: `src/train_mt5.py`

**Interfaces — Produces:** `checkpoints/mt5/best/` (HF format + tokenizer); log `results/logs/train_mt5.json` (same keys as Task 4).

- [ ] **Step 1:** Implement with `Seq2SeqTrainer`: prefix from config, max len 128 both sides, AdamW lr 5e-4 linear decay, effective batch 32, 3 epochs, bf16 (not fp16 — mT5 NaNs in fp16), eval each epoch with chrF on val (`predict_with_generate`, greedy), `load_best_model_at_end`, `save_total_limit=1`. Time `trainer.train()` only. `--smoke`: 200 pairs, 20 steps.
- [ ] **Step 2:** Run smoke → log `status: ok`, `best/` exists. **Step 3:** Commit.

### Task 7: LoRA SFT for LLaMA and Gemma

**Files:**
- Create: `src/train_lora.py`, `tests/test_lora_labels.py`

**Interfaces:**
- Consumes: Task 5 `build_messages`.
- Produces: `encode_example(tokenizer, src, tgt, max_len) -> dict(input_ids, attention_mask, labels)`; `checkpoints/{llama,gemma}_lora/` adapter; log `results/logs/train_{llama,gemma}_lora.json` with `params_total`, `params_trainable`, `train_seconds`, `final_train_loss`, `val_loss`.

- [ ] **Step 1:** Test (uses the real LLaMA tokenizer, skip if not logged in): for `("Hello.","Hallo.")`, decoded tokens where `labels != -100` equal `"Hallo."` + the model's end-of-turn token; all other positions are -100; `len(input_ids) <= max_len`.
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement: `apply_chat_template` on prompt-only messages (`add_generation_prompt=True`) and on full messages; labels masked for the prompt-prefix length; PEFT `LoraConfig` from YAML; bf16 base, gradient checkpointing, `use_cache=False`; HF `Trainer` with dynamic padding collator (pad labels with -100); lr 2e-4 cosine warmup 3%, 1 epoch, effective batch 32; eval loss on 500 val pairs at end. `--model {llama,gemma}` picks config. `--smoke`: 200 pairs, 20 steps.
- [ ] **Step 4:** Run test → PASS. Smoke for both models → `status: ok`, adapter saved, peak VRAM logged < 8 GB.
- [ ] **Step 5:** Commit.

### Task 8: Unified translation + timing

**Files:**
- Create: `src/translate.py`, `tests/test_translate.py`

**Interfaces:**
- Consumes: `load_transformer` (Task 4), mT5 checkpoint (Task 6), adapters (Task 7), `build_messages`/`postprocess`/`few_shot_examples` (Task 5).
- Produces: `load_system(row: str) -> Translator`; `Translator.translate(srcs: list[str], temperature: float = 0.0, beam: int = 1, batch_size: int = 16, max_new_tokens: int = 256, seed: int = 42) -> list[str]`; CLI `python -m src.translate --system <row> [--temp T] [--beam 4] [--subset N] [--input data/flores.en]` → `results/predictions/<row><suffix>.de` and `results/timing/<row><suffix>.json` (`n`, `total_s`, `ms_per_sent`, `gen_tokens`, `tok_per_s`, `peak_vram_mb`, `postproc_changed`). Suffix: `""`, `_beam4`, `_t0.7`.
- Decoder-only rows: tokenizer `padding_side="left"`, chat template with `add_generation_prompt=True`, decode only new tokens, then `postprocess`. LoRA rows load base + `PeftModel.from_pretrained` and `merge_and_unload()`. Temperature 0 → `do_sample=False`; >0 → `do_sample=True, top_p=1.0, top_k=0`, `torch.manual_seed(seed)`.
- Timing: first batch run once untimed; `torch.cuda.synchronize()` before/after timed loop; `torch.cuda.reset_peak_memory_stats()` at start.

- [ ] **Step 1:** Tests (GPU, marked `slow`): `load_system("llama_prompt")` tokenizer has `padding_side == "left"`; translating `["Hello.", "", "Line one.\nLine two."]` returns exactly 3 strings, none containing `"\n"`; output file line count == input count.
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Run → PASS.
- [ ] **Step 5:** Smoke: for each of the 6 rows (using smoke checkpoints) `--subset 16` → 16 lines, timing JSON written.
- [ ] **Step 6:** Commit.

### Task 9: Overnight queue

**Files:**
- Create: `run_all.ps1`

- [ ] **Step 1:** Script runs in order, each wrapped so failure logs and continues: train transformer → train mt5 → train lora llama → train lora gemma → translate all 6 rows greedy → translate `transformer`, `mt5` with `--beam 4`. Stdout/stderr of each step to `results/logs/<step>.out.txt`. Writes `results/logs/queue_summary.txt` with each step's exit code.
- [ ] **Step 2:** Delete smoke checkpoints; launch in background. Check after 5 minutes: transformer log shows epoch progress, no errors.
- [ ] **Step 3:** Commit script.
- [ ] **Step 4 (morning):** Read `queue_summary.txt`; all exit codes 0; each `predictions/<row>.de` has 1012 lines. Re-run only failed steps.

### Task 10: Evaluation

**Files:**
- Create: `src/evaluate.py`, `tests/test_evaluate.py`

**Interfaces — Produces:** `score(hyps: list[str], refs: list[str]) -> dict(bleu, chrf, bleu_sig, chrf_sig)`; `tokenizer_fertility() -> pd.DataFrame` (mean tokens/sentence EN & DE for spm16k, mt5, llama, gemma); CLI writes `results/metrics.csv` (columns: `row, arch, adaptation, bleu, chrf, bleu_beam4, chrf_beam4, params_total, params_trainable, train_seconds, infer_total_s, ms_per_sent, tok_per_s, peak_vram_mb, postproc_pct`) and `results/fertility.csv`, `results/signatures.json`.

- [ ] **Step 1:** Tests: `score(["Guten Morgen ."],["Guten Morgen ."])["bleu"] == 100`; `score` raises `ValueError` when lengths differ; empty hypothesis gives `bleu == 0` and no crash.
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement (sacrebleu `BLEU(tokenize="13a")`, `CHRF(char_order=6, word_order=0)`). **Step 4:** Run → PASS.
- [ ] **Step 5:** Run CLI. Verify: 6 rows; recompute one row's BLEU by hand from files → matches CSV.
- [ ] **Step 6:** Commit code + `results/`.

### Task 11: Temperature sweep

**Files:**
- Create: `src/temperature_sweep.py`

**Interfaces:** Consumes `load_system`, `score`. Produces `results/temp_sweep.csv` (`row, temperature, bleu, chrf`; temperature 0 rows copied from greedy scores on the same 300 lines) and `results/predictions/<row>_t<T>.de`.

- [ ] **Step 1:** Implement: loop rows × {0.3,0.7,1.0,1.3} on first 300 FLORES lines; load each model once; seed 42.
- [ ] **Step 2:** Run (~1 h). Verify 30 rows (6 × 5) in CSV.
- [ ] **Step 3:** Commit.

### Task 12: Responsible-AI probe and review

**Files:**
- Create: `data/rai_probe.tsv` (columns `id, category, en`), `src/rai_probe.py`, `tests/test_rai_flags.py`

**Interfaces — Produces:** `flags(src: str, hyp: str) -> list[str]` from {`length_ratio`, `not_german`, `untranslated`, `repetition`, `number_mismatch`}; CLI writes `samples/rai_probe.csv` (probe × 6 rows outputs) and `samples/rai_review.csv` (`source, system, output, flags, category, back_translation, label, note`; last three empty).

- [ ] **Step 1:** Tests: `flags("I have 3 cats.", "Ich habe 4 Katzen.")` contains `number_mismatch`; `flags("Hello world.", "Hello world.")` contains `untranslated`; `flags("Hi.", "ja ja ja ja ja ja ja ja")` contains `repetition`; `flags("Good morning.", "Guten Morgen.") == []`.
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Run → PASS.
- [ ] **Step 5:** Write ~40 probe sentences across the 6 spec categories (occupational gender, Sie/du, idioms/culture, entities/numbers/dates, compounds/verb-final, sensitive). Run CLI.
- [ ] **Step 6 (Claude):** Fill `back_translation`, `label` (`ok`/`minor`/`major`), `note` for ~30 rows: all probe outputs of `mt5`, `llama_lora`, `gemma_lora` for gender + sensitive categories, plus flagged FLORES outputs.
- [ ] **Step 7 (USER):** Spot-check 10 labelled rows; corrections applied.
- [ ] **Step 8:** Write `docs/ethics_analysis.md` (per category: examples, which systems, mitigations from spec §9). Commit.

### Task 13: Plots

**Files:**
- Create: `src/plots.py`

**Produces:** `plots/scores.png` (BLEU & chrF grouped bars per row), `plots/quality_vs_params.png`, `plots/quality_vs_latency.png`, `plots/temperature.png` (chrF and BLEU vs T, one line per row), `plots/fertility.png`, `plots/loss_curves.png` (from training logs).

- [ ] **Step 1:** Implement (use the `dataviz` skill palette; consistent colour per row across all plots; log-x for params).
- [ ] **Step 2:** Run; verify 6 PNGs exist and open them for a visual check.
- [ ] **Step 3:** Commit.

### Task 14: Report notebook

**Files:**
- Create: `notebooks/report.ipynb`

- [ ] **Step 1:** Sections: setup → dataset stats → metrics table → plots → 20 side-by-side samples (`samples/side_by_side.md` generated here too) → temperature analysis → tokenizer analysis → RAI table → conclusion. Reads files only, never trains.
- [ ] **Step 2:** Verify: `jupyter nbconvert --to notebook --execute notebooks/report.ipynb --output report.ipynb` succeeds.
- [ ] **Step 3:** Commit (with outputs).

### Task 15: README

**Files:**
- Create: `README.md`

- [ ] **Step 1:** Write the 8 required sections (spec §10) with numbers pulled from `results/metrics.csv` / `temp_sweep.csv`; dataset links (WMT14 HF page + statmt.org, FLORES+ HF page); execution steps (venv, requirements, licences, `run_all.ps1`, evaluation, notebook); limitations (subset size, 1B models, AI-assisted review).
- [ ] **Step 2:** Verify every number in README matches CSVs (script or manual check); every relative link resolves.
- [ ] **Step 3:** Commit.

### Task 16: Submission document

**Files:**
- Create: `submission/<Name>_<RegNo>_FGAI_Assignment3.docx` and `.pdf`

- [ ] **Step 1 (USER):** Provide name, registration number, GitHub repo URL (after pushing).
- [ ] **Step 2:** Generate docx (anthropic-skills:docx) with: name, reg no., assignment title, system title "Comparative English–German Machine Translation: Transformer, mT5, LLaMA and Gemma", datasets + links, repo link. Convert to PDF (Word or LibreOffice).
- [ ] **Step 3:** Verify PDF opens and links are correct. Commit.
- [ ] **Step 4 (USER):** Push repo public; upload PDF to LMS.
