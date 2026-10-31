"""Static figures for the README and report notebook (read from results/ only)."""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.common import PATHS, ROWS

# Colour = model family (validated categorical slots 1-4); prompting rows are hatched / dashed.
FAMILY = {"transformer": "#2a78d6", "mt5": "#eb6834", "llama": "#1baf7a", "gemma": "#eda100"}
LABEL = {"transformer": "Transformer (scratch)", "mt5": "mT5-small (FT)", "llama_prompt": "LLaMA-3.2-1B (5-shot)",
         "llama_lora": "LLaMA-3.2-1B (LoRA)", "gemma_prompt": "Gemma-3-1B (5-shot)", "gemma_lora": "Gemma-3-1B (LoRA)"}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def color(row):
    return FAMILY[row.split("_")[0]]


def is_prompt(row):
    return row.endswith("_prompt")


def _style(ax, title, ylabel=None, xlabel=None):
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=10)
    if ylabel:
        ax.set_ylabel(ylabel, color=MUTED)
    if xlabel:
        ax.set_xlabel(xlabel, color=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED)
    ax.set_facecolor(SURFACE)


def _save(fig, name):
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    fig.savefig(PATHS["plots"] / name, dpi=160)
    plt.close(fig)


def scores(m):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, metric in zip(axes, ["bleu", "chrf"]):
        vals = m.set_index("row").loc[ROWS, metric]
        bars = ax.bar(range(len(ROWS)), vals, color=[color(r) for r in ROWS], width=0.72, edgecolor=SURFACE, linewidth=2,
                      hatch=None)
        for b, r in zip(bars, ROWS):
            if is_prompt(r):
                b.set_hatch("//")
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{b.get_height():.1f}", ha="center", va="bottom",
                    fontsize=9, color=INK)
        ax.set_xticks(range(len(ROWS)), [LABEL[r] for r in ROWS], rotation=25, ha="right", fontsize=9)
        _style(ax, f"{metric.upper() if metric == 'bleu' else 'chrF'} on FLORES+ devtest (greedy)", metric)
    _save(fig, "scores.png")


LABEL_OFFSET = {"gemma_lora": (-8, -14)}  # Gemma LoRA crowds its neighbours in both scatters: label it below-left


def _scatter(m, xcol, xlabel, title, name, logx=False):
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for _, r in m.iterrows():
        ax.scatter(r[xcol], r["chrf"], s=90, color=color(r["row"]), edgecolor=SURFACE, linewidth=2,
                   marker="s" if is_prompt(r["row"]) else "o", zorder=3)
        ax.annotate(LABEL[r["row"]], (r[xcol], r["chrf"]), textcoords="offset points",
                    xytext=LABEL_OFFSET.get(r["row"], (8, 4)), fontsize=8.5, color=INK,
                    ha="right" if r["row"] == "gemma_lora" else "left")
    if logx:
        ax.set_xscale("log")
    _style(ax, title, "chrF", xlabel)
    _save(fig, name)


def temperature(t):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, metric in zip(axes, ["bleu", "chrf"]):
        for row in ROWS:
            d = t[t.row == row].sort_values("temperature")
            ax.plot(d.temperature, d[metric], color=color(row), linewidth=2, marker="o", markersize=8,
                    linestyle="--" if is_prompt(row) else "-", markeredgecolor=SURFACE, markeredgewidth=2,
                    label=LABEL[row])
        ax.set_xticks([0, 0.3, 0.7, 1.0, 1.3], ["greedy", "0.3", "0.7", "1.0", "1.3"])
        _style(ax, f"{'BLEU' if metric == 'bleu' else 'chrF'} vs sampling temperature (300 sentences)", metric,
               "temperature")
    # one shared legend below the panels: end labels collide where the curves converge at T=1.3
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False, fontsize=9,
               handlelength=3.5)
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(PATHS["plots"] / "temperature.png", dpi=160)
    plt.close(fig)


def fertility(f):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = range(len(f))
    w = 0.38
    b1 = ax.bar([i - w / 2 for i in x], f.tokens_per_sent_en, w, color="#2a78d6", edgecolor=SURFACE, linewidth=2,
                label="English")
    b2 = ax.bar([i + w / 2 for i in x], f.tokens_per_sent_de, w, color="#eb6834", edgecolor=SURFACE, linewidth=2,
                label="German")
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{b.get_height():.1f}", ha="center", va="bottom",
                fontsize=8.5, color=INK)
    ax.set_xticks(list(x), f.tokenizer.str.replace(" (", "\n("), fontsize=9)
    ax.legend(frameon=False)
    _style(ax, "Tokens per FLORES+ sentence by tokenizer", "mean tokens / sentence")
    _save(fig, "fertility.png")


MT5_RESUME_STEP = 938  # epoch 1 ran with grad_accum 4; the resumed epochs 2-3 with grad_accum 8


def mt5_per_token_loss(hist):
    """The Trainer logs mT5's loss summed over gradient-accumulation micro-batches; divide that back out."""
    return [[s, loss / (4 if s <= MT5_RESUME_STEP else 8)] for s, loss in hist]


def loss_curves():
    fig, ax = plt.subplots(figsize=(8, 4.4))
    for name, row in [("train_transformer", "transformer"), ("train_mt5", "mt5"), ("train_llama_lora", "llama_lora"),
                      ("train_gemma_lora", "gemma_lora")]:
        path = PATHS["logs"] / f"{name}.json"
        if not path.exists():
            continue
        hist = json.loads(path.read_text(encoding="utf-8")).get("loss_history", [])
        if not hist:
            continue
        if row == "mt5":
            hist = mt5_per_token_loss(hist)
        steps, losses = zip(*hist)
        total = max(steps)
        ax.plot([s / total for s in steps], losses, color=color(row), linewidth=2, label=LABEL[row])
    ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=9)
    _style(ax, "Training loss (x = fraction of each model's training run)", "loss (log)", "training progress")
    _save(fig, "loss_curves.png")


def main():
    m = pd.read_csv(PATHS["results"] / "metrics.csv")
    scores(m)
    _scatter(m, "params_total", "total parameters (log)", "Quality vs model size", "quality_vs_params.png", logx=True)
    _scatter(m, "ms_per_sent", "inference ms / sentence (log, batch 16, greedy)", "Quality vs inference cost",
             "quality_vs_latency.png", logx=True)
    temperature(pd.read_csv(PATHS["results"] / "temp_sweep.csv"))
    fertility(pd.read_csv(PATHS["results"] / "fertility.csv"))
    loss_curves()
    print(sorted(p.name for p in PATHS["plots"].glob("*.png")))


if __name__ == "__main__":
    main()
