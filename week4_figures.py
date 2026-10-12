"""Week 4: figures for the report (PNG, light background) in figures/.

    fig1_verification.png     accuracy vs corruption level: Qwen, fine-tuned NLI (mean and range over
                              three training seeds) and, once run, fine-tuned NLI on corruption seed 1
    fig2_retrieval.png        recall@1 vs corruption level: BM25, dense, dense + fine-tuned reranker
    fig3_reliability.png      reliability diagram of the fine-tuned verifier (seed 0)
    fig4_confident_wrong.png  share of claims answered wrongly with confidence >= 0.9

Colours: the first three categorical slots of the dataviz reference palette (validated all-pairs);
each series also has its own marker and a value label, so colour is never the only cue.

Usage:
    python week4_figures.py     # after week4_calibration.py
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

FIG_DIR = "figures"
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # blue, orange, aqua
MARKERS = ["o", "s", "^"]
RATES = [0.1, 0.25, 0.5]
DEFECTS = [("duplicate", "Near-duplicates"), ("stale", "Stale (flipped) copies"), ("parse", "Broken parsing")]
RATE_X, RATE_TICKS = [0] + RATES, ["clean", "10%", "25%", "50%"]
CHUNK_RUNS = ["clean_rerun", "chunk_5", "chunk_3", "chunk_1"]
CHUNK_X, CHUNK_TICKS = [0, 1, 2, 3], ["whole", "5 sent.", "3 sent.", "1 sent."]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11,
    "axes.titleweight": "bold", "lines.linewidth": 2, "lines.markersize": 7,
})


def load(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def run_names(defect, seed=0):
    return ["clean_rerun"] + [f"{defect}_r{r}_gold_s{seed}" for r in RATES]


def plot_series(ax, x, y, i, label=None, lo=None, hi=None):
    ax.plot(x, y, color=SERIES[i], marker=MARKERS[i], markeredgecolor=SURFACE, markeredgewidth=1.5,
            label=label, zorder=3)
    if lo is not None:
        ax.fill_between(x, lo, hi, color=SERIES[i], alpha=0.15, linewidth=0, zorder=2)


def label_ends(ax, x_end, values, gap):
    """Value labels to the right of each line's last point, spread so they never overlap."""
    order = np.argsort(values)
    last = -np.inf
    for k in order:
        y = max(values[k], last + gap)
        ax.annotate(f"{values[k]:.2f}", (x_end, y), xytext=(7, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=INK_2, annotation_clip=False)
        last = y


def save(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print("wrote", path)


def panel_figure(title, ylabel, ylim):
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.7), sharey=True)
    fig.suptitle(title, x=0.01, y=0.98, ha="left", fontsize=12.5, fontweight="bold")
    for ax in axes:
        ax.set_ylim(*ylim)
    axes[0].set_ylabel(ylabel)
    return fig, axes


def finish_panels(fig, axes):
    for ax, (_, title) in zip(axes, DEFECTS + [(None, "Chunking")]):
        ax.set_title(title, loc="left")
    for ax in axes[:3]:
        ax.set_xticks(RATE_X, RATE_TICKS)
        ax.set_xlabel("share of gold abstracts corrupted")
    axes[3].set_xticks(CHUNK_X, CHUNK_TICKS)
    axes[3].set_xlabel("record size")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", ncol=len(labels), frameon=False,
               bbox_to_anchor=(0.005, 0.915), fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 0.985, 0.84))


def fig_verification(summary):
    rows = {r["run"]: r for r in summary["runs"]}
    seeds = summary["seeds"]

    def finetuned(runs):
        vals = np.array([[rows[r][f"nli_finetuned_seed{s}"]["accuracy"] for s in seeds] for r in runs])
        return vals.mean(axis=1), vals.min(axis=1), vals.max(axis=1)

    def second_seed(runs):
        out = [load(os.path.join("results", "week3", "seed0", "nli_finetuned", r, "verification.json"))
               for r in runs]
        return [v["summary"]["accuracy"] for v in out] if all(out) else None

    fig, axes = panel_figure("Fine-tuned verifier vs zero-shot LLM under corruption",
                             "verification accuracy (300 claims)", (0.45, 0.75))
    panels = [(ax, RATE_X, run_names(d), run_names(d, 1)) for ax, (d, _) in zip(axes, DEFECTS)]
    panels.append((axes[3], CHUNK_X, CHUNK_RUNS, None))
    for k, (ax, x, runs, runs_s1) in enumerate(panels):
        qwen = [rows[r]["qwen"]["accuracy"] for r in runs]
        mean, lo, hi = finetuned(runs)
        plot_series(ax, x, qwen, 0, "Qwen (zero-shot LLM)" if k == 0 else None)
        plot_series(ax, x, mean, 1, "NLI fine-tuned (mean; band = range of 3 training seeds)" if k == 0 else None,
                    lo, hi)
        ends = [qwen[-1], mean[-1]]
        s1 = second_seed(runs_s1) if runs_s1 else None
        if s1:
            plot_series(ax, x, s1, 2, "NLI fine-tuned seed 0, second corruption seed" if k == 0 else None)
            ends.append(s1[-1])
        label_ends(ax, x[-1], ends, gap=0.022)
    finish_panels(fig, axes)
    save(fig, "fig1_verification.png")


def fig_retrieval():
    def recall(run):
        r = load(os.path.join("results", run, "retrieval.json"))
        rr = load(os.path.join("results", "week3", "reranker", "finetuned", run, "retrieval.json"))
        return r["bm25"]["recall@1"], r["dense_bge_small"]["recall@1"], rr["reranked"]["recall@1"]

    fig, axes = panel_figure("Copies push the gold abstract out of first place; reranking recovers part of it",
                             "recall@1 (188 claims with evidence)", (0.4, 0.85))
    panels = [(ax, RATE_X, run_names(d)) for ax, (d, _) in zip(axes, DEFECTS)] + [(axes[3], CHUNK_X, CHUNK_RUNS)]
    names = ["BM25", "Dense (BGE-small)", "Dense + fine-tuned reranker"]
    for k, (ax, x, runs) in enumerate(panels):
        vals = np.array([recall(r) for r in runs])
        for i, name in enumerate(names):
            plot_series(ax, x, vals[:, i], i, name if k == 0 else None)
        label_ends(ax, x[-1], list(vals[-1]), gap=0.03)
    finish_panels(fig, axes)
    save(fig, "fig2_retrieval.png")


def reliability_points(run, bins=5, min_n=10):
    """Confidence vs accuracy in wide bins (300 claims are too few for 10 bins)."""
    from week3_verifier import load_preds
    from week4_calibration import claim_confidence
    preds = list(load_preds(os.path.join("results", "week3", "seed0", "nli_finetuned", run,
                                         "verification.json"))[1].values())
    conf = np.array([claim_confidence(r) for r in preds])
    corr = np.array([r["pred"] == r["gold"] for r in preds], dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    pts = [(conf[idx == b].mean(), corr[idx == b].mean(), int((idx == b).sum()))
           for b in range(bins) if (idx == b).sum() >= min_n]
    return [p[0] for p in pts], [p[1] for p in pts], [p[2] for p in pts]


def fig_reliability(cal):
    runs = [("clean_rerun", "clean"), ("stale_r0.5_gold_s0", "stale 50%"), ("chunk_1", "1-sentence chunks")]
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.plot([0, 1], [0, 1], color=INK_2, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
    ax.text(0.97, 0.995, "confidence = accuracy", ha="right", va="top", fontsize=8.5, color=INK_2)
    for i, (run, label) in enumerate(runs):
        x, y, n = reliability_points(run)
        ece = cal["verifiers"]["seed0"][run]["ece"]
        plot_series(ax, x, y, i, f"{label} (ECE {ece:.2f})")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal")
    ax.set_xlabel("verifier confidence (5 bins; bins with at least 10 claims)")
    ax.set_ylabel("share of answers that were right")
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    ax.set_title("Below the dashed line = overconfident.\nFine-tuned verifier, seed 0", loc="left", fontsize=11.5)
    fig.tight_layout()
    save(fig, "fig3_reliability.png")


def fig_confident_wrong(cal):
    runs = [("clean_rerun", "clean"), ("duplicate_r0.5_gold_s0", "duplicate 50%"),
            ("stale_r0.5_gold_s0", "stale 50%"), ("parse_r0.5_gold_s0", "parse 50%"),
            ("chunk_3", "3-sent. chunks"), ("chunk_1", "1-sent. chunks")]
    seeds = sorted(cal["verifiers"])
    vals = np.array([[cal["verifiers"][s][r]["confidently_wrong"] * 100 for s in seeds] for r, _ in runs])
    fig, ax = plt.subplots(figsize=(8, 3.8))
    x = np.arange(len(runs))
    ax.bar(x, vals.mean(axis=1), width=0.6, color=SERIES[1], zorder=2)
    for j in range(len(seeds)):
        ax.scatter(x, vals[:, j], s=22, color=INK, zorder=3, label="one training seed" if j == 0 else None)
    for xi, mean, top in zip(x, vals.mean(axis=1), vals.max(axis=1)):
        ax.annotate(f"{mean:.0f}%", (xi, top), xytext=(0, 7), textcoords="offset points", ha="center",
                    fontsize=9, color=INK_2)
    ax.set_xticks(x, [label for _, label in runs])
    ax.set_ylabel("% of claims: wrong, confidence >= 0.9")
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, max(25, vals.max() + 4))
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    ax.set_title("Confidently wrong answers (fine-tuned verifier; bar = mean of 3 training seeds)", loc="left")
    fig.tight_layout()
    save(fig, "fig4_confident_wrong.png")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    fig_verification(load(os.path.join("results", "week3", "summary.json")))
    fig_retrieval()
    cal = load(os.path.join("results", "week4", "calibration.json"))
    fig_reliability(cal)
    fig_confident_wrong(cal)


if __name__ == "__main__":
    main()
