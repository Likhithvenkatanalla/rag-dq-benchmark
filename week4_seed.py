"""Week 4: repeat the random Week 2 corruptions with a second seed.

The duplicate, parse and stale defects pick which documents to corrupt (and how) at random; seed 0
was used in Weeks 2-3. This re-runs those 9 settings with seed 1, so an effect that shows up for both
seeds does not depend on which documents happened to be hit. Chunking has no randomness and is not
repeated.

For each setting: corrupt the corpus, retrieve (BM25 and dense, as in Week 2), and verify with the
fine-tuned NLI verifier (Week 3, seed 0, same threshold). The Qwen LLM is not re-run: it needs CPUs
with native bfloat16 to finish in reasonable time on GitHub Actions (see Week 2).

Usage:
    python week4_seed.py --only stale_r0.25_gold_s1   # one setting (as the workflow runs it)
    python week4_seed.py                              # all 9
    python week4_seed.py --summary                    # results/week4/seed_comparison.json
Needs models/nli_finetuned_seed0/ and results/week3/seed0/training.json.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

from week3_verifier import corrupt_args, evaluate_verifier, load_preds, model_dir, paired, seed_dir

SEED = 1
RATES = [0.1, 0.25, 0.5]
NAMES = [f"{d}_r{r}_gold_s{SEED}" for d in ("duplicate", "parse", "stale") for r in RATES]
OUT_DIR = os.path.join("results", "week4")


def run_one(name):
    subprocess.run([sys.executable, "corrupt_corpus.py"] + corrupt_args(name), check=True)
    os.makedirs(os.path.join("results", name), exist_ok=True)
    shutil.copy(os.path.join("data", "corrupted", name, "manifest.json"),
                os.path.join("results", name, "manifest.json"))
    corpus = os.path.join("data", "corrupted", name, "corpus.jsonl")
    subprocess.run([sys.executable, "week1_baseline.py", "--step", "retrieve", "--corpus", corpus], check=True)
    with open(os.path.join(seed_dir(0), "training.json")) as f:
        threshold = json.load(f)["threshold"]["threshold"]
    evaluate_verifier("nli_finetuned_seed0", model_dir(0), threshold,
                      os.path.join("results", "week3", "seed0", "nli_finetuned"), runs=[name])


def summary():
    """Seed 0 vs seed 1 for each defect and rate: retrieval and verification, each vs clean."""
    clean = load_preds(os.path.join("results", "week3", "seed0", "nli_finetuned", "clean_rerun",
                                    "verification.json"))[1]
    rows = []
    for name in NAMES:
        row = {"setting": name.rsplit("_s", 1)[0]}
        for seed in (0, 1):
            run = name.replace(f"_s{SEED}", f"_s{seed}")
            path = os.path.join("results", run, "retrieval.json")
            ver = load_preds(os.path.join("results", "week3", "seed0", "nli_finetuned", run, "verification.json"))
            if not os.path.exists(path) or ver is None:
                continue
            with open(path) as f:
                retrieval = json.load(f)
            row[f"seed{seed}"] = {"bm25_recall@1": retrieval["bm25"]["recall@1"],
                                  "bm25_recall@10": retrieval["bm25"]["recall@10"],
                                  "dense_recall@1": retrieval["dense_bge_small"]["recall@1"],
                                  "dense_recall@10": retrieval["dense_bge_small"]["recall@10"],
                                  "nli_accuracy": ver[0]["accuracy"], "nli_macro_f1": ver[0]["macro_f1"],
                                  "nli_vs_clean": paired(clean, ver[1])}
        rows.append(row)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "seed_comparison.json"), "w") as f:
        json.dump(rows, f, indent=2)
    fmt = lambda r, s: (f"{r[s]['dense_recall@1']} | {r[s]['nli_accuracy']} | "
                        f"{r[s]['nli_vs_clean']['right_to_wrong']}/{r[s]['nli_vs_clean']['wrong_to_right']}") \
        if s in r else "- | - | -"
    print("| setting | s0 dense R@1 | s0 NLI acc | s0 lost/gained | s1 dense R@1 | s1 NLI acc | s1 lost/gained |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['setting']} | {fmt(r, 'seed0')} | {fmt(r, 'seed1')} |")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=NAMES)
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    if args.summary:
        summary()
    else:
        for name in [args.only] if args.only else NAMES:
            run_one(name)
