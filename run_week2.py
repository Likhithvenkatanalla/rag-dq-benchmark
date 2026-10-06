"""Week 2 grid: build every corrupted corpus, run the pipeline on each, and summarise.

Usage:
    python run_week2.py                 # retrieve + verify + analyze for every setting (GPU advised)
    python run_week2.py --sparse-only   # BM25 retrieval only: no model downloads, a few minutes on CPU
    python run_week2.py --summary-only  # rebuild the summary from existing results
    python run_week2.py --only stale_r0.25_gold_s0   # one setting (used by the GitHub Actions workflow)
    python run_week2.py --only clean_rerun           # re-run the clean corpus into results/clean_rerun/
Writes results/week2_summary.json and prints a Markdown table.
Settings that already have results are skipped, so an interrupted run can simply be restarted.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

RATES = [0.1, 0.25, 0.5]
CHUNK_SIZES = [1, 3, 5]
SEED = 0
RESULTS_DIR = "results"


def settings():
    """(run name, corrupt_corpus.py arguments) for every grid cell."""
    grid = []
    for defect in ["duplicate", "parse", "stale"]:
        for rate in RATES:
            grid.append((f"{defect}_r{rate}_gold_s{SEED}",
                         ["--defect", defect, "--rate", str(rate), "--seed", str(SEED)]))
    for size in CHUNK_SIZES:
        grid.append((f"chunk_{size}", ["--defect", "chunk", "--chunk-size", str(size)]))
    return grid


def run(args):
    print("+", " ".join(args), flush=True)
    subprocess.run([sys.executable] + args, check=True)


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def summarise():
    rows = []
    week1_verification = load_json(os.path.join(RESULTS_DIR, "week1_verification.json"))
    for name, retrieval, verification in (
            [("clean (Week 1)", load_json(os.path.join(RESULTS_DIR, "week1_retrieval.json")),
              week1_verification and week1_verification["summary"])]
            + [(name, load_json(os.path.join(RESULTS_DIR, name, "retrieval.json")),
                (load_json(os.path.join(RESULTS_DIR, name, "verification.json")) or {}).get("summary"))
               for name in ["clean_rerun"] + [n for n, _ in settings()]]):
        if retrieval is None:
            continue
        row = {"run": name}
        for retriever, scores in retrieval.items():
            for metric, value in scores.items():
                row[f"{retriever}_{metric}"] = value
        if verification:
            row.update({"accuracy": verification["accuracy"], "macro_f1": verification["macro_f1"],
                        "gold_doc_in_context_rate": verification["gold_doc_in_context_rate"]})
        manifest = load_json(os.path.join(RESULTS_DIR, name, "manifest.json"))
        if manifest and "stale_copies_without_edits" in manifest:
            row["stale_copies_without_edits"] = manifest["stale_copies_without_edits"]
        rows.append(row)

    with open(os.path.join(RESULTS_DIR, "week2_summary.json"), "w") as f:
        json.dump(rows, f, indent=2)
    columns = [c for c in ["run", "bm25_recall@1", "bm25_recall@3", "bm25_recall@5", "bm25_recall@10",
                           "dense_bge_small_recall@1", "dense_bge_small_recall@10",
                           "accuracy", "macro_f1", "gold_doc_in_context_rate"]
               if any(c in r for r in rows)]
    print("| " + " | ".join(columns) + " |")
    print("|" + "---|" * len(columns))
    for r in rows:
        print("| " + " | ".join(str(r.get(c, "")) for c in columns) + " |")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sparse-only", action="store_true", help="BM25 retrieval only")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--only", help="run a single setting by name, or clean_rerun")
    args = parser.parse_args()
    grid = settings()
    if args.only == "clean_rerun":
        grid = []
        if not args.summary_only:
            run(["week1_baseline.py", "--step", "retrieve", "--out", "clean_rerun"])
            run(["week1_baseline.py", "--step", "verify", "--out", "clean_rerun"])
            run(["analyze_week1.py", "--run", "clean_rerun"])
    elif args.only:
        grid = [g for g in grid if g[0] == args.only]
        if not grid:
            parser.error(f"unknown setting {args.only}; choose from clean_rerun, "
                         + ", ".join(n for n, _ in settings()))
    if not args.summary_only:
        for name, corrupt_args in grid:
            done = "retrieval.json" if args.sparse_only else "analysis.json"
            if os.path.exists(os.path.join(RESULTS_DIR, name, done)):
                print(f"skip {name}: results/{name}/{done} exists", flush=True)
                continue
            run(["corrupt_corpus.py"] + corrupt_args)
            corpus = os.path.join("data", "corrupted", name, "corpus.jsonl")
            # Keep the manifest next to the results (data/ is not committed).
            os.makedirs(os.path.join(RESULTS_DIR, name), exist_ok=True)
            shutil.copy(os.path.join("data", "corrupted", name, "manifest.json"),
                        os.path.join(RESULTS_DIR, name, "manifest.json"))
            if args.sparse_only:
                run(["week1_baseline.py", "--step", "retrieve", "--corpus", corpus, "--sparse-only"])
                continue
            run(["week1_baseline.py", "--step", "retrieve", "--corpus", corpus])
            run(["week1_baseline.py", "--step", "verify", "--corpus", corpus])
            run(["analyze_week1.py", "--run", name])
    summarise()


if __name__ == "__main__":
    main()
