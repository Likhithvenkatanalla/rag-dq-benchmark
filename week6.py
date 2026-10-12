"""Week 6: robustness checks for the paper.

    seeds    the 9 random corruptions (duplicate, parse, stale at 10/25/50%) with corruption seeds
             2, 3 and 4, verified by the Week 3 fine-tuned NLI verifier (training seed 0) as in
             Week 4. With seeds 0 and 1 that gives five corruption seeds per setting.
    qwen3b   the zero-shot LLM verifier at twice the size (Qwen2.5-3B-Instruct, bfloat16, same prompt)
             on the clean corpus and the 12 Week 2 settings, reusing each run's dense rankings.
    climate  a second dataset: Climate-FEVER (Diggelmann et al., 2020) as a closed corpus of its
             5,240 evidence sentences (Wikipedia), so it is close in size to SciFact's 5,183
             abstracts. Each record is one sentence titled with its article; DISPUTED claims are
             dropped (1,381 claims left, 907 with evidence). A claim's gold records are its evidence
             sentences whose label matches the claim label. The clean corpus and the 9 seed-0
             corruptions (chunking does not apply to one-sentence records) go through the unchanged
             pipeline: BM25 + dense retrieval, Qwen2.5-1.5B and the three SciFact-trained NLI
             verifiers (no Climate-FEVER training; SciFact thresholds).
    summary  results/week6/summary.json (run after the three above).

Usage:
    python week6.py seeds --only stale_r0.25_gold_s2
    python week6.py qwen3b --only chunk_1            # or clean_rerun
    python week6.py climate --only parse_r0.5_gold_s0   # or clean_rerun
    python week6.py summary
seeds needs models/nli_finetuned_seed0/; climate needs models/nli_finetuned_seed{0,1,2}/ (the
models-week3 release). Climate-FEVER runs in work_climate/ (data and intermediate results, not
committed); its results are copied to results/climate_fever/.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import urllib.request

HERE = os.path.dirname(os.path.realpath(__file__))
RATES = [0.1, 0.25, 0.5]
DEFECTS = ["duplicate", "parse", "stale"]
NEW_SEEDS = [2, 3, 4]
ALL_SEEDS = [0, 1, 2, 3, 4]
SEED_RUNS = [f"{d}_r{r}_gold_s{s}" for s in NEW_SEEDS for d in DEFECTS for r in RATES]
OUT_DIR = os.path.join("results", "week6")

QWEN3B = "Qwen/Qwen2.5-3B-Instruct"
QWEN3B_DIR = os.path.join(OUT_DIR, "qwen3b")

CLIMATE_URL = ("https://raw.githubusercontent.com/tdiggelm/climate-fever-dataset/main/dataset/"
               "climate-fever.jsonl")
CLIMATE_SHA256 = "8a4b9032d861be482ffb49dddfd283ffa6089e654f1e968040011882c5eb6e0b"
CLIMATE_WORK = os.path.join(HERE, "work_climate")
CLIMATE_DIR = os.path.join("results", "climate_fever")
CLIMATE_RUNS = ["clean_rerun"] + [f"{d}_r{r}_gold_s0" for d in DEFECTS for r in RATES]
# Extra directional pairs for the stale defect: the SciFact list is biomedical, and most one-sentence
# climate records contain none of its words (corrupt_corpus.py reads data/data/flips.json).
CLIMATE_FLIPS = [("warming", "cooling"), ("warmer", "cooler"), ("warm", "cool"), ("rising", "falling"),
                 ("rise", "fall"), ("rises", "falls"), ("rose", "fell"), ("increasing", "decreasing"),
                 ("above", "below"), ("highest", "lowest"), ("high", "low"), ("most", "least"),
                 ("largest", "smallest"), ("large", "small"), ("stronger", "weaker"),
                 ("faster", "slower"), ("gain", "loss"), ("growth", "decline"), ("likely", "unlikely")]
CLIMATE_LABELS = {"SUPPORTS": "SUPPORT", "REFUTES": "CONTRADICT", "NOT_ENOUGH_INFO": "NOT_ENOUGH_INFO"}


def py(args, cwd=None):
    print("+", " ".join(args), flush=True)
    subprocess.run([sys.executable] + args, check=True, cwd=cwd)


# ---------------------------------------------------------------- corruption seeds

def run_seed(name):
    from week4_seed import run_one
    run_one(name)


# ---------------------------------------------------------------- larger LLM

def run_qwen3b(run):
    from week3_verifier import RUNS, corpus_for
    if run not in RUNS:
        raise SystemExit(f"unknown run {run}; choose from {', '.join(RUNS)}")
    if run == "clean_rerun":
        corpus_args = ["--out", "clean_rerun"]
    else:
        corpus_for(run)  # regenerate the corrupted corpus and check it against the committed manifest
        corpus_args = ["--corpus", os.path.join("data", "corrupted", run, "corpus.jsonl")]
    py(["week1_baseline.py", "--step", "verify", *corpus_args, "--llm", QWEN3B,
        "--verification-out", os.path.join(QWEN3B_DIR, run, "verification.json")])


# ---------------------------------------------------------------- Climate-FEVER

def climate_prepare():
    """Write work_climate/data/data/{corpus,claims_dev}.jsonl in SciFact format and link the scripts."""
    data = os.path.join(CLIMATE_WORK, "data", "data")
    os.makedirs(data, exist_ok=True)
    raw = os.path.join(CLIMATE_WORK, "data", "climate-fever.jsonl")
    if not os.path.exists(raw):
        urllib.request.urlretrieve(CLIMATE_URL, raw)
    with open(raw, "rb") as f:
        if hashlib.sha256(f.read()).hexdigest() != CLIMATE_SHA256:
            raise RuntimeError("climate-fever.jsonl differs from the version this project used")
    with open(raw) as f:
        rows = [json.loads(line) for line in f if line.strip()]

    doc_id, corpus = {}, []
    for r in rows:
        for e in r["evidences"]:
            if e["evidence_id"] not in doc_id:
                doc_id[e["evidence_id"]] = len(corpus)
                corpus.append({"doc_id": len(corpus), "title": e["article"], "abstract": [e["evidence"]],
                               "evidence_id": e["evidence_id"]})
    claims = []
    for r in rows:
        if r["claim_label"] == "DISPUTED":
            continue
        label = CLIMATE_LABELS[r["claim_label"]]
        evidence = {}
        if label != "NOT_ENOUGH_INFO":
            for e in r["evidences"]:
                if e["evidence_label"] == r["claim_label"]:
                    evidence[str(doc_id[e["evidence_id"]])] = [{"sentences": [0], "label": label}]
        claims.append({"id": int(r["claim_id"]), "claim": r["claim"], "evidence": evidence,
                       "cited_doc_ids": [doc_id[e["evidence_id"]] for e in r["evidences"]]})
    for name, items in (("corpus.jsonl", corpus), ("claims_dev.jsonl", claims)):
        with open(os.path.join(data, name), "w") as f:
            for item in items:
                f.write(json.dumps(item) + "\n")

    with open(os.path.join(data, "flips.json"), "w") as f:
        json.dump(CLIMATE_FLIPS, f)

    # The pipeline scripts use paths relative to the working directory (data/, results/).
    for script in os.listdir(HERE):
        if script.endswith(".py") and not os.path.exists(os.path.join(CLIMATE_WORK, script)):
            os.symlink(os.path.join(HERE, script), os.path.join(CLIMATE_WORK, script))
    for seed in (0, 1, 2):  # SciFact-tuned thresholds
        dst = os.path.join(CLIMATE_WORK, "results", "week3", f"seed{seed}")
        os.makedirs(dst, exist_ok=True)
        shutil.copy(os.path.join(HERE, "results", "week3", f"seed{seed}", "training.json"), dst)
    print(f"Climate-FEVER: {len(corpus)} records, {len(claims)} claims, "
          f"{sum(bool(c['evidence']) for c in claims)} with evidence", flush=True)


def run_climate(run):
    if run not in CLIMATE_RUNS:
        raise SystemExit(f"unknown run {run}; choose from {', '.join(CLIMATE_RUNS)}")
    climate_prepare()
    w = CLIMATE_WORK
    if run == "clean_rerun":
        corpus_args = ["--out", "clean_rerun"]
    else:
        from week3_verifier import corrupt_args
        py(["corrupt_corpus.py", *corrupt_args(run)], cwd=w)
        os.makedirs(os.path.join(w, "results", run), exist_ok=True)
        shutil.copy(os.path.join(w, "data", "corrupted", run, "manifest.json"),
                    os.path.join(w, "results", run, "manifest.json"))
        corpus_args = ["--corpus", os.path.join("data", "corrupted", run, "corpus.jsonl")]
    py(["week1_baseline.py", "--step", "retrieve", *corpus_args], cwd=w)
    py(["week1_baseline.py", "--step", "verify", *corpus_args], cwd=w)
    py(["analyze_week1.py", "--run", run], cwd=w)
    py(["week6.py", "climate-nli", "--only", run], cwd=w)

    dst = os.path.join(HERE, CLIMATE_DIR)
    shutil.copytree(os.path.join(w, "results", run), os.path.join(dst, run), dirs_exist_ok=True)
    for seed in (0, 1, 2):
        shutil.copytree(os.path.join(w, "results", "week3", f"seed{seed}", "nli_finetuned", run),
                        os.path.join(dst, f"nli_finetuned_seed{seed}", run), dirs_exist_ok=True)


def climate_nli(run):
    """Runs inside work_climate/: the three SciFact-trained verifiers on one Climate-FEVER run."""
    from week3_verifier import evaluate_verifier, model_dir, seed_dir
    for seed in (0, 1, 2):
        with open(os.path.join(seed_dir(seed), "training.json")) as f:
            threshold = json.load(f)["threshold"]["threshold"]
        evaluate_verifier(f"nli_finetuned_seed{seed}", os.path.join(HERE, model_dir(seed)), threshold,
                          os.path.join(seed_dir(seed), "nli_finetuned"), runs=[run])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["seeds", "qwen3b", "climate", "climate-nli", "summary"])
    parser.add_argument("--only", help="one run (seeds/qwen3b/climate)")
    args = parser.parse_args()
    if args.step == "seeds":
        for name in [args.only] if args.only else SEED_RUNS:
            if name not in SEED_RUNS:
                parser.error(f"unknown run {name}")
            run_seed(name)
    elif args.step == "qwen3b":
        run_qwen3b(args.only)
    elif args.step == "climate":
        run_climate(args.only)
    elif args.step == "climate-nli":
        climate_nli(args.only)
    else:
        from week6_summary import main as summary
        summary()
