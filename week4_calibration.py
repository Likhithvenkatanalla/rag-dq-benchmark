"""Week 4: is the fine-tuned verifier's confidence trustworthy, and does corruption change that?

For every claim the verifier gives a confidence in its answer:
    SUPPORT / CONTRADICT   the highest probability of that label over the top-3 records
    NOT_ENOUGH_INFO        1 - the highest SUPPORT or CONTRADICT probability
Calibration compares that confidence with whether the answer was right:
    ECE      expected calibration error: average gap between confidence and accuracy over 10
             equal-width confidence bins, weighted by the share of claims in each bin (0 = perfect)
    Brier    mean squared difference between confidence and correctness (0/1); lower is better
    confidently wrong   share of all claims answered wrongly with confidence >= 0.9
For stale and duplicate runs it also reports the claims whose top-3 contained a corrupted copy.

The Qwen LLM gave a one-word answer without probabilities, so it is not included.

Usage:
    python week4_calibration.py      # results/week4/calibration.json + a printed table
"""
import glob
import json
import os

import numpy as np

from week3_verifier import RUNS, load_preds

OUT = os.path.join("results", "week4", "calibration.json")
BINS = 10
CONFIDENT = 0.9


def claim_confidence(record):
    p = np.array(record["probs"])
    s, c = p[:, 0].max(), p[:, 1].max()
    return {"SUPPORT": s, "CONTRADICT": c}.get(record["pred"], 1 - max(s, c))


def calibration(confidences, correct):
    conf, corr = np.asarray(confidences, dtype=float), np.asarray(correct, dtype=float)
    edges = np.linspace(0, 1, BINS + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, BINS - 1)
    ece, bins = 0.0, []
    for b in range(BINS):
        mask = idx == b
        if mask.any():
            gap = abs(conf[mask].mean() - corr[mask].mean())
            ece += mask.mean() * gap
            bins.append({"bin": [round(edges[b], 1), round(edges[b + 1], 1)], "n": int(mask.sum()),
                         "mean_confidence": round(float(conf[mask].mean()), 4),
                         "accuracy": round(float(corr[mask].mean()), 4)})
    return {"n": len(conf), "accuracy": round(float(corr.mean()), 4),
            "mean_confidence": round(float(conf.mean()), 4), "ece": round(float(ece), 4),
            "brier": round(float(((conf - corr) ** 2).mean()), 4),
            "confidently_wrong": round(float(((conf >= CONFIDENT) & (corr == 0)).mean()), 4),
            "bins": bins}


def copies_of(run):
    path = os.path.join("results", run, "manifest.json")
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return {c["doc_id"] for c in json.load(f)["changes"] if "derived_from" in c}


def main():
    verifiers = sorted(glob.glob(os.path.join("results", "week3", "seed*", "nli_finetuned")))
    week4_runs = sorted(os.path.basename(p) for p in glob.glob(os.path.join("results", "*_s1"))
                        if os.path.isdir(p))
    out = {"confidence_threshold": CONFIDENT, "bins": BINS, "verifiers": {}}
    for vdir in verifiers:
        tag = vdir.split(os.sep)[-2]
        out["verifiers"][tag] = {}
        for run in RUNS + week4_runs:
            loaded = load_preds(os.path.join(vdir, run, "verification.json"))
            if loaded is None:
                continue
            preds = list(loaded[1].values())
            conf = [claim_confidence(r) for r in preds]
            corr = [r["pred"] == r["gold"] for r in preds]
            entry = calibration(conf, corr)
            copies = copies_of(run)
            if copies:
                hit = [i for i, r in enumerate(preds) if copies & set(r["top_docs"])]
                entry["claims_with_copy_in_context"] = calibration([conf[i] for i in hit], [corr[i] for i in hit])
                entry["claims_without_copy"] = calibration([conf[i] for i in range(len(preds)) if i not in set(hit)],
                                                           [corr[i] for i in range(len(preds)) if i not in set(hit)])
            out["verifiers"][tag][run] = entry
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)

    print("| run | " + " | ".join(f"{t} acc / conf / ECE / Brier / conf-wrong" for t in out["verifiers"]) + " |")
    print("|" + "---|" * (len(out["verifiers"]) + 1))
    for run in RUNS + week4_runs:
        cells = []
        for t, runs in out["verifiers"].items():
            e = runs.get(run)
            cells.append(f"{e['accuracy']} / {e['mean_confidence']} / {e['ece']} / {e['brier']} / "
                         f"{e['confidently_wrong']}" if e else "")
        print(f"| {run} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
