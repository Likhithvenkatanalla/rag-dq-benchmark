"""Week 1 analysis: per-class metrics, confusion matrix, majority baseline, and
verification accuracy split by whether the gold abstract reached the LLM context.

Usage (after week1_baseline.py --step verify):
    python analyze_week1.py
    python analyze_week1.py --run <name>   # a corrupted-corpus run in results/<name>/
"""
import argparse
import json
import os
from collections import Counter

from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

LABELS = ["SUPPORT", "CONTRADICT", "NOT_ENOUGH_INFO"]
RESULTS_DIR = "results"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", help="results subfolder of a corrupted-corpus run (default: Week 1)")
    args = parser.parse_args()
    if args.run:
        src = os.path.join(RESULTS_DIR, args.run, "verification.json")
        dst = os.path.join(RESULTS_DIR, args.run, "analysis.json")
    else:
        src = os.path.join(RESULTS_DIR, "week1_verification.json")
        dst = os.path.join(RESULTS_DIR, "week1_analysis.json")

    with open(src) as f:
        preds = json.load(f)["predictions"]
    with open(os.path.join("data", "data", "claims_dev.jsonl")) as f:
        claims = {c["id"]: c for c in (json.loads(l) for l in f if l.strip())}

    gold = [r["gold"] for r in preds]
    pred = [r["pred"] for r in preds]
    p, r, f, s = precision_recall_fscore_support(gold, pred, labels=LABELS, zero_division=0)
    majority_label, majority_n = Counter(gold).most_common(1)[0]

    # Claims with gold evidence: was a gold abstract among the top-3 given to the LLM?
    in_ctx, out_ctx = [], []
    for rec in preds:
        gdocs = {int(k) for k in claims[rec["id"]].get("evidence", {})}
        if not gdocs:
            continue
        # top_parent_docs is present for chunked corpora (top_docs are then chunk ids)
        top = set(rec.get("top_parent_docs", rec["top_docs"]))
        (in_ctx if gdocs & top else out_ctx).append(rec["gold"] == rec["pred"])

    out = {
        "gold_distribution": dict(Counter(gold)),
        "pred_distribution": dict(Counter(pred)),
        "majority_baseline": {"label": majority_label, "accuracy": round(majority_n / len(gold), 4),
                              "macro_f1": round(f1_score(gold, [majority_label] * len(gold),
                                                         labels=LABELS, average="macro"), 4)},
        "per_class": {lab: {"precision": round(float(p[i]), 4), "recall": round(float(r[i]), 4),
                            "f1": round(float(f[i]), 4), "support": int(s[i])}
                      for i, lab in enumerate(LABELS)},
        "confusion_matrix": {"rows_gold_cols_pred": LABELS,
                             "matrix": confusion_matrix(gold, pred, labels=LABELS).tolist()},
        "evidence_claims_accuracy": {
            "gold_doc_in_top3": {"n": len(in_ctx), "accuracy": round(sum(in_ctx) / len(in_ctx), 4)},
            "gold_doc_not_in_top3": {"n": len(out_ctx),
                                     "accuracy": round(sum(out_ctx) / len(out_ctx), 4) if out_ctx else None},
        },
    }
    print(json.dumps(out, indent=2))
    with open(dst, "w") as fh:
        json.dump(out, fh, indent=2)


if __name__ == "__main__":
    main()
