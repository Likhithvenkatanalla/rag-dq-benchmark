"""Week 1 analysis: per-class metrics, confusion matrix, majority baseline, and
verification accuracy split by whether the gold abstract reached the LLM context.

Usage (after week1_baseline.py --step verify):
    python analyze_week1.py
"""
import json
import os
from collections import Counter

from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

LABELS = ["SUPPORT", "CONTRADICT", "NOT_ENOUGH_INFO"]
RESULTS_DIR = "results"


def main():
    with open(os.path.join(RESULTS_DIR, "week1_verification.json")) as f:
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
        (in_ctx if gdocs & set(rec["top_docs"]) else out_ctx).append(rec["gold"] == rec["pred"])

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
    with open(os.path.join(RESULTS_DIR, "week1_analysis.json"), "w") as fh:
        json.dump(out, fh, indent=2)


if __name__ == "__main__":
    main()
