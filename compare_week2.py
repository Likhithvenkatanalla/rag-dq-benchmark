"""Week 2: paired comparison of each corrupted-corpus run against the clean re-run.

Accuracy differences of 1-2 points over 300 claims can be noise, so this compares predictions
claim by claim: how many changed, how many went right->wrong and wrong->right, and an exact
McNemar test on those discordant pairs. For duplicate/stale runs it also counts the claims whose
LLM context contained a corrupted copy, and how many of those changed prediction.

Usage (after the Week 2 runs):
    python compare_week2.py     # writes results/week2_paired.json and prints a table
"""
import json
import math
import os

RESULTS_DIR = "results"
BASELINE = "clean_rerun"


def load_predictions(name):
    path = os.path.join(RESULTS_DIR, name, "verification.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return {r["id"]: r for r in json.load(f)["predictions"]}


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p-value for b and c discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n)


def main():
    base = load_predictions(BASELINE)
    rows = []
    for name in sorted(os.listdir(RESULTS_DIR)):
        cur = load_predictions(name) if name != BASELINE else None
        if cur is None:
            continue
        right_to_wrong = sum(base[i]["pred"] == base[i]["gold"] and cur[i]["pred"] != cur[i]["gold"] for i in base)
        wrong_to_right = sum(base[i]["pred"] != base[i]["gold"] and cur[i]["pred"] == cur[i]["gold"] for i in base)
        row = {"run": name,
               "predictions_changed": sum(base[i]["pred"] != cur[i]["pred"] for i in base),
               "right_to_wrong": right_to_wrong, "wrong_to_right": wrong_to_right,
               "mcnemar_p": round(mcnemar_exact(right_to_wrong, wrong_to_right), 3)}
        manifest_path = os.path.join(RESULTS_DIR, name, "manifest.json")
        if os.path.exists(manifest_path):
            with open(manifest_path) as f:
                copies = {c["doc_id"] for c in json.load(f)["changes"] if "derived_from" in c}
            if copies:
                hit = [i for i in cur if copies & set(cur[i]["top_docs"])]
                row["claims_with_copy_in_context"] = len(hit)
                row["of_which_prediction_changed"] = sum(base[i]["pred"] != cur[i]["pred"] for i in hit)
        rows.append(row)

    with open(os.path.join(RESULTS_DIR, "week2_paired.json"), "w") as f:
        json.dump({"baseline": BASELINE, "runs": rows}, f, indent=2)
    cols = ["run", "predictions_changed", "right_to_wrong", "wrong_to_right", "mcnemar_p",
            "claims_with_copy_in_context", "of_which_prediction_changed"]
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for r in rows:
        print("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")


if __name__ == "__main__":
    main()
