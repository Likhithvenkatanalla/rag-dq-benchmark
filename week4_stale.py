"""Week 4: re-read the stale-copy results using only copies whose findings were actually flipped.

The stale defect flips directional words by rule; about 1 in 5 copies had nothing to flip and are
identical to the original (`n_edits` = 0 in the manifest), so they act as plain duplicates. This
splits the claims of every stale run by what their top-3 context contained: an edited (really
conflicting) copy, only unedited copies, or no copy; and compares each group with the same claims
on the clean corpus, for every verifier that has results.

Usage:
    python week4_stale.py     # results/week4/stale_edited.json + a printed table
"""
import json
import os

from week3_verifier import load_preds, paired
from week4_calibration import CONFIDENT, claim_confidence

OUT = os.path.join("results", "week4", "stale_edited.json")
RUNS = [f"stale_r{r}_gold_s{s}" for s in (0, 1) for r in (0.1, 0.25, 0.5)]


def verifiers(run):
    """(name, path to this run's predictions, path to the clean predictions) for every verifier."""
    out = [("qwen", os.path.join("results", run, "verification.json"),
            os.path.join("results", "clean_rerun", "verification.json"))]
    for seed in (0, 1, 2):
        root = os.path.join("results", "week3", f"seed{seed}", "nli_finetuned")
        out.append((f"nli_finetuned_seed{seed}", os.path.join(root, run, "verification.json"),
                    os.path.join(root, "clean_rerun", "verification.json")))
    return out


def group_stats(ids, base, cur):
    if not ids:
        return {"n": 0}
    sub_base, sub_cur = {i: base[i] for i in ids}, {i: cur[i] for i in ids}
    stats = {"n": len(ids),
             "predictions_changed": sum(base[i]["pred"] != cur[i]["pred"] for i in ids),
             "moved_to_contradict": sum(cur[i]["pred"] == "CONTRADICT" != base[i]["pred"] for i in ids),
             **paired(sub_base, sub_cur)}
    if "probs" in next(iter(sub_cur.values())):
        wrong_conf = lambda p: int(sum(claim_confidence(p[i]) >= CONFIDENT and p[i]["pred"] != p[i]["gold"]
                                       for i in ids))
        stats["confidently_wrong_clean"] = wrong_conf(base)
        stats["confidently_wrong_stale"] = wrong_conf(cur)
    return stats


def main():
    rows = []
    for run in RUNS:
        manifest_path = os.path.join("results", run, "manifest.json")
        if not os.path.exists(manifest_path):
            continue
        with open(manifest_path) as f:
            changes = json.load(f)["changes"]
        edited = {c["doc_id"] for c in changes if c["n_edits"] > 0}
        unedited = {c["doc_id"] for c in changes if c["n_edits"] == 0}
        for name, path, clean_path in verifiers(run):
            cur, base = load_preds(path), load_preds(clean_path)
            if cur is None or base is None:
                continue
            cur, base = cur[1], base[1]
            groups = {"edited_copy_in_context": [], "only_unedited_copy": [], "no_copy": []}
            for i, r in cur.items():
                top = set(r["top_docs"])
                key = ("edited_copy_in_context" if top & edited else
                       "only_unedited_copy" if top & unedited else "no_copy")
                groups[key].append(i)
            rows.append({"run": run, "verifier": name, "copies_edited": len(edited),
                         "copies_unedited": len(unedited),
                         **{k: group_stats(v, base, cur) for k, v in groups.items()}})
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(rows, f, indent=2)

    print("| run | verifier | claims with an edited copy | changed (to CONTRADICT) | lost / gained | "
          "confidently wrong clean -> stale |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        e = r["edited_copy_in_context"]
        if not e["n"]:
            continue
        cw = (f"{e['confidently_wrong_clean']} -> {e['confidently_wrong_stale']}"
              if "confidently_wrong_clean" in e else "n/a")
        print(f"| {r['run']} | {r['verifier']} | {e['n']} | {e['predictions_changed']} ({e['moved_to_contradict']}) | "
              f"{e['right_to_wrong']} / {e['wrong_to_right']} (p={e['mcnemar_p']}) | {cw} |")


if __name__ == "__main__":
    main()
