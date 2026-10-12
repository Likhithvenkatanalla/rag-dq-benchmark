"""Week 6 summary: results/week6/summary.json and printed tables (run via `python week6.py summary`).

    seeds    per setting, corruption seeds 0-4: dense/BM25 recall@1, fine-tuned verifier (training seed 0)
             accuracy and right->wrong / wrong->right vs clean; for stale, the claims whose top-3 held
             an edited copy (answers changed, moved to CONTRADICT, confidently wrong clean -> stale)
    qwen3b   Qwen2.5-3B accuracy per run, paired vs its own clean run, and vs Qwen2.5-1.5B on clean
    climate  Climate-FEVER retrieval, Qwen2.5-1.5B and the three fine-tuned verifiers per run, paired
             vs clean, calibration, and the edited-stale-copy analysis
"""
import json
import os
from collections import Counter

import numpy as np

from week3_verifier import load_preds, paired
from week4_calibration import CONFIDENT, calibration, claim_confidence
from week4_stale import group_stats
from week6 import ALL_SEEDS, CLIMATE_DIR, CLIMATE_RUNS, DEFECTS, OUT_DIR, QWEN3B_DIR, RATES

FT0 = os.path.join("results", "week3", "seed0", "nli_finetuned")


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def mean_sd(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return {"mean": round(float(np.mean(values)), 4),
            "sd": round(float(np.std(values, ddof=1)), 4) if len(values) > 1 else None, "n": len(values)}


def stale_groups(manifest, base, cur):
    """Claims whose top-3 held an edited stale copy, vs the same claims on the clean corpus."""
    changes = manifest["changes"]
    edited = {c["doc_id"] for c in changes if c.get("n_edits", 0) > 0}
    ids = [i for i, r in cur.items() if set(r["top_docs"]) & edited]
    return {"copies": len(changes), "copies_edited": len(edited), **group_stats(ids, base, cur)}


def conf_wrong(preds):
    return round(float(np.mean([claim_confidence(r) >= CONFIDENT and r["pred"] != r["gold"]
                                for r in preds.values()])), 4)


# ---------------------------------------------------------------- seeds

def seeds_section():
    clean = load_preds(os.path.join(FT0, "clean_rerun", "verification.json"))
    clean_ret = load_json(os.path.join("results", "clean_rerun", "retrieval.json"))
    out = {"clean": {"dense_recall@1": clean_ret["dense_bge_small"]["recall@1"],
                     "bm25_recall@1": clean_ret["bm25"]["recall@1"],
                     "nli_accuracy": clean[0]["accuracy"], "confidently_wrong": conf_wrong(clean[1])},
           "settings": []}
    for defect in DEFECTS:
        for rate in RATES:
            row = {"setting": f"{defect}_r{rate}", "seeds": {}}
            for s in ALL_SEEDS:
                run = f"{defect}_r{rate}_gold_s{s}"
                ret = load_json(os.path.join("results", run, "retrieval.json"))
                ver = load_preds(os.path.join(FT0, run, "verification.json"))
                if ret is None or ver is None:
                    continue
                entry = {"dense_recall@1": ret["dense_bge_small"]["recall@1"],
                         "dense_recall@10": ret["dense_bge_small"]["recall@10"],
                         "bm25_recall@1": ret["bm25"]["recall@1"], "bm25_recall@10": ret["bm25"]["recall@10"],
                         "nli_accuracy": ver[0]["accuracy"], "nli_macro_f1": ver[0]["macro_f1"],
                         "confidently_wrong": conf_wrong(ver[1]), "vs_clean": paired(clean[1], ver[1])}
                if defect == "stale":
                    entry["edited_copy_in_context"] = stale_groups(
                        load_json(os.path.join("results", run, "manifest.json")), clean[1], ver[1])
                row["seeds"][s] = entry
            seeds = row["seeds"].values()
            row["over_seeds"] = {
                "dense_recall@1": mean_sd([e["dense_recall@1"] for e in seeds]),
                "bm25_recall@1": mean_sd([e["bm25_recall@1"] for e in seeds]),
                "nli_accuracy": mean_sd([e["nli_accuracy"] for e in seeds]),
                "confidently_wrong": mean_sd([e["confidently_wrong"] for e in seeds]),
                "right_to_wrong_total": sum(e["vs_clean"]["right_to_wrong"] for e in seeds),
                "wrong_to_right_total": sum(e["vs_clean"]["wrong_to_right"] for e in seeds),
                "seeds_net_loss": sum(e["vs_clean"]["right_to_wrong"] > e["vs_clean"]["wrong_to_right"] for e in seeds),
                "seeds_net_gain": sum(e["vs_clean"]["right_to_wrong"] < e["vs_clean"]["wrong_to_right"] for e in seeds),
            }
            if defect == "stale":
                groups = [e["edited_copy_in_context"] for e in seeds if e["edited_copy_in_context"]["n"]]
                row["over_seeds"]["edited_copy_in_context"] = {
                    k: sum(g[k] for g in groups)
                    for k in ("n", "predictions_changed", "moved_to_contradict", "right_to_wrong", "wrong_to_right",
                              "confidently_wrong_clean", "confidently_wrong_stale")}
                row["over_seeds"]["seeds_more_confidently_wrong"] = sum(
                    g["confidently_wrong_stale"] > g["confidently_wrong_clean"] for g in groups)
                row["over_seeds"]["seeds_fewer_confidently_wrong"] = sum(
                    g["confidently_wrong_stale"] < g["confidently_wrong_clean"] for g in groups)
            out["settings"].append(row)

    print("\n## Corruption seeds 0-4 (fine-tuned verifier, training seed 0)")
    print(f"clean: dense R@1 {out['clean']['dense_recall@1']}, NLI acc {out['clean']['nli_accuracy']}")
    print("| setting | seeds | dense R@1 mean (sd) | NLI acc mean (sd) | lost / gained (all seeds) | "
          "seeds net loss / gain |")
    print("|---|---|---|---|---|---|")
    for r in out["settings"]:
        o = r["over_seeds"]
        if not o["nli_accuracy"]:
            continue
        print(f"| {r['setting']} | {o['nli_accuracy']['n']} | {o['dense_recall@1']['mean']} ({o['dense_recall@1']['sd']}) | "
              f"{o['nli_accuracy']['mean']} ({o['nli_accuracy']['sd']}) | {o['right_to_wrong_total']} / "
              f"{o['wrong_to_right_total']} | {o['seeds_net_loss']} / {o['seeds_net_gain']} |")
    print("| stale setting | claims with edited copy | changed (to CONTRADICT) | confidently wrong clean -> stale | "
          "seeds more / fewer conf. wrong |")
    print("|---|---|---|---|---|")
    for r in out["settings"]:
        e = r["over_seeds"].get("edited_copy_in_context")
        if e:
            print(f"| {r['setting']} | {e['n']} | {e['predictions_changed']} ({e['moved_to_contradict']}) | "
                  f"{e['confidently_wrong_clean']} -> {e['confidently_wrong_stale']} | "
                  f"{r['over_seeds']['seeds_more_confidently_wrong']} / {r['over_seeds']['seeds_fewer_confidently_wrong']} |")
    return out


# ---------------------------------------------------------------- Qwen2.5-3B

def qwen3b_section():
    from week3_verifier import RUNS
    clean = load_preds(os.path.join(QWEN3B_DIR, "clean_rerun", "verification.json"))
    small_clean = load_preds(os.path.join("results", "clean_rerun", "verification.json"))
    if clean is None:
        return None
    out = {"clean_vs_qwen1.5b": paired(small_clean[1], clean[1]),
           "clean_pred_distribution": dict(Counter(r["pred"] for r in clean[1].values())), "runs": []}
    for run in RUNS:
        cur = load_preds(os.path.join(QWEN3B_DIR, run, "verification.json"))
        small = load_preds(os.path.join("results", run, "verification.json"))
        if cur is None:
            continue
        row = {"run": run, "accuracy": cur[0]["accuracy"], "macro_f1": cur[0]["macro_f1"],
               "qwen1.5b_accuracy": small[0]["accuracy"], "qwen1.5b_macro_f1": small[0]["macro_f1"]}
        if run != "clean_rerun":
            row["vs_clean"] = paired(clean[1], cur[1])
            row["qwen1.5b_vs_clean"] = paired(small_clean[1], small[1])
        if run.startswith("stale"):
            row["edited_copy_in_context"] = stale_groups(
                load_json(os.path.join("results", run, "manifest.json")), clean[1], cur[1])
        out["runs"].append(row)

    print("\n## Qwen2.5-3B vs Qwen2.5-1.5B (zero-shot verifier)")
    print(f"clean head-to-head (1.5B -> 3B, lost / gained): {out['clean_vs_qwen1.5b']}; "
          f"3B predictions: {out['clean_pred_distribution']}")
    print("| run | 3B acc / F1 | 3B lost / gained (p) | 1.5B acc / F1 | 1.5B lost / gained (p) |")
    print("|---|---|---|---|---|")
    for r in out["runs"]:
        v, w = r.get("vs_clean"), r.get("qwen1.5b_vs_clean")
        fmt = lambda x: f"{x['right_to_wrong']} / {x['wrong_to_right']} ({x['mcnemar_p']})" if x else "-"
        print(f"| {r['run']} | {r['accuracy']} / {r['macro_f1']} | {fmt(v)} | "
              f"{r['qwen1.5b_accuracy']} / {r['qwen1.5b_macro_f1']} | {fmt(w)} |")
    return out


# ---------------------------------------------------------------- Climate-FEVER

def climate_section():
    root = CLIMATE_DIR
    verifiers = {"qwen": lambda run: os.path.join(root, run, "verification.json")}
    for s in (0, 1, 2):
        verifiers[f"nli_finetuned_seed{s}"] = (lambda s: lambda run: os.path.join(
            root, f"nli_finetuned_seed{s}", run, "verification.json"))(s)
    clean = {v: load_preds(path("clean_rerun")) for v, path in verifiers.items()}
    if clean["qwen"] is None:
        return None
    analysis = load_json(os.path.join(root, "clean_rerun", "analysis.json"))
    out = {"majority_baseline": analysis["majority_baseline"], "gold_distribution": analysis["gold_distribution"],
           "qwen_clean_pred_distribution": analysis["pred_distribution"], "runs": []}
    if clean["nli_finetuned_seed0"]:
        out["clean_head_to_head"] = {f"{v}_vs_qwen": paired(clean["qwen"][1], clean[v][1])
                                     for v in verifiers if v != "qwen" and clean[v]}
    for run in CLIMATE_RUNS:
        ret = load_json(os.path.join(root, run, "retrieval.json"))
        if ret is None:
            continue
        row = {"run": run, "retrieval": ret}
        manifest = load_json(os.path.join(root, run, "manifest.json"))
        if manifest and "stale_copies_without_edits" in manifest:
            row["stale_copies"] = len(manifest["changes"])
            row["stale_copies_without_edits"] = manifest["stale_copies_without_edits"]
        for v, path in verifiers.items():
            cur = load_preds(path(run))
            if cur is None:
                continue
            e = {"accuracy": cur[0]["accuracy"], "macro_f1": cur[0]["macro_f1"]}
            if "gold_doc_in_context_rate" in cur[0]:
                e["gold_in_top3_rate"] = cur[0]["gold_doc_in_context_rate"]
            if run != "clean_rerun" and clean[v]:
                e["vs_clean"] = paired(clean[v][1], cur[1])
            if v != "qwen":
                preds = list(cur[1].values())
                cal = calibration([claim_confidence(r) for r in preds], [r["pred"] == r["gold"] for r in preds])
                e.update({k: cal[k] for k in ("mean_confidence", "ece", "brier", "confidently_wrong")})
            if run.startswith("stale") and clean[v]:
                e["edited_copy_in_context"] = stale_groups(manifest, clean[v][1], cur[1])
            row[v] = e
        ft = [row[f"nli_finetuned_seed{s}"]["accuracy"] for s in (0, 1, 2) if f"nli_finetuned_seed{s}" in row]
        row["nli_finetuned_accuracy"] = mean_sd(ft)
        out["runs"].append(row)

    print("\n## Climate-FEVER")
    print(f"majority baseline {out['majority_baseline']}; gold {out['gold_distribution']}; "
          f"Qwen predictions (clean) {out['qwen_clean_pred_distribution']}")
    print(json.dumps(out.get("clean_head_to_head"), indent=1))
    print("| run | BM25 R@1/R@10 | dense R@1/R@10 | gold in top-3 | Qwen acc (lost/gained) | "
          "NLI ft acc s0/s1/s2 | NLI s0 lost/gained | NLI ECE / conf-wrong (s0) |")
    print("|---|---|---|---|---|---|---|---|")
    for r in out["runs"]:
        q, n0 = r.get("qwen", {}), r.get("nli_finetuned_seed0", {})
        lg = lambda e: (f" ({e['vs_clean']['right_to_wrong']}/{e['vs_clean']['wrong_to_right']}, "
                        f"p={e['vs_clean']['mcnemar_p']})") if "vs_clean" in e else ""
        nli = " / ".join(str(r.get(f"nli_finetuned_seed{s}", {}).get("accuracy", "-")) for s in (0, 1, 2))
        print(f"| {r['run']} | {r['retrieval']['bm25']['recall@1']}/{r['retrieval']['bm25']['recall@10']} | "
              f"{r['retrieval']['dense_bge_small']['recall@1']}/{r['retrieval']['dense_bge_small']['recall@10']} | "
              f"{q.get('gold_in_top3_rate', '-')} | {q.get('accuracy', '-')}{lg(q)} | {nli} | {lg(n0).strip(' ()') or '-'} | "
              f"{n0.get('ece', '-')} / {n0.get('confidently_wrong', '-')} |")
    for r in out["runs"]:
        if r["run"].startswith("stale"):
            for v in verifiers:
                e = r.get(v, {}).get("edited_copy_in_context")
                if e and e["n"]:
                    cw = (f", conf. wrong {e['confidently_wrong_clean']} -> {e['confidently_wrong_stale']}"
                          if "confidently_wrong_clean" in e else "")
                    print(f"{r['run']} {v}: {e['n']} claims with an edited copy, {e['predictions_changed']} changed "
                          f"({e['moved_to_contradict']} to CONTRADICT), lost/gained {e['right_to_wrong']}/"
                          f"{e['wrong_to_right']}{cw}")
    return out


def main():
    out = {"seeds": seeds_section(), "qwen3b": qwen3b_section(), "climate_fever": climate_section()}
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote", os.path.join(OUT_DIR, "summary.json"))


if __name__ == "__main__":
    main()
