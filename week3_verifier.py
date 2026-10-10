"""Week 3: replace the zero-shot LLM verifier with an NLI cross-encoder, fine-tuned on SciFact train,
and measure how it holds up on the Week 2 corrupted corpora.

Verifiers compared on every Week 2 run (same dense top-3 context as Weeks 1-2):
    qwen            Week 1-2 zero-shot LLM (results already in results/<run>/verification.json)
    nli_zeroshot    off-the-shelf NLI cross-encoder (MNLI/SNLI-trained), no SciFact training
    nli_finetuned   the same model fine-tuned on SciFact train claims, once per training seed

The cross-encoder scores (claim, abstract) pairs as SUPPORT / CONTRADICT / NOT_ENOUGH_INFO.
A claim's label comes from its top-3 retrieved records: take the highest SUPPORT and CONTRADICT
probabilities over the three; if neither reaches a threshold, answer NOT_ENOUGH_INFO, otherwise the
larger one. The threshold is tuned for macro-F1 on a held-out 20% of the train claims (never on dev),
using the same dense top-3 context (BGE-small) that dev claims get.

Training pairs (SciFact train, 809 claims, disjoint from dev): each gold abstract with its label;
for claims without evidence, their cited abstracts as NOT_ENOUGH_INFO; for claims with evidence, the
top BM25 abstract that is not gold, as NOT_ENOUGH_INFO (hard negative). The train/held-out split is
fixed; --seed changes only the training run (data order, dropout).

Usage:
    python week3_verifier.py all --seed 0    # train, evaluate on clean_rerun + 12 Week 2 runs, summarise
    python week3_verifier.py train --seed 1
    python week3_verifier.py evaluate --seed 1
    python week3_verifier.py summary         # aggregates every seed (and the reranker, if run)
Seed 0 also evaluates the untuned model (nli_zeroshot).
Outputs: results/week3/seed<N>/ (training log, per-run predictions), results/week3/nli_zeroshot/,
results/week3/summary.json; the model is saved to models/nli_finetuned_seed<N>/.
"""
import argparse
import glob
import json
import os
import random
import subprocess
import sys

import numpy as np

from compare_week2 import mcnemar_exact
from run_week2 import settings
from week1_baseline import CLEAN_CORPUS, DATA_DIR, LABELS, gold_label, read_jsonl

# WEEK3_BASE_MODEL overrides the base model (e.g. a small local model for testing).
BASE_MODEL = os.environ.get("WEEK3_BASE_MODEL", "cross-encoder/nli-deberta-v3-xsmall")
OUT_DIR = os.path.join("results", "week3")
ZEROSHOT_DIR = os.path.join(OUT_DIR, "nli_zeroshot")
MAX_LEN = 384
SPLIT_SEED = 0
RUNS = ["clean_rerun"] + [name for name, _ in settings()]


def seed_dir(seed):
    return os.path.join(OUT_DIR, f"seed{seed}")


def model_dir(seed):
    return os.path.join("models", f"nli_finetuned_seed{seed}")


def doc_text(d):
    return d["title"] + ". " + " ".join(d["abstract"])


def nli_label_index(model):
    """Map the model's NLI labels (entailment/contradiction/neutral) to our label order."""
    names = {i: n.lower() for i, n in model.config.id2label.items()}
    find = lambda key: next(i for i, n in names.items() if key in n)
    return [find("entail"), find("contra"), find("neutral")]  # = LABELS order


def score_pairs(model, tokenizer, pairs, batch_size=32):
    """Return an (n, 3) array of [SUPPORT, CONTRADICT, NEI] probabilities."""
    import torch
    order = nli_label_index(model)
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(pairs), batch_size):
            claims, texts = zip(*pairs[i:i + batch_size])
            enc = tokenizer(list(claims), list(texts), truncation="only_second", max_length=MAX_LEN,
                            padding=True, return_tensors="pt")
            probs = torch.softmax(model(**enc).logits, dim=-1)[:, order]
            out.append(probs.numpy())
    return np.concatenate(out) if out else np.zeros((0, 3))


def decide(probs, threshold):
    """Claim label from its records' probabilities (rows: records; cols: SUPPORT, CONTRADICT, NEI)."""
    s, c = probs[:, 0].max(), probs[:, 1].max()
    if max(s, c) < threshold:
        return "NOT_ENOUGH_INFO"
    return "SUPPORT" if s >= c else "CONTRADICT"


def macro_f1(gold, pred):
    from sklearn.metrics import f1_score
    return f1_score(gold, pred, labels=LABELS, average="macro")


def tune_threshold(claim_probs, golds):
    grid = [round(t, 2) for t in np.arange(0.05, 1.0, 0.05)]
    scores = [(macro_f1(golds, [decide(p, t) for p in claim_probs]), t) for t in grid]
    best_f1, best_t = max(scores)
    return best_t, best_f1


# ---------------------------------------------------------------- training

def split_train_claims():
    """SciFact train claims and the fixed set of held-out claim ids (20%)."""
    claims = read_jsonl(os.path.join(DATA_DIR, "data", "claims_train.jsonl"))
    order = list(range(len(claims)))
    random.Random(SPLIT_SEED).shuffle(order)
    return claims, {claims[i]["id"] for i in order[:len(claims) // 5]}


def build_training_data():
    from rank_bm25 import BM25Okapi
    from week1_baseline import dense_retrieve

    corpus = read_jsonl(CLEAN_CORPUS)
    by_id = {d["doc_id"]: d for d in corpus}
    ids = [d["doc_id"] for d in corpus]
    bm25 = BM25Okapi([doc_text(d).lower().split() for d in corpus])
    claims, val_ids = split_train_claims()

    pairs = {"train": [], "val": []}
    for c in claims:
        split = "val" if c["id"] in val_ids else "train"
        gold = {int(k): v[0]["label"] for k, v in c.get("evidence", {}).items() if v}
        if gold:
            for d, lab in gold.items():
                pairs[split].append((c["claim"], doc_text(by_id[d]), lab))
            scores = bm25.get_scores(c["claim"].lower().split())
            neg = next(ids[i] for i in np.argsort(-scores) if ids[i] not in gold)
            pairs[split].append((c["claim"], doc_text(by_id[neg]), "NOT_ENOUGH_INFO"))
        else:
            for d in c.get("cited_doc_ids", []):
                if d in by_id:
                    pairs[split].append((c["claim"], doc_text(by_id[d]), "NOT_ENOUGH_INFO"))

    # Held-out claims get the same context dev claims get: dense (BGE-small) top-3.
    val = [c for c in claims if c["id"] in val_ids]
    ranked = dense_retrieve([doc_text(d) for d in corpus], ids, [c["claim"] for c in val], 3)
    val_claims = [{"claim": c["claim"], "gold": gold_label(c), "texts": [doc_text(by_id[d]) for d in top]}
                  for c, top in zip(val, ranked)]
    return pairs, val_claims


def claim_probs_for(model, tokenizer, val_claims):
    flat = [(vc["claim"], t) for vc in val_claims for t in vc["texts"]]
    probs = score_pairs(model, tokenizer, flat)
    out, i = [], 0
    for vc in val_claims:
        out.append(probs[i:i + len(vc["texts"])])
        i += len(vc["texts"])
    return out


def train(seed, epochs=3, lr=2e-5, batch_size=16):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    pairs, val_claims = build_training_data()
    golds = [v["gold"] for v in val_claims]
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL)
    order = nli_label_index(model)  # our label index -> model output index
    to_model = {lab: order[i] for i, lab in enumerate(LABELS)}
    log = {"base_model": BASE_MODEL, "seed": seed, "n_pairs": {k: len(v) for k, v in pairs.items()},
           "label_counts": {k: {lab: sum(p[2] == lab for p in v) for lab in LABELS} for k, v in pairs.items()},
           "threshold_context": "dense BGE-small top-3", "epochs": []}

    if seed == 0:
        # Threshold for the untouched model (the nli_zeroshot verifier).
        zs_t, zs_f1 = tune_threshold(claim_probs_for(model, tokenizer, val_claims), golds)
        os.makedirs(ZEROSHOT_DIR, exist_ok=True)
        with open(os.path.join(ZEROSHOT_DIR, "threshold.json"), "w") as f:
            json.dump({"model": BASE_MODEL, "threshold": zs_t, "val_claim_macro_f1": round(zs_f1, 4)}, f, indent=2)
        print("zero-shot threshold", zs_t, "val macro-F1", round(zs_f1, 4), flush=True)

    data = list(pairs["train"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * ((len(data) + batch_size - 1) // batch_size)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * steps), steps)
    rng = random.Random(seed)
    for epoch in range(epochs):
        model.train()
        rng.shuffle(data)
        total = 0.0
        for i in range(0, len(data), batch_size):
            claims, texts, labs = zip(*data[i:i + batch_size])
            enc = tokenizer(list(claims), list(texts), truncation="only_second", max_length=MAX_LEN,
                            padding=True, return_tensors="pt")
            labels = torch.tensor([to_model[l] for l in labs])
            loss = model(**enc, labels=labels).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total += loss.item() * len(labs)
        vp = score_pairs(model, tokenizer, [(c, t) for c, t, _ in pairs["val"]])
        val_acc = float(np.mean([LABELS[int(np.argmax(p))] == l for p, (_, _, l) in zip(vp, pairs["val"])]))
        log["epochs"].append({"epoch": epoch + 1, "train_loss": round(total / len(data), 4),
                              "val_pair_accuracy": round(val_acc, 4)})
        print(log["epochs"][-1], flush=True)

    ft_t, ft_f1 = tune_threshold(claim_probs_for(model, tokenizer, val_claims), golds)
    log["threshold"] = {"threshold": ft_t, "val_claim_macro_f1": round(ft_f1, 4)}
    print("fine-tuned threshold", ft_t, "val macro-F1", round(ft_f1, 4), flush=True)
    model.save_pretrained(model_dir(seed))
    tokenizer.save_pretrained(model_dir(seed))
    os.makedirs(seed_dir(seed), exist_ok=True)
    with open(os.path.join(seed_dir(seed), "training.json"), "w") as f:
        json.dump(log, f, indent=2)


# ---------------------------------------------------------------- evaluation

def corpus_for(run):
    """The corpus a Week 2 run used; corrupted ones are regenerated (seeded) and checked."""
    if run == "clean_rerun":
        return read_jsonl(CLEAN_CORPUS)
    path = os.path.join(DATA_DIR, "corrupted", run, "corpus.jsonl")
    if not os.path.exists(path):
        args = dict(settings())[run]
        subprocess.run([sys.executable, "corrupt_corpus.py"] + args, check=True, stdout=subprocess.DEVNULL)
    with open(os.path.join(DATA_DIR, "corrupted", run, "manifest.json")) as f:
        regenerated = json.load(f)["changes"]
    with open(os.path.join("results", run, "manifest.json")) as f:
        if json.load(f)["changes"] != regenerated:
            raise RuntimeError(f"{run}: regenerated corpus differs from the Week 2 one")
    return read_jsonl(path)


def dense_rankings(run):
    with open(os.path.join("results", run, "dense_rankings.json")) as f:
        return json.load(f)


def evaluate_verifier(tag, model_path, threshold, out_root, rankings_for=dense_rankings):
    """Verify every dev claim of every run from its top-3 records; one verification.json per run."""
    from sklearn.metrics import accuracy_score, f1_score
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    claims = read_jsonl(os.path.join(DATA_DIR, "data", "claims_dev.jsonl"))
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    for run in RUNS:
        by_id = {d["doc_id"]: d for d in corpus_for(run)}
        rankings = rankings_for(run)
        tops = [rankings[str(c["id"])][:3] for c in claims]
        probs = score_pairs(model, tokenizer, [(c["claim"], doc_text(by_id[d])) for c, top in zip(claims, tops)
                                               for d in top])
        records, gold, pred = [], [], []
        for k, (c, top) in enumerate(zip(claims, tops)):
            p = probs[3 * k:3 * k + 3]
            g, y = gold_label(c), decide(p, threshold)
            gold.append(g)
            pred.append(y)
            records.append({"id": c["id"], "gold": g, "pred": y, "top_docs": top,
                            "probs": [[round(float(x), 4) for x in row] for row in p]})
        summary = {"verifier": tag, "model": model_path, "threshold": threshold, "n_claims": len(gold),
                   "accuracy": round(accuracy_score(gold, pred), 4),
                   "macro_f1": round(f1_score(gold, pred, labels=LABELS, average="macro"), 4)}
        os.makedirs(os.path.join(out_root, run), exist_ok=True)
        with open(os.path.join(out_root, run, "verification.json"), "w") as f:
            json.dump({"summary": summary, "predictions": records}, f, indent=2)
        print(tag, run, summary["accuracy"], summary["macro_f1"], flush=True)


def evaluate(seed):
    with open(os.path.join(seed_dir(seed), "training.json")) as f:
        threshold = json.load(f)["threshold"]["threshold"]
    evaluate_verifier(f"nli_finetuned_seed{seed}", model_dir(seed), threshold,
                      os.path.join(seed_dir(seed), "nli_finetuned"))
    if seed == 0:
        with open(os.path.join(ZEROSHOT_DIR, "threshold.json")) as f:
            threshold = json.load(f)["threshold"]
        evaluate_verifier("nli_zeroshot", BASE_MODEL, threshold, ZEROSHOT_DIR)


# ---------------------------------------------------------------- summary

def load_preds(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    return data["summary"], {r["id"]: r for r in data["predictions"]}


def paired(base, cur):
    r2w = sum(base[i]["pred"] == base[i]["gold"] and cur[i]["pred"] != cur[i]["gold"] for i in base)
    w2r = sum(base[i]["pred"] != base[i]["gold"] and cur[i]["pred"] == cur[i]["gold"] for i in base)
    return {"right_to_wrong": r2w, "wrong_to_right": w2r, "mcnemar_p": round(mcnemar_exact(r2w, w2r), 3)}


def mean_sd(values):
    return {"mean": round(float(np.mean(values)), 4), "sd": round(float(np.std(values, ddof=1)), 4)
            if len(values) > 1 else None, "values": values}


def summary():
    seeds = sorted(int(p.rsplit("seed", 1)[1]) for p in glob.glob(os.path.join(OUT_DIR, "seed*")))
    sources = {"qwen": lambda run: os.path.join("results", run, "verification.json"),
               "nli_zeroshot": lambda run: os.path.join(ZEROSHOT_DIR, run, "verification.json")}
    for s in seeds:
        sources[f"nli_finetuned_seed{s}"] = (lambda s: lambda run: os.path.join(
            seed_dir(s), "nli_finetuned", run, "verification.json"))(s)
    for variant in ("zeroshot", "finetuned"):
        root = os.path.join(OUT_DIR, "reranker", variant)
        if os.path.isdir(root):
            sources[f"reranked_{variant}+nli_finetuned_seed0"] = (lambda root: lambda run: os.path.join(
                root, run, "verification.json"))(root)

    clean = {tag: load_preds(path("clean_rerun")) for tag, path in sources.items()}
    rows = []
    for run in RUNS:
        row = {"run": run}
        for tag, path in sources.items():
            loaded = load_preds(path(run))
            if loaded is None:
                continue
            s, preds = loaded
            row[tag] = {"accuracy": s["accuracy"], "macro_f1": s["macro_f1"]}
            if run != "clean_rerun" and clean[tag] is not None:
                row[tag]["vs_own_clean"] = paired(clean[tag][1], preds)
        ft = [row[f"nli_finetuned_seed{s}"] for s in seeds if f"nli_finetuned_seed{s}" in row]
        if ft:
            row["nli_finetuned_over_seeds"] = {"accuracy": mean_sd([r["accuracy"] for r in ft]),
                                               "macro_f1": mean_sd([r["macro_f1"] for r in ft])}
        rows.append(row)

    head_to_head = {}
    for tag in sources:
        if tag != "qwen" and clean.get(tag) and clean.get("qwen"):
            head_to_head[f"{tag}_vs_qwen"] = paired(clean["qwen"][1], clean[tag][1])
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump({"seeds": seeds, "runs": rows, "clean_head_to_head": head_to_head}, f, indent=2)

    fmt = lambda r, k: f"{r[k]['accuracy']}/{r[k]['macro_f1']}" if k in r else ""
    cols = ["qwen", "nli_zeroshot"] + [f"nli_finetuned_seed{s}" for s in seeds]
    print("| run | " + " | ".join(cols) + " | fine-tuned mean acc (sd) |")
    print("|" + "---|" * (len(cols) + 2))
    for r in rows:
        agg = r.get("nli_finetuned_over_seeds", {}).get("accuracy")
        print("| " + " | ".join([r["run"]] + [fmt(r, k) for k in cols]
                                + [f"{agg['mean']} ({agg['sd']})" if agg else ""]) + " |")
    print(json.dumps(head_to_head, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["train", "evaluate", "summary", "all"])
    parser.add_argument("--seed", type=int, default=0, help="training seed (0 also evaluates the untuned model)")
    args = parser.parse_args()
    if args.step in ("train", "all"):
        train(args.seed)
    if args.step in ("evaluate", "all"):
        evaluate(args.seed)
    if args.step in ("summary", "all"):
        summary()
