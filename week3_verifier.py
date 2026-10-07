"""Week 3: replace the zero-shot LLM verifier with an NLI cross-encoder, fine-tuned on SciFact train,
and measure how it holds up on the Week 2 corrupted corpora.

Verifiers compared on every Week 2 run (same dense top-3 context as Weeks 1-2):
    qwen            Week 1-2 zero-shot LLM (results already in results/<run>/verification.json)
    nli_zeroshot    off-the-shelf NLI cross-encoder (MNLI/SNLI-trained), no SciFact training
    nli_finetuned   the same model fine-tuned on SciFact train claims

The cross-encoder scores (claim, abstract) pairs as SUPPORT / CONTRADICT / NOT_ENOUGH_INFO.
A claim's label comes from its top-3 retrieved records: take the highest SUPPORT and CONTRADICT
probabilities over the three; if neither reaches a threshold, answer NOT_ENOUGH_INFO, otherwise the
larger one. The threshold is tuned for macro-F1 on a held-out 20% of the train claims (never on dev).

Training pairs (SciFact train, 809 claims, disjoint from dev): each gold abstract with its label;
for claims without evidence, their cited abstracts as NOT_ENOUGH_INFO; for claims with evidence, the
top BM25 abstract that is not gold, as NOT_ENOUGH_INFO (hard negative).

Usage:
    python week3_verifier.py train       # fine-tune, tune thresholds; model saved to models/
    python week3_verifier.py evaluate    # both NLI verifiers on clean_rerun + the 12 Week 2 runs
    python week3_verifier.py summary     # results/week3/summary.json + paired comparisons
    python week3_verifier.py all
"""
import argparse
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
FINETUNED_DIR = os.path.join("models", "nli_finetuned")
OUT_DIR = os.path.join("results", "week3")
MAX_LEN = 384
SEED = 0
RUNS = ["clean_rerun"] + [name for name, _ in settings()]


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

def build_training_data():
    from rank_bm25 import BM25Okapi
    corpus = read_jsonl(CLEAN_CORPUS)
    by_id = {d["doc_id"]: d for d in corpus}
    ids = [d["doc_id"] for d in corpus]
    bm25 = BM25Okapi([doc_text(d).lower().split() for d in corpus])
    claims = read_jsonl(os.path.join(DATA_DIR, "data", "claims_train.jsonl"))

    rng = random.Random(SEED)
    order = list(range(len(claims)))
    rng.shuffle(order)
    val_ids = {claims[i]["id"] for i in order[:len(claims) // 5]}

    pairs = {"train": [], "val": []}
    val_claims = []
    for c in claims:
        split = "val" if c["id"] in val_ids else "train"
        gold = {int(k): v[0]["label"] for k, v in c.get("evidence", {}).items() if v}
        scores = bm25.get_scores(c["claim"].lower().split())
        ranked = [ids[i] for i in np.argsort(-scores)[:20]]
        if gold:
            for d, lab in gold.items():
                pairs[split].append((c["claim"], doc_text(by_id[d]), lab))
            neg = next(d for d in ranked if d not in gold)
            pairs[split].append((c["claim"], doc_text(by_id[neg]), "NOT_ENOUGH_INFO"))
        else:
            for d in c.get("cited_doc_ids", []):
                if d in by_id:
                    pairs[split].append((c["claim"], doc_text(by_id[d]), "NOT_ENOUGH_INFO"))
        if split == "val":
            # Threshold tuning uses BM25 top-3 as context (dense retrieval is not run on train claims).
            val_claims.append({"claim": c["claim"], "gold": gold_label(c),
                               "texts": [doc_text(by_id[d]) for d in ranked[:3]]})
    return pairs, val_claims


def claim_probs_for(model, tokenizer, val_claims):
    flat = [(vc["claim"], t) for vc in val_claims for t in vc["texts"]]
    probs = score_pairs(model, tokenizer, flat)
    out, i = [], 0
    for vc in val_claims:
        out.append(probs[i:i + len(vc["texts"])])
        i += len(vc["texts"])
    return out


def train(epochs=3, lr=2e-5, batch_size=16):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    pairs, val_claims = build_training_data()
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL)
    order = nli_label_index(model)  # our label index -> model output index
    to_model = {lab: order[i] for i, lab in enumerate(LABELS)}
    log = {"base_model": BASE_MODEL, "n_pairs": {k: len(v) for k, v in pairs.items()},
           "label_counts": {k: {lab: sum(p[2] == lab for p in v) for lab in LABELS} for k, v in pairs.items()},
           "epochs": []}

    # Zero-shot threshold first, on the untouched model.
    os.makedirs(OUT_DIR, exist_ok=True)
    zs_t, zs_f1 = tune_threshold(claim_probs_for(model, tokenizer, val_claims), [v["gold"] for v in val_claims])
    log["zeroshot_threshold"] = {"threshold": zs_t, "val_claim_macro_f1": round(zs_f1, 4)}
    print("zero-shot threshold", zs_t, "val macro-F1", round(zs_f1, 4), flush=True)

    data = pairs["train"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * ((len(data) + batch_size - 1) // batch_size)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * steps), steps)
    rng = random.Random(SEED)
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

    ft_t, ft_f1 = tune_threshold(claim_probs_for(model, tokenizer, val_claims), [v["gold"] for v in val_claims])
    log["finetuned_threshold"] = {"threshold": ft_t, "val_claim_macro_f1": round(ft_f1, 4)}
    print("fine-tuned threshold", ft_t, "val macro-F1", round(ft_f1, 4), flush=True)
    model.save_pretrained(FINETUNED_DIR)
    tokenizer.save_pretrained(FINETUNED_DIR)
    with open(os.path.join(OUT_DIR, "training.json"), "w") as f:
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


def evaluate():
    from sklearn.metrics import accuracy_score, f1_score
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    with open(os.path.join(OUT_DIR, "training.json")) as f:
        log = json.load(f)
    verifiers = {"nli_zeroshot": (BASE_MODEL, log["zeroshot_threshold"]["threshold"]),
                 "nli_finetuned": (FINETUNED_DIR, log["finetuned_threshold"]["threshold"])}
    claims = read_jsonl(os.path.join(DATA_DIR, "data", "claims_dev.jsonl"))
    for tag, (path, threshold) in verifiers.items():
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSequenceClassification.from_pretrained(path)
        for run in RUNS:
            by_id = {d["doc_id"]: d for d in corpus_for(run)}
            with open(os.path.join("results", run, "dense_rankings.json")) as f:
                rankings = json.load(f)
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
            summary = {"verifier": tag, "model": path, "threshold": threshold, "n_claims": len(gold),
                       "accuracy": round(accuracy_score(gold, pred), 4),
                       "macro_f1": round(f1_score(gold, pred, labels=LABELS, average="macro"), 4)}
            os.makedirs(os.path.join(OUT_DIR, tag, run), exist_ok=True)
            with open(os.path.join(OUT_DIR, tag, run, "verification.json"), "w") as f:
                json.dump({"summary": summary, "predictions": records}, f, indent=2)
            print(tag, run, summary["accuracy"], summary["macro_f1"], flush=True)


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


def summary():
    sources = {"qwen": lambda run: os.path.join("results", run, "verification.json"),
               "nli_zeroshot": lambda run: os.path.join(OUT_DIR, "nli_zeroshot", run, "verification.json"),
               "nli_finetuned": lambda run: os.path.join(OUT_DIR, "nli_finetuned", run, "verification.json")}
    rows = []
    for run in RUNS:
        row = {"run": run}
        for tag, path in sources.items():
            loaded, base = load_preds(path(run)), load_preds(path("clean_rerun"))
            if loaded is None:
                continue
            s, preds = loaded
            row[f"{tag}_accuracy"], row[f"{tag}_macro_f1"] = s["accuracy"], s["macro_f1"]
            if run != "clean_rerun" and base is not None:
                row[f"{tag}_vs_clean"] = paired(base[1], preds)
        rows.append(row)
    # Verifiers against each other on the clean corpus.
    clean = {tag: load_preds(path("clean_rerun")) for tag, path in sources.items()}
    head_to_head = {}
    if clean["qwen"] and clean["nli_finetuned"]:
        head_to_head["nli_finetuned_vs_qwen"] = paired(clean["qwen"][1], clean["nli_finetuned"][1])
    if clean["nli_zeroshot"] and clean["nli_finetuned"]:
        head_to_head["nli_finetuned_vs_zeroshot"] = paired(clean["nli_zeroshot"][1], clean["nli_finetuned"][1])
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump({"runs": rows, "clean_head_to_head": head_to_head}, f, indent=2)

    print("| run | Qwen acc | Qwen F1 | NLI zero-shot acc | F1 | NLI fine-tuned acc | F1 |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        print("| " + " | ".join([r["run"]] + [str(r.get(k, "")) for k in
              ["qwen_accuracy", "qwen_macro_f1", "nli_zeroshot_accuracy", "nli_zeroshot_macro_f1",
               "nli_finetuned_accuracy", "nli_finetuned_macro_f1"]]) + " |")
    print(json.dumps(head_to_head, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["train", "evaluate", "summary", "all"])
    args = parser.parse_args()
    if args.step in ("train", "all"):
        train()
    if args.step in ("evaluate", "all"):
        evaluate()
    if args.step in ("summary", "all"):
        summary()
