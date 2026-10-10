"""Week 3: rerank the dense top-10 with a cross-encoder, then verify from the reranked top-3.

Rerankers (both score how relevant an abstract is to a claim):
    zeroshot    cross-encoder/ms-marco-MiniLM-L-6-v2 as released (trained on web search, not SciFact)
    finetuned   the same model fine-tuned on SciFact train: gold abstracts are relevant (1); the top
                BM25 abstracts that are not gold are not (0), 4 per claim

Each Week 2 run's committed dense top-10 is reordered; recall@1/3 is computed as in Weeks 1-2
(copies are not gold; chunks count as their parent abstract). Recall@10 cannot change, since only
the top 10 is reordered. The fine-tuned verifier (seed 0) then labels each claim from the new top-3.

Usage:
    python week3_reranker.py train      # fine-tune; model saved to models/reranker_finetuned/
    python week3_reranker.py rerank     # both rerankers on clean_rerun + the 12 Week 2 runs
    python week3_reranker.py verify     # needs models/nli_finetuned_seed0 and results/week3/seed0/
Outputs: results/week3/reranker/<variant>/<run>/{rankings,retrieval,verification}.json,
results/week3/reranker/training.json.
"""
import argparse
import json
import os
import random

import numpy as np

from week1_baseline import CLEAN_CORPUS, DATA_DIR, parent_of, read_jsonl, recall_at_k, to_docs
from week3_verifier import OUT_DIR, RUNS, corpus_for, dense_rankings, doc_text, evaluate_verifier, \
    model_dir, seed_dir, split_train_claims

# WEEK3_BASE_RERANKER overrides the base model (e.g. a small local model for testing).
BASE_RERANKER = os.environ.get("WEEK3_BASE_RERANKER", "cross-encoder/ms-marco-MiniLM-L-6-v2")
FINETUNED_DIR = os.path.join("models", "reranker_finetuned")
RERANK_DIR = os.path.join(OUT_DIR, "reranker")
MAX_LEN = 384
NEGATIVES = 4
SEED = 0


def relevance(model, tokenizer, pairs, batch_size=32):
    """One relevance score per (claim, text) pair."""
    import torch
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(pairs), batch_size):
            claims, texts = zip(*pairs[i:i + batch_size])
            enc = tokenizer(list(claims), list(texts), truncation="only_second", max_length=MAX_LEN,
                            padding=True, return_tensors="pt")
            out.append(model(**enc).logits[:, 0].numpy())
    return np.concatenate(out) if out else np.zeros(0)


def rerank(model, tokenizer, claim, candidates, by_id):
    scores = relevance(model, tokenizer, [(claim, doc_text(by_id[d])) for d in candidates])
    return [candidates[i] for i in np.argsort(-scores, kind="stable")]


def train(epochs=2, lr=2e-5, batch_size=16):
    import torch
    from rank_bm25 import BM25Okapi
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    torch.manual_seed(SEED)
    random.seed(SEED)
    corpus = read_jsonl(CLEAN_CORPUS)
    by_id = {d["doc_id"]: d for d in corpus}
    ids = [d["doc_id"] for d in corpus]
    bm25 = BM25Okapi([doc_text(d).lower().split() for d in corpus])
    claims, val_ids = split_train_claims()

    data, val = [], []
    for c in claims:
        gold = {int(k) for k, v in c.get("evidence", {}).items() if v}
        top = [ids[i] for i in np.argsort(-bm25.get_scores(c["claim"].lower().split()))[:10]]
        if c["id"] in val_ids:
            if gold:
                val.append((c["claim"], top, gold))
            continue
        data += [(c["claim"], doc_text(by_id[d]), 1.0) for d in gold]
        data += [(c["claim"], doc_text(by_id[d]), 0.0) for d in [d for d in top if d not in gold][:NEGATIVES]]

    tokenizer = AutoTokenizer.from_pretrained(BASE_RERANKER)
    model = AutoModelForSequenceClassification.from_pretrained(BASE_RERANKER)

    def val_recall_at_1():
        hits = [len(gold & set(rerank(model, tokenizer, claim, top, by_id)[:1])) / len(gold)
                for claim, top, gold in val]
        return round(float(np.mean(hits)), 4)

    log = {"base_model": BASE_RERANKER, "n_pairs": len(data), "n_positive": sum(p[2] == 1.0 for p in data),
           "val_claims": len(val),
           "val_recall@1_bm25": round(float(np.mean([len(g & set(t[:1])) / len(g) for _, t, g in val])), 4),
           "val_recall@1_reranked_before_training": val_recall_at_1(), "epochs": []}
    print(log, flush=True)

    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * ((len(data) + batch_size - 1) // batch_size)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * steps), steps)
    rng = random.Random(SEED)
    for epoch in range(epochs):
        model.train()
        rng.shuffle(data)
        total = 0.0
        for i in range(0, len(data), batch_size):
            claims_b, texts, labels = zip(*data[i:i + batch_size])
            enc = tokenizer(list(claims_b), list(texts), truncation="only_second", max_length=MAX_LEN,
                            padding=True, return_tensors="pt")
            loss = loss_fn(model(**enc).logits[:, 0], torch.tensor(labels))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total += loss.item() * len(labels)
        log["epochs"].append({"epoch": epoch + 1, "train_loss": round(total / len(data), 4),
                              "val_recall@1_reranked": val_recall_at_1()})
        print(log["epochs"][-1], flush=True)

    model.save_pretrained(FINETUNED_DIR)
    tokenizer.save_pretrained(FINETUNED_DIR)
    os.makedirs(RERANK_DIR, exist_ok=True)
    with open(os.path.join(RERANK_DIR, "training.json"), "w") as f:
        json.dump(log, f, indent=2)


def rerank_all():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    claims = read_jsonl(os.path.join(DATA_DIR, "data", "claims_dev.jsonl"))
    for variant, path in (("zeroshot", BASE_RERANKER), ("finetuned", FINETUNED_DIR)):
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSequenceClassification.from_pretrained(path)
        for run in RUNS:
            corpus = corpus_for(run)
            by_id, parent = {d["doc_id"]: d for d in corpus}, parent_of(corpus)
            dense = dense_rankings(run)
            reranked = {str(c["id"]): rerank(model, tokenizer, c["claim"], dense[str(c["id"])], by_id) for c in claims}
            metrics = {}
            for name, ranks in (("dense", dense), ("reranked", reranked)):
                docs = [to_docs(ranks[str(c["id"])], parent, 10) for c in claims]
                metrics[name] = {f"recall@{k}": round(recall_at_k(docs, claims, k), 4) for k in (1, 3)}
            out = os.path.join(RERANK_DIR, variant, run)
            os.makedirs(out, exist_ok=True)
            with open(os.path.join(out, "rankings.json"), "w") as f:
                json.dump(reranked, f)
            with open(os.path.join(out, "retrieval.json"), "w") as f:
                json.dump({"reranker": path, **metrics}, f, indent=2)
            print(variant, run, metrics, flush=True)


def verify_all():
    with open(os.path.join(seed_dir(0), "training.json")) as f:
        threshold = json.load(f)["threshold"]["threshold"]
    for variant in ("zeroshot", "finetuned"):
        root = os.path.join(RERANK_DIR, variant)

        def rankings_for(run, root=root):
            with open(os.path.join(root, run, "rankings.json")) as f:
                return json.load(f)
        evaluate_verifier(f"reranked_{variant}+nli_finetuned_seed0", model_dir(0), threshold, root, rankings_for)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["train", "rerank", "verify"])
    args = parser.parse_args()
    {"train": train, "rerank": rerank_all, "verify": verify_all}[args.step]()
