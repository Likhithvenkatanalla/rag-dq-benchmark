"""Week 1 baseline: retrieval + claim verification on SciFact.

Research question of the project: how do data-quality defects in a document
collection change the accuracy of retrieval-augmented answers?
Week 1 builds the clean baseline that every later experiment is compared with.

Usage (runs on CPU; a GPU such as a Colab/Kaggle T4 is faster):
    python week1_baseline.py --step download
    python week1_baseline.py --step retrieve
    python week1_baseline.py --step verify

Week 2+: run the same pipeline on a corrupted corpus from corrupt_corpus.py. Results go to
results/<name>/ (name defaults to the corpus folder), leaving the Week 1 files untouched:
    python week1_baseline.py --step retrieve --corpus data/corrupted/<name>/corpus.jsonl
    python week1_baseline.py --step verify --corpus data/corrupted/<name>/corpus.jsonl
"""
import argparse
import json
import os
import tarfile
import urllib.request

import numpy as np

DATA_URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"
DATA_DIR = "data"
CLEAN_CORPUS = os.path.join(DATA_DIR, "data", "corpus.jsonl")
RESULTS_DIR = "results"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
LLM_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
LABELS = ["SUPPORT", "CONTRADICT", "NOT_ENOUGH_INFO"]


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def download():
    os.makedirs(DATA_DIR, exist_ok=True)
    archive = os.path.join(DATA_DIR, "data.tar.gz")
    if not os.path.exists(archive):
        urllib.request.urlretrieve(DATA_URL, archive)
    with tarfile.open(archive) as tar:
        tar.extractall(DATA_DIR)
    print("Files:", os.listdir(os.path.join(DATA_DIR, "data")))


def output_paths(corpus_path, out):
    """Week 1 file names for the clean corpus; results/<out>/ for anything else."""
    if out is None and os.path.abspath(corpus_path) == os.path.abspath(CLEAN_CORPUS):
        return {"retrieval": os.path.join(RESULTS_DIR, "week1_retrieval.json"),
                "rankings": os.path.join(RESULTS_DIR, "dense_rankings.json"),
                "verification": os.path.join(RESULTS_DIR, "week1_verification.json")}
    folder = os.path.join(RESULTS_DIR, out or os.path.basename(os.path.dirname(os.path.abspath(corpus_path))))
    return {"retrieval": os.path.join(folder, "retrieval.json"),
            "rankings": os.path.join(folder, "dense_rankings.json"),
            "verification": os.path.join(folder, "verification.json")}


def load(corpus_path=CLEAN_CORPUS):
    corpus = read_jsonl(corpus_path)
    claims = read_jsonl(os.path.join(DATA_DIR, "data", "claims_dev.jsonl"))
    docs = [d["title"] + ". " + " ".join(d["abstract"]) for d in corpus]
    doc_ids = [d["doc_id"] for d in corpus]
    return corpus, claims, docs, doc_ids


def parent_of(corpus):
    """Chunks (corrupt_corpus.py --defect chunk) map to the document they were cut from;
    every other record, including duplicate and stale copies, is its own document."""
    return {d["doc_id"]: d.get("parent_doc_id", d["doc_id"]) for d in corpus}


def to_docs(ranking, parent, k):
    """First k distinct documents in a ranking of corpus records."""
    seen = []
    for rid in ranking:
        p = parent[rid]
        if p not in seen:
            seen.append(p)
            if len(seen) == k:
                break
    return seen


def gold_label(claim):
    """SciFact dev claims: evidence maps doc_id -> list of {sentences, label}."""
    for entries in claim.get("evidence", {}).values():
        if entries:
            return entries[0]["label"]
    return "NOT_ENOUGH_INFO"


def gold_docs(claim):
    return {int(k) for k in claim.get("evidence", {}).keys()}


def recall_at_k(ranked, claims, k):
    hits, total = 0, 0
    for claim, ranking in zip(claims, ranked):
        gold = gold_docs(claim)
        if not gold:
            continue
        total += 1
        hits += len(gold & set(ranking[:k])) / len(gold)
    return hits / total


def retrieve(corpus_path=CLEAN_CORPUS, out=None, top_k=10, sparse_only=False):
    from rank_bm25 import BM25Okapi

    corpus, claims, docs, doc_ids = load(corpus_path)
    parent = parent_of(corpus)
    # Chunked corpora need a deeper record ranking to fill top_k distinct documents.
    depth = top_k if all(p == r for r, p in parent.items()) else min(len(docs), top_k * 30)
    queries = [c["claim"] for c in claims]

    # Sparse baseline: BM25
    bm25 = BM25Okapi([d.lower().split() for d in docs])
    bm25_ranked = []
    for q in queries:
        scores = bm25.get_scores(q.lower().split())
        bm25_ranked.append([doc_ids[i] for i in np.argsort(-scores)[:depth]])

    rankings = {"bm25": bm25_ranked}
    if not sparse_only:
        rankings["dense_bge_small"] = dense_retrieve(docs, doc_ids, queries, depth)
    results = {}
    for name, ranked in rankings.items():
        doc_ranked = [to_docs(r, parent, top_k) for r in ranked]
        results[name] = {f"recall@{k}": round(recall_at_k(doc_ranked, claims, k), 4) for k in (1, 3, 5, 10)}
    print(json.dumps(results, indent=2))

    paths = output_paths(corpus_path, out)
    os.makedirs(os.path.dirname(paths["retrieval"]), exist_ok=True)
    with open(paths["retrieval"], "w") as f:
        json.dump(results, f, indent=2)
    if sparse_only:
        return
    # Record ids (chunk ids for a chunked corpus): these are what verify puts in the LLM context.
    with open(paths["rankings"], "w") as f:
        json.dump({str(c["id"]): r[:top_k] for c, r in zip(claims, rankings["dense_bge_small"])}, f)


def dense_retrieve(docs, doc_ids, queries, depth):
    """BGE embeddings + FAISS inner product (cosine on normalised vectors)."""
    import faiss
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBED_MODEL)
    doc_emb = model.encode(docs, batch_size=64, normalize_embeddings=True, show_progress_bar=True)
    q_emb = model.encode([QUERY_PREFIX + q for q in queries], normalize_embeddings=True)
    index = faiss.IndexFlatIP(doc_emb.shape[1])
    index.add(np.asarray(doc_emb, dtype="float32"))
    _, idx = index.search(np.asarray(q_emb, dtype="float32"), depth)
    return [[doc_ids[i] for i in row] for row in idx]


def parse_label(text):
    t = text.upper()
    if "CONTRADICT" in t or "REFUTE" in t:
        return "CONTRADICT"
    if "SUPPORT" in t:
        return "SUPPORT"
    return "NOT_ENOUGH_INFO"


def verify(corpus_path=CLEAN_CORPUS, out=None, n_docs=3):
    import torch
    from sklearn.metrics import accuracy_score, f1_score
    from tqdm import tqdm
    from transformers import pipeline

    corpus, claims, _, _ = load(corpus_path)
    by_id = {d["doc_id"]: d for d in corpus}
    parent = parent_of(corpus)
    paths = output_paths(corpus_path, out)
    with open(paths["rankings"]) as f:
        rankings = json.load(f)

    # float16 on GPU; bfloat16 on CPU (fits in ~3 GB RAM; float32 needs >7 GB)
    if torch.cuda.is_available():
        generator = pipeline("text-generation", model=LLM_MODEL,
                             dtype=torch.float16, device_map="auto")
    else:
        generator = pipeline("text-generation", model=LLM_MODEL, dtype=torch.bfloat16, device="cpu")

    gold, pred, grounded, records = [], [], [], []
    for claim in tqdm(claims):
        top = rankings[str(claim["id"])][:n_docs]
        context = "\n\n".join(
            f"[{i + 1}] {by_id[d]['title']}. {' '.join(by_id[d]['abstract'])}" for i, d in enumerate(top))
        messages = [
            {"role": "system", "content": "You verify scientific claims against abstracts. "
             "Answer with exactly one word: SUPPORT, CONTRADICT or NOT_ENOUGH_INFO."},
            {"role": "user", "content": f"Abstracts:\n{context}\n\nClaim: {claim['claim']}\n\nAnswer:"},
        ]
        out = generator(messages, max_new_tokens=8, do_sample=False)
        answer = out[0]["generated_text"][-1]["content"]
        g, p = gold_label(claim), parse_label(answer)
        gold.append(g)
        pred.append(p)
        top_parents = {parent[d] for d in top}
        if gold_docs(claim):
            grounded.append(bool(gold_docs(claim) & top_parents))
        record = {"id": claim["id"], "gold": g, "pred": p, "raw": answer, "top_docs": top}
        if top_parents != set(top):
            record["top_parent_docs"] = sorted(top_parents)
        records.append(record)

    results = {
        "llm": LLM_MODEL,
        "n_claims": len(gold),
        "accuracy": round(accuracy_score(gold, pred), 4),
        "macro_f1": round(f1_score(gold, pred, labels=LABELS, average="macro"), 4),
        "gold_doc_in_context_rate": round(float(np.mean(grounded)), 4),
    }
    print(json.dumps(results, indent=2))
    with open(paths["verification"], "w") as f:
        json.dump({"summary": results, "predictions": records}, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", choices=["download", "retrieve", "verify"], required=True)
    parser.add_argument("--corpus", default=CLEAN_CORPUS, help="corpus.jsonl to retrieve from")
    parser.add_argument("--out", help="results subfolder name (default: the corpus folder name)")
    parser.add_argument("--sparse-only", action="store_true",
                        help="retrieve: BM25 only (no model download; verify then has no rankings to use)")
    args = parser.parse_args()
    if args.step == "download":
        download()
    elif args.step == "retrieve":
        retrieve(args.corpus, args.out, sparse_only=args.sparse_only)
    else:
        verify(args.corpus, args.out)
