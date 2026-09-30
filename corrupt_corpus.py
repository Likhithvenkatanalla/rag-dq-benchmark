"""Week 2: inject controlled data-quality defects into the SciFact corpus.

Reads the clean corpus and writes a corrupted copy plus a manifest of every change,
so each later experiment can be traced back to exactly which documents were affected.
Claims and gold labels are never modified: the question is how the same claims fare
against a dirtier collection.

Defects:
    duplicate  add near-duplicate copies (new doc_ids) of selected documents
    parse      replace selected documents in place with a broken parse
               (truncated, sentence order scrambled, or OCR-style character noise)
    stale      add a conflicting "stale version" copy (new doc_id) of selected documents,
               with directional findings flipped by rule-based word substitution
    chunk      split every abstract into chunks of --chunk-size sentences (no --rate)

Copies from `duplicate` and `stale` are different documents: retrieving one does not count
as retrieving the gold abstract (`derived_from` records where they came from). Chunks are
pieces of their document: `parent_doc_id` maps them back for recall.

--target gold (default) samples only from documents that are gold evidence for a dev claim,
so the defects actually reach the claims being evaluated; --target all samples from the
whole corpus.

Usage:
    python corrupt_corpus.py --defect duplicate --rate 0.25
    python corrupt_corpus.py --defect chunk --chunk-size 3
Output: data/corrupted/<name>/corpus.jsonl and manifest.json, where <name> defaults to
e.g. duplicate_r0.25_gold_s0. Pass it on with
    python week1_baseline.py --step retrieve --corpus data/corrupted/<name>/corpus.jsonl
"""
import argparse
import json
import os
import random
import re

DATA_DIR = "data"
CLEAN_CORPUS = os.path.join(DATA_DIR, "data", "corpus.jsonl")
DEV_CLAIMS = os.path.join(DATA_DIR, "data", "claims_dev.jsonl")
OUT_DIR = os.path.join(DATA_DIR, "corrupted")

# Directional word pairs for `stale`; each is applied in both directions.
FLIPS = [
    ("increased", "decreased"), ("increases", "decreases"), ("increase", "decrease"),
    ("higher", "lower"), ("more", "less"), ("greater", "smaller"),
    ("improved", "worsened"), ("improves", "worsens"),
    ("enhanced", "reduced"), ("promotes", "inhibits"), ("promoted", "inhibited"),
    ("activates", "suppresses"), ("activated", "suppressed"),
    ("positively", "negatively"), ("positive", "negative"),
    ("upregulated", "downregulated"), ("up-regulated", "down-regulated"),
    ("elevated", "diminished"), ("risk factor", "protective factor"),
]
# Negation insertions for `stale`: "is associated" -> "is not associated", etc.
NEGATE = re.compile(r"\b(is|are|was|were)\s+(significantly\s+)?(associated|correlated|required|necessary|sufficient)\b",
                    re.IGNORECASE)
# OCR-style confusions for `parse`.
OCR = [("m", "rn"), ("rn", "m"), ("l", "1"), ("O", "0"), ("e", "c"), ("fi", "f"), ("cl", "d")]


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def match_case(word, template):
    if template.isupper():
        return word.upper()
    if template[:1].isupper():
        return word[:1].upper() + word[1:]
    return word


def flip_findings(text):
    """Flip directional wording; returns (new_text, number_of_edits)."""
    table = {}
    for a, b in FLIPS:
        table[a], table[b] = b, a
    pattern = re.compile(r"\b(" + "|".join(sorted(map(re.escape, table), key=len, reverse=True)) + r")\b",
                         re.IGNORECASE)
    text, n_flip = pattern.subn(lambda m: match_case(table[m.group(0).lower()], m.group(0)), text)
    text, n_neg = NEGATE.subn(lambda m: f"{m.group(1)} not {m.group(2) or ''}{m.group(3)}", text)
    return text, n_flip + n_neg


def near_duplicate(sentences, rng):
    """Light rewording: drop one sentence (if >2), then delete ~5% of words and swap
    one adjacent word pair per sentence. Close enough to fool exact-match dedup."""
    sents = list(sentences)
    if len(sents) > 2:
        sents.pop(rng.randrange(len(sents)))
    out = []
    for s in sents:
        words = [w for w in s.split() if rng.random() > 0.05] or s.split()
        if len(words) > 3:
            i = rng.randrange(len(words) - 1)
            words[i], words[i + 1] = words[i + 1], words[i]
        out.append(" ".join(words))
    return out


def ocr_noise(text, rng, rate=0.08):
    """Apply an OCR confusion to ~rate of the words that contain a confusable letter pair,
    and drop ~2% of spaces (merging words)."""
    words = text.split(" ")
    for i, w in enumerate(words):
        if rng.random() < rate:
            options = [(a, b) for a, b in OCR if a in w]
            if options:
                a, b = rng.choice(options)
                words[i] = w.replace(a, b, 1)
    out = words[0]
    for w in words[1:]:
        out += ("" if rng.random() < 0.02 else " ") + w
    return out


def broken_parse(doc, rng):
    """Returns (title, abstract_sentences, mode)."""
    mode = rng.choice(["truncate", "scramble", "ocr"])
    sents = list(doc["abstract"])
    if mode == "truncate":
        text = " ".join(sents)
        cut = int(len(text) * rng.uniform(0.3, 0.6))
        return doc["title"], [text[:cut]], mode  # cut mid-word, sentence boundaries lost
    if mode == "scramble":
        rng.shuffle(sents)
        return doc["title"], sents, mode
    return ocr_noise(doc["title"], rng), [ocr_noise(s, rng) for s in sents], mode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--defect", choices=["duplicate", "parse", "stale", "chunk"], required=True)
    parser.add_argument("--rate", type=float, help="fraction of target documents to corrupt")
    parser.add_argument("--target", choices=["gold", "all"], default="gold")
    parser.add_argument("--copies", type=int, default=1, help="copies per document (duplicate only)")
    parser.add_argument("--chunk-size", type=int, help="sentences per chunk (chunk only)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", help="output folder name under data/corrupted/")
    args = parser.parse_args()

    if args.defect == "chunk":
        if not args.chunk_size or args.chunk_size < 1:
            parser.error("--defect chunk needs --chunk-size >= 1")
        name = args.name or f"chunk_{args.chunk_size}"
    else:
        if args.rate is None or not 0 < args.rate <= 1:
            parser.error(f"--defect {args.defect} needs --rate in (0, 1]")
        name = args.name or f"{args.defect}_r{args.rate}_{args.target}_s{args.seed}"

    rng = random.Random(args.seed)
    corpus = read_jsonl(CLEAN_CORPUS)
    next_id = max(d["doc_id"] for d in corpus) + 1
    changes = []

    if args.defect == "chunk":
        out = []
        for d in corpus:
            for start in range(0, max(len(d["abstract"]), 1), args.chunk_size):
                out.append({"doc_id": next_id, "parent_doc_id": d["doc_id"], "title": d["title"],
                            "abstract": d["abstract"][start:start + args.chunk_size]})
                next_id += 1
        changes.append({"n_docs": len(corpus), "n_chunks": len(out)})
    else:
        if args.target == "gold":
            gold = {int(k) for c in read_jsonl(DEV_CLAIMS) for k in c.get("evidence", {})}
            pool = sorted(d["doc_id"] for d in corpus if d["doc_id"] in gold)
        else:
            pool = sorted(d["doc_id"] for d in corpus)
        selected = set(rng.sample(pool, round(args.rate * len(pool))))
        out, extra = [], []
        for d in corpus:
            if d["doc_id"] not in selected:
                out.append(d)
                continue
            if args.defect == "parse":
                title, abstract, mode = broken_parse(d, rng)
                out.append({"doc_id": d["doc_id"], "title": title, "abstract": abstract})
                changes.append({"doc_id": d["doc_id"], "mode": mode})
            elif args.defect == "duplicate":
                out.append(d)
                for _ in range(args.copies):
                    extra.append({"doc_id": next_id, "derived_from": d["doc_id"], "title": d["title"],
                                  "abstract": near_duplicate(d["abstract"], rng)})
                    changes.append({"doc_id": next_id, "derived_from": d["doc_id"]})
                    next_id += 1
            else:  # stale
                out.append(d)
                title, t_edits = flip_findings(d["title"])
                edited = [flip_findings(s) for s in d["abstract"]]
                n_edits = t_edits + sum(n for _, n in edited)
                extra.append({"doc_id": next_id, "derived_from": d["doc_id"], "title": title,
                              "abstract": [s for s, _ in edited]})
                changes.append({"doc_id": next_id, "derived_from": d["doc_id"], "n_edits": n_edits})
                next_id += 1
        # Copies go at the end so clean documents keep their original positions.
        out.extend(extra)

    folder = os.path.join(OUT_DIR, name)
    os.makedirs(folder, exist_ok=True)
    write_jsonl(os.path.join(folder, "corpus.jsonl"), out)
    manifest = {"args": vars(args), "clean_docs": len(corpus), "corrupted_docs": len(out), "changes": changes}
    if args.defect == "stale":
        manifest["stale_copies_without_edits"] = sum(c["n_edits"] == 0 for c in changes)
    with open(os.path.join(folder, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    summary = {k: v for k, v in manifest.items() if k != "changes"}
    summary["n_changes"] = len(changes)
    print(json.dumps(summary, indent=2))
    print("Wrote", folder)


if __name__ == "__main__":
    main()
