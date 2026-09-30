# RAG Data-Quality Benchmark

**Question:** how do data-quality defects in a document collection (duplicates, broken parsing,
stale or conflicting versions, chunking choices) change the accuracy of retrieval-augmented answers?

**Data:** SciFact (Wadden et al., EMNLP 2020): 5,183 scientific abstracts and expert-written claims
labelled SUPPORT / CONTRADICT (or no evidence, treated as NOT_ENOUGH_INFO). The dev split
(300 claims: 124 SUPPORT, 64 CONTRADICT, 112 NOT_ENOUGH_INFO) is used for evaluation.

## Week 1: clean baseline (complete)

```bash
pip install -r requirements.txt
python week1_baseline.py --step download
python week1_baseline.py --step retrieve   # BM25 vs dense (BGE-small + FAISS), recall@1/3/5/10
python week1_baseline.py --step verify     # open LLM (Qwen2.5-1.5B-Instruct) labels each claim from top-3 dense abstracts
python analyze_week1.py                    # per-class metrics, confusion matrix, majority baseline
```

### Retrieval (188 dev claims with gold evidence documents)

| Retriever | R@1 | R@3 | R@5 | R@10 |
|---|---|---|---|---|
| BM25 (whitespace tokens) | 0.615 | 0.770 | 0.819 | 0.883 |
| Dense: BGE-small-en-v1.5 + FAISS (cosine) | **0.744** | **0.859** | **0.898** | **0.957** |

### Claim verification (all 300 dev claims, zero-shot, top-3 dense abstracts as context)

| System | Accuracy | Macro-F1 |
|---|---|---|
| Majority class (always SUPPORT) | 0.413 | 0.195 |
| Qwen2.5-1.5B-Instruct, greedy, one-word answer | **0.557** | **0.502** |

Per class (precision / recall / F1): SUPPORT 0.540 / 0.871 / 0.667; CONTRADICT 0.447 / 0.328 / 0.378;
NOT_ENOUGH_INFO 0.717 / 0.339 / 0.461. The model predicts SUPPORT for 200 of 300 claims.

A gold abstract was in the top-3 context for 87.2% of evidence claims (164/188). On those claims
accuracy was 0.701; on the 24 claims without it, 0.583 (small sample; the model often still
answers SUPPORT without the evidence being present).

Raw outputs: `results/week1_retrieval.json`, `results/dense_rankings.json`,
`results/week1_verification.json` (every prediction and raw model answer), `results/week1_analysis.json`.

### Environment for the reported numbers

CPU only (2 vCPU Intel Xeon 2.10 GHz, 7 GB RAM, no GPU). Python 3.11, torch 2.14.0+cpu,
transformers 5.17.0, sentence-transformers 6.1.0, faiss-cpu 1.15.1, scikit-learn 1.8.0.
The LLM runs in bfloat16 on CPU (float16 on GPU). Decoding is greedy, so results are deterministic
up to numerical differences across hardware. Runtime: dense encoding about 11 min, verification about 21 min.

## Week 2: corrupted corpora (in progress: BM25 done; dense retrieval and verification pending)

`corrupt_corpus.py` writes a corrupted copy of the corpus to `data/corrupted/<name>/`, with a
`manifest.json` listing every changed document. Claims and gold labels are never changed.

| Defect | What it does | Options |
|---|---|---|
| `duplicate` | adds near-duplicate copies (one sentence dropped, ~5% of words deleted, one word pair swapped per sentence) | `--rate`, `--copies` |
| `parse` | replaces documents in place with a broken parse: truncated mid-text, sentence order scrambled, or OCR-style character noise | `--rate` |
| `stale` | adds a conflicting copy with directional findings flipped by rule (increased/decreased, higher/lower, "is associated" to "is not associated", ...) | `--rate` |
| `chunk` | splits every abstract into chunks of N sentences | `--chunk-size` |

`--target gold` (default) samples only from the 184 documents that are gold evidence for a dev
claim, so the defects reach the evaluated claims; `--target all` samples from the whole corpus.
`--seed` (default 0) makes every corruption reproducible.

```bash
python corrupt_corpus.py --defect stale --rate 0.25      # -> data/corrupted/stale_r0.25_gold_s0/
python week1_baseline.py --step retrieve --corpus data/corrupted/stale_r0.25_gold_s0/corpus.jsonl
python week1_baseline.py --step verify   --corpus data/corrupted/stale_r0.25_gold_s0/corpus.jsonl
python analyze_week1.py --run stale_r0.25_gold_s0        # results in results/stale_r0.25_gold_s0/
```

Models, prompt and decoding are identical to Week 1, so any change comes from the corpus.
Scoring rules: a duplicate or stale copy is a different document, so retrieving it does not
count as retrieving the gold abstract. A chunk counts as its parent document, and recall@k is
computed over the first k distinct parent documents. The LLM context is still the top-3
retrieved records (chunks, for a chunked corpus).

To run the whole grid (`duplicate`, `parse`, `stale` at rates 0.1 / 0.25 / 0.5; `chunk` at 1 / 3 / 5
sentences) and write `results/week2_summary.json`:

```bash
python run_week2.py                 # retrieve + verify + analyze everything (GPU advised)
python run_week2.py --sparse-only   # BM25 retrieval only: no model downloads, ~2 min on CPU
```

### BM25 retrieval under corruption (seed 0; dense retrieval and verification not yet run)

Recall over the 188 dev claims with gold evidence. Defects target gold-evidence documents.

| Corpus | R@1 | R@3 | R@5 | R@10 |
|---|---|---|---|---|
| clean (Week 1) | 0.615 | 0.770 | 0.819 | 0.883 |
| duplicate 10% | 0.594 | 0.770 | 0.817 | 0.883 |
| duplicate 25% | 0.541 | 0.765 | 0.811 | 0.883 |
| duplicate 50% | 0.463 | 0.747 | 0.809 | 0.883 |
| stale 10% | 0.583 | 0.770 | 0.817 | 0.883 |
| stale 25% | 0.530 | 0.754 | 0.812 | 0.883 |
| stale 50% | 0.466 | 0.753 | 0.804 | 0.883 |
| parse 10% | 0.615 | 0.770 | 0.819 | 0.883 |
| parse 25% | 0.599 | 0.754 | 0.792 | 0.861 |
| parse 50% | 0.578 | 0.711 | 0.768 | 0.845 |
| chunk, 1 sentence | 0.520 | 0.668 | 0.704 | 0.773 |
| chunk, 3 sentences | 0.590 | 0.728 | 0.785 | 0.827 |
| chunk, 5 sentences | 0.583 | 0.754 | 0.801 | 0.851 |

Early reading (BM25 only):
- Duplicates and stale copies mostly hurt R@1: a copy often outranks the gold abstract
  (R@1 drops from 0.615 to about 0.46 at 50%), but pushes it down only one place, so R@10 does not move.
  For verification this matters most for `stale`, since the top-ranked copy says the opposite.
- Broken parsing loses documents outright (R@10 falls to 0.845 at 50%). At 10% no score changed:
  4 of the 18 affected documents were only scrambled, which BM25 cannot see (it ignores word order),
  and the other 14 did not cross a rank cutoff.
- Chunking hurts at every size tested, and smaller chunks hurt more (R@10 0.773 for single sentences).

Per-run files: `results/<run>/retrieval.json`; all runs: `results/week2_summary.json`.

Known limitation: the `stale` flips are rule-based. At rate 0.5 with seed 0, 17 of 91 stale
copies had no matching words and are identical to the original (reported as
`stale_copies_without_edits` in the manifest).

## Roadmap

- Week 2: inject controlled corpus defects (near-duplicates, truncated/garbled parsing, stale
  conflicting versions, chunk sizes) and measure the drop in retrieval and verification.
- Week 3: fine-tune a reranker/verifier in PyTorch (Hugging Face) on SciFact train; measure recovery.
- Week 4: calibration analysis (ECE, Brier), ablations, figures.
- Week 5: technical report and release (Zenodo DOI).

## Notes

Code was developed with AI assistance (Claude); all reported numbers come from the runs whose
outputs are in `results/`.

SciFact data are downloaded from the official release and are not redistributed here; see the
SciFact repository for its licence.
