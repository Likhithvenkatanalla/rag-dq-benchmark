# How Corpus Defects Affect Retrieval-Augmented Claim Verification: A Controlled Study on SciFact

**Likhith Venkata Sai Manikanta Nalla**

Technical report, October 2026.<br>
Code, results and figures: <https://github.com/Likhithvenkatanalla/rag-dq-benchmark><br>
Archived release: <https://doi.org/10.5281/zenodo.23314458>

## Abstract

Retrieval-augmented systems inherit the quality of the collection they retrieve from, but the effect of
specific defects is rarely measured in isolation. We inject four controlled defects into the 5,183-abstract
SciFact corpus (near-duplicate copies, stale copies with flipped findings, broken parsing, and
fixed-size chunking) at 10-50% of the gold-evidence abstracts, and measure their effect on retrieval
(BM25 and a dense retriever) and on claim verification by three verifiers that read the top-3
retrieved records: a zero-shot LLM (Qwen2.5-1.5B-Instruct), an off-the-shelf NLI cross-encoder, and the
same cross-encoder fine-tuned on SciFact train. Every comparison is paired at the claim level, and the
main findings are repeated across two corruption seeds and three training seeds. Duplicate and stale
copies push the gold abstract out of first place (dense recall@1 from 0.744 to as low as 0.494) without
removing it from the top 10; a fine-tuned reranker recovers part of the duplicate loss but little of the
stale loss. Broken parsing barely affects retrieval but consistently costs verification. Single-sentence
chunks make the fine-tuned verifier both less accurate (0.67 to 0.57) and more confident, doubling its
confidently wrong answers to about 20% of claims. Faced with flipped evidence, the fine-tuned verifier
moves towards CONTRADICT while the zero-shot LLM largely ignores it; neither changes overall accuracy
significantly. The fine-tuned verifier is overconfident even on clean data.

## 1. Introduction

Retrieval-augmented generation (RAG) answers a question from documents retrieved out of a collection,
so its answers can be no better than that collection. Real collections are imperfect: documents are
duplicated across sources, older versions sit next to newer ones that contradict them, PDF parsing
truncates or scrambles text, and chunking decisions split documents into pieces of varying size.

This report asks how much each of these defects, on its own, changes retrieval and the final answer.
We use scientific claim verification as the task because it has expert labels for both the answer
(does the evidence support or contradict the claim?) and the evidence itself (which abstracts), so the
effect of a defect can be traced through retrieval to the verdict. Defects are injected under control,
one type and rate at a time, into an otherwise unchanged corpus, and every model, prompt and setting
is held fixed, so any change in the results comes from the corpus.

Contributions:

- A seeded corruption tool for four defect types with a manifest of every change, and a pipeline that
  reproduces each run from a fixed corruption seed.
- Paired, claim-level measurements of retrieval and verification under 12 corrupted corpora, for two
  retrievers and three verifiers, repeated over two corruption seeds and three verifier training seeds.
- A calibration analysis showing which defects make a verifier confidently wrong.

## 2. Setup

### 2.1 Data

SciFact (Wadden et al., 2020) pairs expert-written scientific claims with 5,183 abstracts. A claim is
labelled SUPPORT or CONTRADICT with respect to its evidence abstracts, or has no evidence
(NOT_ENOUGH_INFO). We evaluate on the 300 dev claims (124 SUPPORT, 64 CONTRADICT, 112
NOT_ENOUGH_INFO); 188 of them have gold evidence, among 184 distinct gold abstracts. Training uses
the 809 train claims, which are disjoint from dev.

### 2.2 Pipeline

**Retrieval.** BM25 over whitespace tokens, and dense retrieval with BGE-small-en-v1.5 embeddings and
exact cosine search (FAISS). Recall@k is the share of a claim's gold abstracts found in the top k,
averaged over the 188 claims with evidence.

**Verification.** A verifier reads the claim and the top-3 dense-retrieved records and outputs SUPPORT,
CONTRADICT or NOT_ENOUGH_INFO. Three verifiers:

- *Qwen*: Qwen2.5-1.5B-Instruct, zero-shot, greedy decoding, instructed to answer with one word.
- *NLI zero-shot*: the cross-encoder `cross-encoder/nli-deberta-v3-xsmall`, trained on general natural
  language inference data, used as released.
- *NLI fine-tuned*: the same model fine-tuned for 3 epochs (learning rate 2e-5, batch 16, max length
  384) on 1,123 SciFact train pairs: gold abstracts with their labels, the cited abstracts of claims
  without evidence as NOT_ENOUGH_INFO, and the top non-gold BM25 abstract of each claim with evidence
  as a hard NOT_ENOUGH_INFO negative. Trained three times (seeds 0, 1, 2).

The NLI verifiers score each (claim, record) pair; a claim takes the highest SUPPORT or CONTRADICT
probability over its three records, or NOT_ENOUGH_INFO if neither reaches a threshold. The threshold
is tuned for macro-F1 on 20% of the train claims held out from training, using the same dense top-3
context, never on dev (tuned values: 0.15, 0.30, 0.35 for the three fine-tuned seeds; 0.05 zero-shot).

**Reranking.** A cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`), as released or fine-tuned for
2 epochs on SciFact train (gold abstracts relevant, the top 4 non-gold BM25 abstracts not), reorders
the dense top-10; the seed-0 fine-tuned verifier then labels each claim from the reranked top-3.

### 2.3 Defects

Defects are applied to the gold-evidence abstracts only, so they reach the evaluated claims; claims and
labels are never changed.

| Defect | What it does | Levels |
|---|---|---|
| Duplicate | adds a near-copy of an abstract (one sentence dropped, about 5% of words deleted, one adjacent word pair swapped per sentence) | 10%, 25%, 50% of gold abstracts |
| Stale | adds a copy with directional findings flipped by rule (increased/decreased, higher/lower, "is associated" to "is not associated", ...) | 10%, 25%, 50% |
| Parse | replaces the abstract with a broken parse: cut off at 30-60% of its length, sentences scrambled, or OCR-style character noise | 10%, 25%, 50% |
| Chunk | splits every abstract into records of 1, 3 or 5 sentences | all abstracts |

A copy (duplicate or stale) is a different document: retrieving it does not count as retrieving the
gold abstract. A chunk counts as its parent abstract. About 1 in 5 stale copies had nothing to flip and
are identical to their original (17 of 91 at 50%, corruption seed 0); Section 3.4 analyses the truly
flipped copies separately.

### 2.4 Statistics

All runs share the same 300 dev claims, so comparisons are paired: for each claim we record whether a
verifier's answer went from right to wrong or from wrong to right relative to the clean corpus, and
report both counts with an exact McNemar test. There are several dozen such tests and no correction
for multiple comparisons, so we treat as findings only effects that repeat across corruption seeds
(0 and 1) and, for the fine-tuned verifier, across training seeds (0, 1, 2).

### 2.5 Reproducibility

The clean pipeline was re-run on GitHub Actions CPU runners and reproduced every Week 1 number exactly
(all retrieval scores, accuracy and the full confusion matrix). Every corrupted corpus is regenerated
from its seed and checked against the committed manifest before use. All predictions, per-claim
outputs, manifests and the fine-tuned models are published (Section 6).

## 3. Results

### 3.1 Clean baseline

Dense retrieval outperforms BM25 at every cut-off (recall@1 0.744 against 0.615; recall@10 0.957
against 0.883). Qwen reaches 0.557 accuracy and 0.502 macro-F1 (majority class: 0.413 and 0.195) but
answers SUPPORT for 200 of the 300 claims. The fine-tuned NLI verifier reaches 0.666 mean accuracy
(0.650 / 0.680 / 0.667 for the three seeds; macro-F1 0.631 / 0.655 / 0.647); each seed is
significantly better than Qwen (McNemar p = 0.005, 0.001, 0.001). The untuned NLI model reaches only
0.353: general NLI training does not transfer to scientific claims by itself. The fine-tuned verifier's
advantage is expected, since it saw in-domain training data and Qwen saw none; we report it as context
for the robustness results rather than as a contest.

### 3.2 Retrieval under corruption

| Corpus | BM25 R@1 | BM25 R@10 | Dense R@1 | Dense R@10 | Dense + fine-tuned reranker R@1 |
|---|---|---|---|---|---|
| clean | 0.615 | 0.883 | 0.744 | 0.957 | 0.805 |
| duplicate 10 / 25 / 50% | 0.594 / 0.541 / 0.463 | 0.883 | 0.689 / 0.668 / 0.604 | 0.957 | 0.774 / 0.751 / 0.644 |
| stale 10 / 25 / 50% | 0.583 / 0.530 / 0.466 | 0.883 | 0.691 / 0.622 / 0.551 | 0.957 | 0.752 / 0.657 / 0.571 |
| parse 10 / 25 / 50% | 0.615 / 0.599 / 0.578 | 0.883 / 0.861 / 0.845 | 0.744 / 0.734 / 0.739 | 0.957 / 0.957 / 0.941 | 0.795 / 0.799 / 0.758 |
| chunk 5 / 3 / 1 sentences | 0.583 / 0.590 / 0.520 | 0.851 / 0.827 / 0.773 | 0.749 / 0.742 / 0.762 | 0.960 / 0.963 / 0.963 | 0.803 / 0.818 / 0.784 |

*Table 1. Retrieval recall over the 188 claims with evidence (corruption seed 0). The reranker reorders
the dense top-10, so its recall@10 equals dense.*

Copies displace the gold abstract from first place in proportion to their rate: at 50%, dense
recall@1 falls to 0.604 (duplicates) and 0.551 (stale), and to 0.569 and 0.494 on corruption seed 1,
while recall@10 does not change; a copy usually sits just above its original. The fine-tuned reranker
lifts clean recall@1 to 0.805 and recovers part of the duplicate loss (0.644 at 50%) but little of the
stale loss (0.571): a flipped copy is as relevant to the claim as its original, so a relevance model
cannot tell them apart. Broken parsing removes documents from BM25's reach (recall@10 0.845 at 50%)
but barely affects dense retrieval. Chunking hurts BM25 at every size but not dense retrieval.

![Figure 1](../figures/fig2_retrieval.png)
*Figure 1. Recall@1 by corruption level for BM25, dense retrieval and dense retrieval with the
fine-tuned reranker.*

### 3.3 Verification under corruption

| Corpus | Qwen | NLI zero-shot | NLI fine-tuned, mean (sd) over 3 seeds |
|---|---|---|---|
| clean | 0.557 | 0.353 | 0.666 (0.015) |
| duplicate 10 / 25 / 50% | 0.543 / 0.547 / 0.557 | 0.360 / 0.367 / 0.353 | 0.670 / 0.670 / 0.677 |
| stale 10 / 25 / 50% | 0.553 / 0.550 / 0.560 | 0.353 / 0.357 / 0.347 | 0.672 / 0.661 / 0.656 |
| parse 10 / 25 / 50% | 0.550 / 0.547 / 0.530 | 0.357 / 0.347 / 0.340 | 0.664 / 0.658 / 0.647 |
| chunk 5 / 3 / 1 sentences | 0.543 / 0.573 / 0.583 | 0.390 / 0.353 / 0.333 | 0.667 / 0.631 / 0.573 |

*Table 2. Verification accuracy on the 300 dev claims (corruption seed 0).*

Accuracy changes are small for most defects, so we read them claim by claim (Table 3).

| Corpus | Qwen | NLI fine-tuned, seed 0 / 1 / 2 | NLI fine-tuned seed 0, corruption seed 1 |
|---|---|---|---|
| parse 10% | 2 / 0 | 1/1, 1/1, 1/0 | 5 / 0 |
| parse 25% | 3 / 0 | 2/1, 4/1, 4/1 | 6 / 1 |
| parse 50% | 8 / 0 (p = 0.008) | 10/4, 9/4, 8/2 | 7 / 2 |
| stale 50% | 3 / 4 | 7/6, 10/8, 11/5 | 4 / 4 |
| chunk 1 sentence | 22 / 30 | 58/38 (p = 0.052), 65/32 (p = 0.001), 64/34 (p = 0.003) | - |

*Table 3. Claims that went right to wrong / wrong to right relative to the clean corpus.*

**Broken parsing costs verification.** In no parse setting did a verifier gain more correct answers
than it lost, and at 25-50% it lost clearly more, for Qwen, for all three fine-tuned seeds and on both
corruption seeds. Only Qwen at 50% is individually significant (8 lost, 0 gained), but the direction
never reverses.

**Single-sentence chunks hurt the fine-tuned verifier**, consistently across training seeds (0.573
mean accuracy; two of three seeds significant). It was trained on whole abstracts; 5-sentence chunks
do not hurt it. Qwen is not hurt by chunking.

**Duplicates do not change verification**, despite their effect on retrieval: the gold abstract is
still in the top 3 for nearly as many claims (gold-in-context rate 0.872 clean, 0.867 at 50%), and a
near-copy carries the same evidence.

![Figure 2](../figures/fig1_verification.png)
*Figure 2. Verification accuracy by corruption level. The orange band spans the three fine-tuned
training seeds; the green line is the seed-0 verifier on corruption seed 1 (compare it with its own
clean value, 0.650).*

**Reranking barely changes verification.** Verifying from the fine-tuned reranker's top-3 moves
accuracy by 0 to +2.3 points across the 13 corpora, with no run changing significantly: the gold
abstract is usually already within the dense top-3, so its position inside it matters little to the
verifier.

### 3.4 Conflicting evidence

To isolate genuinely conflicting evidence, we keep only claims whose top-3 context held a stale copy
whose findings were actually flipped.

| Corruption seed | Verifier | Claims with a flipped copy | Answers changed (to CONTRADICT) | Lost / gained |
|---|---|---|---|---|
| 0 | Qwen | 78 | 6 (1) | 1 / 4 |
| 0 | NLI fine-tuned, seeds 0 / 1 / 2 | 78 | 13 (11) / 19 (15) / 16 (12) | 7/6, 10/8, 11/5 |
| 1 | NLI fine-tuned, seed 0 | 76 | 9 (6) | 4 / 4 |

*Table 4. Stale 50%: claims whose context contained a flipped copy, compared with the same claims on
the clean corpus.*

The two verifier types react differently. The fine-tuned verifier changes its answer for 12-24% of
these claims, and most changes move to CONTRADICT, on both corruption seeds; Qwen changes 6 answers and
moves 1 to CONTRADICT, consistent with its strong SUPPORT bias. Claims whose context held only
unflipped copies never changed answer for the NLI verifier. Neither verifier's accuracy changes
significantly: the fine-tuned verifier is misled on some claims and helped on others.

### 3.5 Calibration

We give each answer a confidence (for SUPPORT or CONTRADICT, the highest probability of that label over
the three records; for NOT_ENOUGH_INFO, one minus the highest SUPPORT or CONTRADICT probability) and
compare it with correctness. Qwen gives no probabilities and is excluded.

| Corpus | Accuracy | Mean confidence | ECE | Confidently wrong (>= 0.9) |
|---|---|---|---|---|
| clean | 0.650 / 0.680 / 0.667 | 0.75 / 0.79 / 0.80 | 0.18 / 0.17 / 0.14 | 11% / 8% / 8% |
| stale 50% | 0.647 / 0.673 / 0.647 | 0.76 / 0.80 / 0.81 | 0.20 / 0.19 / 0.17 | 12% / 10% / 10% |
| parse 50% | 0.630 / 0.663 / 0.647 | 0.74 / 0.79 / 0.80 | 0.19 / 0.18 / 0.16 | 11% / 8% / 8% |
| chunk, 3 sentences | 0.650 / 0.630 / 0.613 | 0.77 / 0.82 / 0.83 | 0.23 / 0.21 / 0.23 | 17% / 14% / 13% |
| chunk, 1 sentence | 0.583 / 0.570 / 0.567 | 0.79 / 0.85 / 0.85 | 0.32 / 0.32 / 0.31 | 22% / 19% / 18% |

*Table 5. Calibration of the fine-tuned verifier, training seeds 0 / 1 / 2 (corruption seed 0). ECE:
expected calibration error over 10 equal-width bins.*

The fine-tuned verifier is overconfident even on clean data (mean confidence 0.75-0.80 against
accuracy 0.65-0.68); its most confident answers (0.8-1.0) are right about 70% of the time. Small
chunks make this much worse: with 1-sentence records, confidence rises while accuracy falls, ECE
roughly doubles and about one claim in five is answered wrongly with confidence of at least 0.9. On
corruption seed 0, claims with a flipped copy in context were more often confidently wrong for all
three training seeds (on the same 95 claims, 15%, 8% and 10% clean against 18%, 17% and 17%), but
this did not repeat on corruption seed 1, so we do not count it as a finding.

![Figure 3](../figures/fig3_reliability.png)
*Figure 3. Reliability of the fine-tuned verifier (seed 0): accuracy against confidence in five bins.
Points below the diagonal are overconfident.*

![Figure 4](../figures/fig4_confident_wrong.png)
*Figure 4. Share of claims answered wrongly with confidence of at least 0.9 (bar: mean over three
training seeds; dots: individual seeds).*

## 4. Discussion

**Retrieval metrics and answer quality can disagree.** Duplicates and stale copies cause the largest
retrieval drops at rank 1 but little change in verification, because the verifier reads the top 3 and
the gold abstract rarely leaves it. Broken parsing shows the opposite: retrieval barely notices it, yet
it is the defect that most consistently costs correct answers, because the verifier receives a
truncated or scrambled version of the right abstract. Monitoring retrieval alone would rank these two
defect types in the wrong order.

**What matters about conflicting evidence depends on the reader.** A verifier that largely ignores its
context (here, a small zero-shot LLM with a strong SUPPORT bias) is unaffected by flipped copies; a
verifier that reads its context closely reacts to them. Stale documents are therefore a latent risk
that becomes visible as the reader improves. Relevance-based reranking does not help, since a flipped
copy is as relevant as its original; detecting version conflicts needs signals beyond relevance (dates,
provenance, deduplication before indexing).

**Chunk size interacts with the verifier's training.** The fine-tuned verifier, trained on whole
abstracts, degrades on single sentences and becomes more confident while doing so. Chunking choices
should match what the verifier was trained on, and confidence scores from a verifier applied outside its
training format should not be trusted.

## 5. Limitations

- One dataset (SciFact dev, 300 claims). Effects of a few claims are within noise, and the p-values are
  not corrected for multiple comparisons; findings rest on repetition across seeds.
- Two corruption seeds; Qwen was evaluated on corruption seed 0 only, because it needs CPUs with native
  bfloat16 support to run in reasonable time on the free GitHub Actions runners we used.
- Stale copies are produced by rule-based word substitution; about 1 in 5 had nothing to flip, and
  flipped copies may read less naturally than real outdated versions.
- Defects target gold-evidence abstracts, which makes their effect on evaluated claims visible but
  is not a model of how defects are distributed in real collections.
- Single small models (BGE-small, Qwen2.5-1.5B, DeBERTa-v3-xsmall); larger models may behave
  differently.

## 6. Availability

Code, all per-claim predictions and corruption manifests, and the figures are at
<https://github.com/Likhithvenkatanalla/rag-dq-benchmark> under the MIT licence; release v1.0.0 is archived on
Zenodo as <https://doi.org/10.5281/zenodo.23314458>. The fine-tuned
models are assets of the release `models-week3`. Exact package versions are in `requirements.txt` and
`requirements-lock.txt`; the GitHub Actions workflows in `.github/workflows/` re-run every experiment.
SciFact is not redistributed and is downloaded from its official release.

Code was developed with AI assistance (Claude); all reported numbers come from the runs whose outputs
are in the repository.

## References

- Wadden, D., Lin, S., Lo, K., Wang, L. L., van Zuylen, M., Cohan, A., & Hajishirzi, H. (2020). Fact or
  Fiction: Verifying Scientific Claims. *Proceedings of EMNLP 2020*.
- Xiao, S., Liu, Z., Zhang, P., & Muennighoff, N. (2023). C-Pack: Packaged Resources to Advance General
  Chinese Embedding. arXiv:2309.07597 (BGE embedding models).
- Qwen Team (2024). Qwen2.5 Technical Report. arXiv:2412.15115.
- He, P., Gao, J., & Chen, W. (2023). DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with
  Gradient-Disentangled Embedding Sharing. *ICLR 2023*.
- Robertson, S., & Zaragoza, H. (2009). The Probabilistic Relevance Framework: BM25 and Beyond.
  *Foundations and Trends in Information Retrieval*, 3(4).
- McNemar, Q. (1947). Note on the sampling error of the difference between correlated proportions or
  percentages. *Psychometrika*, 12(2).
