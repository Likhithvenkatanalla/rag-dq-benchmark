# Submitting the paper

Two files, both built from `main.tex`:

| File | Use it for |
|---|---|
| `Nalla_2026_corpus_defects_RAG.pdf` (and `arxiv.zip`) | arXiv, CV, sharing: shows your name and links |
| `anonymous/main_anon.pdf` | workshop / journal review: no name, no links, line numbers (rebuild with `./make_anonymous.sh`) |

## 1. arXiv

1. Create an account at https://arxiv.org/user/register (use likhithvsm@gmail.com; affiliation: "Independent researcher").
2. Start a new submission. Choose licence **CC BY 4.0** and primary category **cs.CL** (Computation and Language).
3. arXiv will say you need an **endorsement** for cs.CL and show a 6-character **endorsement code**. Send that code to one or two people using the email below. Once one of them endorses you (it takes them 1 minute), continue.
4. Upload `arxiv.zip` (it compiles as is; it includes `main.bbl`).
5. Fill in the form with the metadata below, check the preview PDF, submit. It appears on arXiv the next working day.

**Metadata**
- Title: Copies, Conflicts and Broken Parses: How Corpus Defects Affect Retrieval-Augmented Claim Verification
- Authors: Likhith Venkata Sai Manikanta Nalla
- Primary category: cs.CL; cross-lists: cs.IR, cs.LG
- Comments: 8 pages, 3 figures, 4 tables. Code, data and models: https://github.com/Likhithvenkatanalla/rag-dq-benchmark
- Abstract:

A retrieval-augmented system can only be as reliable as the collection it searches, yet the effect of individual collection defects is seldom measured on its own. We inject four common defects into the SciFact corpus, namely near-duplicate copies, stale copies with reversed findings, broken parsing and fixed-size chunking, at rates of 10 to 50% of the gold evidence. We then track, claim by claim, how retrieval and the verdicts of four verifiers change. The verifiers are two zero-shot LLMs (Qwen2.5 1.5B and 3B) and an NLI cross-encoder, used either off the shelf or fine-tuned. Every comparison is repeated over five corruption seeds and three training seeds, and the main experiments are replicated on Climate-FEVER. Copies push the gold document out of first place (dense recall@1 falls from 0.74 to 0.53 when half of the gold abstracts have a stale copy), but they hardly change the final verdicts. Broken parsing behaves the other way round. Retrieval barely registers it, yet it costs correct answers for both LLMs and for the fine-tuned verifier under every seed. Stale copies pull the fine-tuned verifier towards CONTRADICT; on the one-sentence records of Climate-FEVER they cost it up to 55 correct answers for 14 gained and make it more often confidently wrong, while both LLMs largely ignore them. Single-sentence chunks make the fine-tuned verifier both less accurate and more confident. A team that monitors retrieval quality alone would therefore rank these defects in the wrong order of harm.

**Who to ask for an endorsement.** Anyone who has posted at least a few cs.CL papers on arXiv in the last
five years can endorse. Good candidates are authors of papers you cite (e.g. the SciFact or
Climate-FEVER authors, or authors of RAG robustness papers), or researchers whose recent arXiv papers are
on RAG, fact checking or data quality. Write to 2-3 people, one at a time, a few days apart.

**Email (fill in the [brackets])**

> Subject: arXiv endorsement request (cs.CL): corpus defects in retrieval-augmented claim verification
>
> Dear Dr. [Surname],
>
> I am an independent researcher working on retrieval-augmented generation. I have written a paper,
> "Copies, Conflicts and Broken Parses: How Corpus Defects Affect Retrieval-Augmented Claim Verification",
> which studies how near-duplicates, stale document versions, broken parsing and chunking in a
> collection change retrieval and claim verification on SciFact and Climate-FEVER. [One sentence on why
> them, e.g. "It builds directly on your work on X, which I cite."]
>
> As this is my first arXiv submission, I need an endorsement for cs.CL. The paper is attached, and the
> code, results and an archived release are public (https://github.com/Likhithvenkatanalla/rag-dq-benchmark,
> https://doi.org/10.5281/zenodo.23333457). If you are willing, my endorsement code is [CODE], and the
> endorsement page is https://arxiv.org/auth/endorse. I completely understand if you cannot.
>
> Thank you for your time,
> Likhith Venkata Sai Manikanta Nalla
> likhithvsm@gmail.com

## 2. Peer review (workshops)

No suitable workshop is open on 13 October 2026; the next calls open in late 2026 / early 2027. Use
`anonymous/main_anon.pdf`, reformatted to the workshop's template if it asks for one. Check each call for
page limits and whether a non-archival or "already on arXiv" paper is allowed (most workshops allow it).

| Venue | Fit | Expected deadline (check the call) |
|---|---|---|
| ICLR 2027 workshops (data-centric ML, RAG, trustworthy LLMs) | strong | about 1 February 2027; calls appear Dec 2026 to Jan 2027 at iclr.cc |
| DEEM 2027 @ SIGMOD (Data Management for End-to-End ML) | very strong: data quality for ML pipelines | about early March 2027 (2026 edition: 6 March); deemworkshop.github.io |
| ACL 2027 workshops (fact checking: FEVER; knowledge-intensive NLP; RAG) | strong | about spring 2027 |
| TMLR journal | strong; peer-reviewed journal, rolling, no deadline | any time (openreview.net/group?id=TMLR) |

All of these use OpenReview, CMT or similar: create an OpenReview profile now (it can take a few days
to be approved without an institutional email). After submitting, you can list the paper on your CV
as "under review at [venue]".
