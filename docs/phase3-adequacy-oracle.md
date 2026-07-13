# Phase 3 — Answer-Adequacy Oracle Diagnostic

> **Status:** development diagnostic, 2026-07-13. Runs on DL19/DL20 development data only; the
> NFCorpus confirmation set is not touched. This is a **D3 diagnostic** in the cost taxonomy of
> [`phase3-candidate-axiom-specs.md`](phase3-candidate-axiom-specs.md) §2 — an LLM criterion judge
> used to measure a ceiling, never a cheap final axiom.

## 1. Why this exists

The four D0 candidates — DEFANS, NUMANS, COMPARE, CBP — are not four independent hypotheses. They
are four cheap deterministic detectors of **one** latent property:

> **answer-adequacy** — does the passage actually *deliver the answer* to the query, as opposed to
> merely being on the same topic?

Topical overlap is already modelled by the classical axioms (QCOV) and by BM25. Answer-adequacy is
the harder property layered on top: a passage can be maximally on-topic and still not answer the
question (a definitional query met with an acronym-directory listing; a "how many" query met with a
net-worth figure; a comparison query met with a page that names both entities but contrasts
neither). DEFANS/NUMANS/COMPARE carve that property by question type; CBP is the same property's
floor ("is there substantive content at all").

The D0-v2 development evaluation (specs §7) found the fitted four-rule union adds essentially no
fidelity over the classical axioms and is target- and collection-inconsistent. But that result
alone cannot tell us **which of two very different things is wrong**:

- **(a) the property is irrelevant** — the LLM's pairwise choice is not actually driven by
  answer-adequacy, so no operationalisation of it could help; or
- **(b) the property is real but the regexes cannot see it** — answer-adequacy does drive
  preferences, and the cheap D0 rules are simply bad detectors of it.

Validating the four brittle rules one at a time cannot separate (a) from (b), because a rule's
failure gets misread as the concept failing. This diagnostic tests the **concept directly**, at full
coverage, independently of the D0 operationalisation.

## 2. Design

### 2.1 Per-document, absolute rating

The oracle rates **one (query, document) pair at a time** on an absolute 0–3 answer-adequacy scale:

```
0 = does not answer at all (off-topic, or only boilerplate / navigation / metadata)
1 = on the topic but does not contain the answer
2 = partially answers the question
3 = directly and completely answers the question
```

The rating is deliberately **not** "which of A/B is better." A pairwise "better" prompt would simply
re-run the reranking task the target model already performed, so "oracle agrees with target" would be
a near-tautology and prove nothing. Rating each document in isolation decouples the oracle judgment
from the pairwise decision, so that comparing the two is a genuine test. The 0-vs-1 rung is written
explicitly to pull answer-adequacy apart from mere topical relevance, which is the whole point of the
property.

### 2.2 Scoring (matches the project's PRP logprob convention)

Scoring mirrors `rankers/openai_api.py`: one chat completion per document, `temperature 0`,
`max_tokens 1`, first-token `logprobs` with `top_logprobs=20`, reasoning tokens disabled server-side
via `extra_body` (otherwise the first token is a thinking token and scoring degenerates). Rather than
taking the greedy digit, the logprobs of the four labels `0,1,2,3` are softmaxed into a distribution
and the **expected value** `E[k] = Σ_k p(k)·k` is the adequacy scalar. This yields a smooth score in
[0, 3]: a confident "3" and a hesitant "2-vs-3" separate rather than both collapsing to their argmax,
and the per-document distribution (`p0..p3`) is retained for inspection.

### 2.3 Cost

Because scoring is per-document, not per-pair, the whole development set is cheap:

| collection | queries | documents (top-10 scope) |
|---|---:|---:|
| DL19 | 43 | 425 |
| DL20 | 54 | 540 |
| **total** | 97 | **965** |

That is ~9× cheaper than the 4,330-pair pairwise labelling already cached, and it needs **zero new
pairwise calls** — the pairwise labels for all three targets already exist in the preference store, so
the evaluation only correlates the new scalars against them.

### 2.4 Storage and safety

Adequacy scores are cached under `data/adequacy/<model>/part-*.parquet`, a **new** store entirely
separate from the append-only pairwise preference store in `data/preferences/`, which is never read or
written here. The cache is keyed by `(model, prompt_version, collection, qid, docno)` and is
resumable: an interrupted run skips documents that already have a score. The oracle model is the local
Qwen served over the OpenAI-compatible API (`models/qwen3.6-35B-A3B-AWQ`).

## 3. Evaluation

Let `a(q, d)` be the adequacy scalar and, for a pair `(d1, d2)`, `Δ = a(q, d1) − a(q, d2)`. For each
target model's cached pairwise label `y ∈ {−1, 0, +1}`:

1. **Is the property decision-relevant?** On decisive pairs (`y ≠ 0`), measure whether `sign(Δ)`
   agrees with `y`, and the AUC / rank correlation of `Δ` against `y`. High agreement means the LLM's
   preference *is* largely answer-adequacy; low agreement means it is choosing on something else.
2. **The ceiling for the D0 family.** Treat `sign(Δ)` (with a small dead-band) as an oracle "adequacy
   axiom." Its agreement is the ceiling any DEFANS/NUMANS/COMPARE/CBP detector of this property could
   reach. The D0-v2 rules' agreement is then interpreted as *how far below that ceiling the regexes
   sit*, instead of four separate inconclusive tests.
3. **Circularity control.** Qwen is both the oracle and one of the three targets, so
   "Qwen-adequacy predicts Qwen-preference" is an **upper bound** inflated by self-consistency. The
   honest generalisation test is whether **Qwen-adequacy predicts the FLAN-large and FLAN-XL
   preferences** — a property that transfers across model families is genuinely decision-relevant,
   not one model's idiosyncrasy.

## 4. Results (2026-07-13, Qwen oracle, 965 documents)

All 965 documents scored, zero degenerate outputs; adequacy spans the full range (mean 1.58,
sd 1.13, argmax spread across all four labels). `Δadequacy` predicts the pairwise preference on
decisive pairs (dead-band 0.10):

| target | collection | decisive pairs | discriminating cov. | sign agreement | AUC |
|---|---|---:|---:|---:|---:|
| Qwen (self, upper bound) | DL19 | 1356 | 0.83 | 0.883 | 0.931 |
| Qwen (self, upper bound) | DL20 | 1802 | 0.80 | 0.903 | 0.932 |
| FLAN-large (transfer) | DL19 | 1274 | 0.83 | 0.819 | 0.875 |
| FLAN-large (transfer) | DL20 | 1657 | 0.81 | 0.813 | 0.844 |
| FLAN-XL (transfer) | DL19 | 1531 | 0.83 | 0.805 | 0.861 |
| FLAN-XL (transfer) | DL20 | 1976 | 0.79 | 0.829 | 0.858 |

**Relevance anchor.** Mean Qwen adequacy increases monotonically with the human TREC qrel grade
(DL19 0.78 / 1.46 / 2.28 / 2.53 for grades 0–3, Spearman ρ = 0.649; DL20 0.86 / 1.71 / 2.49 / 2.85,
ρ = 0.719), so the scale is tracking genuine relevance, not an artefact of the judge.

**Reading.** The Qwen-on-Qwen rows are an inflated upper bound (self-consistency), but the honest
cross-model rows are decisive: Qwen's answer-adequacy gap predicts the FLAN-large and FLAN-XL
preferences at ≈0.81–0.83 sign agreement and 0.84–0.87 AUC — far above chance and, unlike the D0
axioms, **stable across DL19 and DL20** (no collection interaction). Answer-adequacy is therefore a
strong, model-transferable, collection-stable driver of the LLM preferences.

**Contrast with the D0 axioms (the ceiling gap).** The oracle "adequacy axiom" reaches 0.80–0.90
sign agreement at ≈0.80 coverage. The D0-v2 detectors of the same property reach far less: CBP
agreement was 0.56–0.74 with sharp target/collection inconsistency, and the fitted all-D0 union
added essentially nothing over the classical axioms (specs §7). The gap between the two is the
answer to the diagnostic: **the concept is right and the cheap rules are bad detectors of it**
(outcome (a) below), not the concept being irrelevant.

**Implication.** Effort should move to a better detector of this scalar — D1 dependency/type
features, or a distilled/pinned D2 regressor trained to reproduce the oracle adequacy — rather than
to more hand-written D0 answer-shape rules. The oracle itself is a D3 diagnostic and cannot be the
cheap final axiom (it is an LLM, and using the target model to score is partly circular); it defines
the ceiling and the training signal, not the deployed feature.

## 4b. How to read the outcomes

- **Δadequacy strongly predicts preference (across targets):** the concept is validated. The entire
  D0 failure is operationalisation — the axioms are bad detectors, and effort should go to a better
  one (D1 dependency/type features, or a distilled D2 scorer of this scalar).
- **Δadequacy does not predict preference:** the answer-adequacy family — not just these four rules —
  is the wrong hypothesis, and the LLM is deciding on fluency / length / position / priors. That
  redirects the thesis away from building better adequacy detectors and is itself a real finding.

Either way the diagnostic distinguishes "the axioms are bad" from "the idea is bad," which the
per-candidate D0 evaluation could not. All numbers remain development-only; they do not authorise
access to the locked confirmation set and do not define any frozen battery.

## 5. Does the scalar rerank? (2026-07-13)

The §4 numbers are measured on *pairs*. The downstream question is whether the same scalar, used
directly as a reranker, produces a better ranking than BM25. It does — and it nearly matches the
full pairwise tournament at a fraction of the cost. Sorting each query's top-10 by `a(q, d)`
descending (first-stage rank breaking ties, the unscored tail held below) and scoring the run
against the collection qrels gives a **depth-matched** three-way comparison: the adequacy oracle only
scored the top-10 all-pairs documents, so the adequacy run, the BM25 run, and the Copeland
PRP-allpair run all share the same top-10 block over the same BM25 pool and differ only in how that
block is ordered.

| collection | metric | BM25 | adequacy rerank | PRP-allpair (ceiling) |
|---|---|---:|---:|---:|
| DL19 | nDCG@10 | 0.4795 | **0.5388** | 0.5483 |
| DL19 | AP | 0.2907 | 0.3060 | 0.3083 |
| DL20 | nDCG@10 | 0.4936 | **0.5500** | 0.5555 |
| DL20 | AP | 0.3144 | 0.3382 | 0.3408 |

Paired query-bootstrap (10k, seed 42):

- **vs BM25 — decisively better.** nDCG@10 +0.059 [+0.038, +0.081] W/T/L 31/5/7 on DL19; +0.056
  [+0.033, +0.080] 38/4/12 on DL20. AP is also positive with a CI clear of zero on both.
- **vs PRP-allpair — essentially the ceiling.** adequacy − PRP nDCG@10 is −0.010 [−0.019, −0.001]
  on DL19 (marginally below) and −0.006 [−0.018, +0.006] on DL20 (statistically tied). The
  per-document scalar recovers ~98-99% of the full tournament's nDCG@10.

**Cost.** The adequacy run is **one LLM call per document** — 10 calls per query over the top-10.
PRP-allpair is the complete tournament — C(10,2)=45 pairs × order-swap = 90 calls per query. So the
adequacy reranker reaches the same effectiveness at ~9× fewer LLM calls, because rating each document
once in isolation is strictly cheaper than scoring every ordered pair. This is a reranking result in
its own right, independent of the D0-axiom framing: the validated answer-adequacy scalar is a cheap,
effective pointwise reranker on DL19/DL20, not merely a diagnostic of what drives the pairwise
preferences.

All numbers remain development-only (DL19/DL20); the NFCorpus confirmation set is untouched, and this
defines no frozen battery.

### 5.1 Deeper pools — rescuing what BM25 buried (2026-07-13)

nDCG@10 at depth-10 only ever sees the BM25 top-10, so the reranker cannot rescue a relevant
document BM25 ranked below 10 — and there are many: BM25 buries 200 (DL19) / 221 (DL20) relevant
documents in ranks 10-19 alone. Scoring each query's **top-50** for adequacy (3,819 additional Qwen
calls; scoring is per-document and resumable) and reranking the deeper block lifts nDCG@10 far
above both BM25 and the depth-10 tournament. PRP-allpair stays at depth-10 (extending it would need a
new pairwise tournament per depth), so it is only the depth-10 reference here.

| collection | rerank depth | nDCG@10 | Δ vs BM25 [95% CI] | % of oracle ceiling* |
|---|---|---:|---|---:|
| DL19 | BM25 | 0.4795 | — | — |
| DL19 | PRP-allpair@10 | 0.5483 | +0.069 | — |
| DL19 | adequacy@10 | 0.5388 | +0.059 [+0.038, +0.081] | 94% |
| DL19 | adequacy@20 | 0.6299 | +0.150 [+0.104, +0.198] | 88% |
| DL19 | adequacy@50 | **0.6724** | +0.193 [+0.136, +0.250] | 82% |
| DL20 | BM25 | 0.4936 | — | — |
| DL20 | PRP-allpair@10 | 0.5555 | +0.062 | — |
| DL20 | adequacy@10 | 0.5500 | +0.056 [+0.033, +0.080] | 94% |
| DL20 | adequacy@20 | 0.6188 | +0.125 [+0.083, +0.167] | 87% |
| DL20 | adequacy@50 | 0.6705 | +0.177 [+0.116, +0.238] | 81% |
| DL19 | adequacy@100 | **0.6840** | +0.205 [+0.141, +0.268] | — |
| DL20 | adequacy@100 | **0.6787** | +0.185 [+0.117, +0.257] | — |

\*Oracle ceiling = the top-N reranked perfectly by the true qrel grade (depth-10 0.575/0.583,
depth-20 0.719/0.716, depth-50 0.824/0.831 for DL19/DL20). AP moves the same way (DL19 0.291 →
0.342, DL20 0.314 → 0.379 at depth-50; both CIs clear of zero).

**Reading.** Every step up in depth is a CI-clear gain. At depth 100, adequacy reaches 0.684/0.679
nDCG@10, +0.205/+0.185 over BM25, well past the depth-10 pairwise tournament. Gains attenuate as the
block grows (DL19 +0.043 from 20→50 and +0.012 from 50→100; DL20 +0.052 and +0.008), consistent with
the coarse 0–3 scale accumulating top-band ties, but the absolute result and AP continue to rise
(0.361/0.396 at depth 100). The 10→100 headroom is the answer-adequacy signal rescuing passages BM25
mis-ranked lexically. A finer rating scale, or a smooth `a(q,d)` with a lexical/first-stage tiebreak
inside an adequacy band, remains the obvious route to more resolution.

## 6. Reproduction

```bash
# generate per-document adequacy scores (resumable; ~965 Qwen calls)
uv run --no-sync python experiments/rq4_candidates/adequacy.py \
  --config configs/rq4_candidates_d0.yaml

# evaluate Δadequacy against the cached pairwise labels (§3)
uv run --no-sync python experiments/rq4_candidates/adequacy_eval.py \
  --config configs/rq4_candidates_d0.yaml

# score the complete top-100 development pool for the depth sweep (§5.1; resumable Qwen calls)
uv run --no-sync python experiments/rq4_candidates/adequacy.py \
  --config configs/rq4_adequacy_top100.yaml --depth 100

# rerank by the adequacy scalar; fails closed if the requested score block is incomplete
uv run --no-sync python experiments/rq4_candidates/adequacy_rerank.py \
  --config configs/rq4_adequacy_top100.yaml --depths 10,20,50,100
```
