# Phase 1 Design — Measurement, RQ1–RQ2 (scientific design)

> Companion document: the engineering plan — cost table, module architecture, config
> naming, output inventory, relaxation-lever mechanics, test discipline and runbook —
> lives in `phase1-implementation.md`. This document holds the scientific design.

Phase 1 of the research plan (§5, weeks 5–9) is the lexical and semantic agreement
studies. **Milestone: the per-axiom and semantic agreement profiles.** It is built
directly on the Phase 0 outcomes and decisions (`phase0-design.md` §7); everything below
assumes those numbers.

## 1. Framing

Phase 0 changed what "measurement" means here. The classical lexical battery already
*has* a headline number on the primary condition — jointly ~1–3 accuracy points over the
base rate on BM25 top-10 pairs, stable across two very different models — so Phase 1 is
not a search for a bigger agreement number. It is the systematic version of that
measurement: replicate it (second collection), validate it (the gap gradient), give the
axioms their best shot (relaxed preconditions, semantic axioms), and produce the
definitive agreement profiles that RQ3's decomposition will consume. A null result that
survives all four is the foundation the rest of the thesis stands on.

## 2. Objectives and exit criteria

Phase 1 is done when all of the following hold:

1. **Grid collected.** Cached verdicts exist for the full grid — {DL19, DL20} ×
   {top-10 all-pairs, uniform depth-100 control} × {Qwen3.6-35B-A3B-AWQ, flan-t5-large}
   — with order swap throughout.
2. **RQ1 profiles.** Per-axiom coverage and agreement (with query-bootstrap CIs) for the
   extended lexical battery, per cell of the grid.
3. **Gap gradient measured.** Agreement as a function of the BM25 rank gap — the validity
   control from Phase 0 decision 5. The expected shape (near-chance on adjacent-rank
   pairs, rising with gap) validates the pipeline and frames the top-10 result; if it does
   not appear, that is a finding to chase before RQ3, not to skip.
4. **Relaxed preconditions evaluated.** Coverage and agreement of margin-relaxed variants
   of the strict-precondition axioms (TFC1, TFC3, LNC1, TF-LNC, M-TDC) at 2–3 margins
   each, and a decision which variants enter the RQ3 feature set.
5. **RQ2 answered at the WordNet tier.** STMC1/STMC2 (and REG/ANTI-REG, which also need
   term similarity) measured with the WordNet backend on the full grid; the
   lexical-vs-combined delta in agreement *and* in joint predictive power reported; an
   explicit go/no-go on the 7.24 GB fastText download recorded.
6. **Joint fit graduated.** The Phase 0 ad-hoc joint-fit analysis (majority vote,
   query-grouped CV logistic regression) is a reproducible script whose outputs exist for
   every grid cell — the direct input to RQ3.
7. **Effectiveness gate passed** (§4). The Copeland aggregation of Qwen's cached top-10
   all-pairs verdicts (= PRP-allpair) beats the BM25 first-stage baseline on nDCG@10 on
   *both* DL19 and DL20; flan-t5-large is reported as contrast, not gated. This validates
   that the top-10 residual the rest of the thesis studies is skill rather than noise;
   stop-and-fix (prompt/scoring/model) if Qwen fails to clearly beat BM25 before any axiom
   conclusion is drawn.
8. **Decisions recorded in §9**: the battery + margins for RQ3, the semantic similarity
   backend, and which grid cells RQ3 builds its decomposition on.

Primary metric: **fidelity** throughout (we characterise the model, not the qrels), as
fixed in plan §4.1.

## 3. Experimental grid

Two collections, two sampling conditions, two rankers. Everything shares the MS MARCO
passage corpus and the prebuilt Terrier index already cached in Phase 0. (The cost and
latency of the new cells is in `phase1-implementation.md` §2.)

**Collections.**
- `msmarco-passage/trec-dl-2019/judged` — 43 queries; top-10 verdicts already in the
  store from Phase 0 (3,800 presentations/model, free to re-analyse).
- `msmarco-passage/trec-dl-2020/judged` — 54 queries; the replication collection (Phase 0
  decision 2). Same corpus, same index, so pooling is the only new step.

**Conditions.**
- `top10` — top-10 all-pairs (45 pairs/query), the *primary* condition: where reranking
  decisions are made and where Phase 0 found the battery to fail.
- `uniform50` — 50 uniformly sampled pairs/query from the depth-100 pool, the *validity
  control*: wide-gap pairs where classical axioms should visibly work. Uniform sampling
  over C(100,2) pairs gives a broad spread of rank gaps for the gradient figure; 43–54
  queries × 50 pairs ≈ 2.2–2.7k pairs per cell is ~200+ pairs per gap decile.

**Rankers** (both passed the Phase 0 sanity gate; the store keys on model +
prompt_version, so nothing is recollected):
- Primary: `models/qwen3.6-35B-A3B-AWQ`, openai backend at `localhost:9086`, prompt v1,
  thinking disabled, logprob-scored.
- Contrast: `google/flan-t5-large`, hf backend, prompt v0, CPU.

Order swap stays mandatory everywhere (Phase 0 finding: position inconsistency is the
dominant noise source, and its direction is model-specific). k=20 pools remain a fallback
if the gradient analysis wants more mid-gap pairs, not a default.

## 4. Effectiveness gate — do the pairwise verdicts make a ranker?

Phase 1's headline finding is a *null*: the classical battery explains almost nothing of
the LLM's top-10 pairwise decisions. Every later phase treats that large residual as the
interesting object — the part of the ranker's behaviour that the axioms miss. That reading
only holds if the residual is *skill*: structure that produces a genuinely better ranking.
If our Qwen setup cannot turn its verdicts into a ranking that beats BM25, the residual is
noise, and the downstream decomposition, new-axiom and surrogate work would be chasing
model error rather than model competence. The motivation in the research plan asserts that
literature rerankers are strong; this gate verifies that the assertion holds *for our
setup* before the thesis leans on it.

**Method.** For each grid cell of the top-10 all-pairs condition, we aggregate a ranker's
cached pairwise verdicts into a ranking by *Copeland scoring*: a document's score is
(wins − losses) over the pairs it appears in, position-inconsistent pairs already having
collapsed to ties that contribute nothing (§5.3 definitions). Copeland over a complete
top-k tournament is exactly **PRP-allpair** from the pairwise-reranking literature (Qin et
al.), so our numbers are directly comparable to published ones. Documents below the
reranked top-10 keep their BM25 order beneath the reranked block, and equal Copeland
scores are broken by BM25 rank, so the run is a deterministic reordering of the first-stage
pool. We score both the BM25 baseline and the reranked run against the TREC qrels with
`ir_measures`; **nDCG@10 is primary, MAP secondary**, reported as an honest paired
per-query comparison (mean deltas and win/tie/loss counts), not two disembodied means.

**Anchors and decision rule.** Literature puts BM25 at ≈ 0.50 nDCG@10 on DL19/DL20 and a
competent PRP-allpair reranker at ≈ 0.65–0.70. The gate: **Qwen must clearly beat BM25 on
nDCG@10 on both collections.** If it does not, we stop and fix the prompt, the logprob
scoring or the model *before* drawing any axiom conclusion — a residual we cannot show to
be skill is not a foundation. flan-t5-large is reported as contrast but not gated (it is
the weaker Phase 0 contrast model, and the thesis's claims rest on Qwen).

The check costs **zero new model calls**: the verdicts are keyed by (dataset, model,
prompt_version) in the preference store and are already collected by the RQ1 top-10 runs;
the gate only re-reads them. It also doubles as the anchor the research plan promised when
it fixed *fidelity* as the primary metric for characterisation (plan §4.1): fidelity is
safe to privilege precisely because this effectiveness gate independently establishes that
the object being characterised is a competent ranker. The engineering — `ranking.py`, the
`ranking_effectiveness` experiment, the `eff_*` configs — is in `phase1-implementation.md`
§3.

## 5. RQ1 — the lexical battery, extended

### 5.1 Battery composition

Three tiers, all computed on every grid cell (axiom computation is local CPU and cheap
relative to verdict collection):

1. **Strict core** — the Phase 0 ten, at ir_axioms defaults, for comparability:
   `TFC1, TFC3, M-TDC, LNC1, TF-LNC, PROX1–PROX5`.
2. **Additions** — the remaining similarity-free axioms from the plan §3 battery that
   Phase 0 never ran: `AND, DIV, LB1`. (REG/ANTI-REG need a term-similarity backend and
   therefore live in RQ2.)
3. **Relaxed variants** — §5.2.

### 5.2 Relaxed preconditions

Phase 0 found the strict-precondition axioms nearly dead on natural pairs (TFC3 0.1%,
M-TDC 0.8%, TF-LNC 4.9%, LNC1 6.3% coverage) while M-TDC — where it fired at all — was the
*best*-agreeing axiom (0.83–0.89). Coverage, not correctness, is what limits the axioms
that show signal; hence margin-parameterised relaxations.

The scientific content of each lever — which precondition is relaxed, and why — is the
following. TFC1 and TFC3 both require the two document lengths to be approximately equal
(a `LEN` precondition, relative margin 0.1 by default); we relax that margin. LNC1
requires equal term frequency per query term (relative margin 0.1); we relax the same
margin. TF-LNC requires the non-query length to be *exactly* equal, which we replace with
an approximate-equality tolerance. M-TDC requires that *exactly one* query term differs in
term frequency; we drop that single-difference gate while keeping the per-term-pair
validity logic. (The ir_axioms class/subclass mechanics that implement these levers are in
`phase1-implementation.md` §4.)

**RelaxedMTdc is a different, inspired axiom, not M-TDC with a tolerance.** Dropping the
single-difference gate makes the variant fire on pairs the original M-TDC excludes on
purpose; it is therefore no longer M-TDC but a variant inspired by it. Its agreement and
coverage numbers are *ours* and are **not comparable to literature M-TDC**, and the thesis
must present them that way. This is a design fact, stated here where the lever is defined,
not only a downstream risk (see §8).

Margins per relaxed variant: **{0.2, 0.5}** for the `LEN`/TF tolerances (0.1 is the strict
default and already in tier 1) and **{0.1, 0.3}** for the two custom subclasses (whose
strict value is 0). Strict and relaxed variants coexist in one agreement table. The
deliverable is a coverage-vs-margin and agreement-vs-margin table per axiom: the
interesting outcome is whether M-TDC's high agreement survives the coverage gain or was a
small-sample artefact of its 15-pair niche.

### 5.3 Analyses and replication targets

Per grid cell × model, three analyses. Each is tagged **[core]** (a headline result that
will appear in the thesis) or **[exploratory]** (a diagnostic that supports the core
results but is not itself a headline).

1. **Agreement profile** — the Phase 0 table (coverage, n_evaluable, agreement) extended
   with 95% query-bootstrap CIs (resample queries with replacement, 2,000 draws): DL19's
   43 queries make per-axiom agreement noisy, and the profiles are the thesis's headline
   tables, so they need honest error bars. **[core]** — the headline agreement profiles
   with CIs. Per-bin *consistency decomposition* and confidence-stratified views of the
   same profiles are **[exploratory]**.
2. **Gap gradient** — the candidate signature figure for RQ1. For each pair, the BM25 rank
   gap |rank₁ − rank₂| from the pool; agreement per axiom and per-pair joint predictive
   accuracy, binned by gap (deciles on the uniform condition; gaps 1–9 on top10).
   **[core]** — the agreement-vs-gap gradient itself. Also per-bin position consistency and
   decisiveness: if the model itself gets *more* consistent on wide-gap pairs, that
   composes part of the gradient and must be reported alongside, not silently folded in —
   this per-bin consistency/decisiveness decomposition is **[exploratory]**.
3. **Joint fit** — graduated from Phase 0: axiom-majority vote vs. majority-class base
   rate, and an L2 logistic regression over all axiom columns with query-grouped CV
   (GroupKFold, 5 folds), reporting accuracy, ROC-AUC and coefficients. Run once on the
   strict core (Phase 0 comparability), once on the full battery (strict + additions +
   relaxed + later semantic) — the delta is the value the extensions add. Out-of-fold
   predictions are kept per pair so the gap analysis can bin them. **[core]** — the joint
   fit's accuracy/AUC and the full-vs-core delta. Inspection of the individual fitted
   coefficients is **[exploratory]**. As an optional robustness add-on, bootstrap
   confidence intervals over the logistic-regression coefficients (resampling queries)
   would quantify how stable the fitted structure is; this is optional, not a headline.

**Replication targets** (from Phase 0, not predictions that must hold): TFC1 below chance
(~0.47), PROX2 anti-agreeing (~0.33–0.35), PROX3 highest (~0.68–0.71), and the whole
profile replicating across the two models. These are the shapes Phase 1 checks itself
against; a deviation from any of them is a *finding to report*, not a failure of the study.
DL20 either replicates the profile (the axiomatic characterisation generalises across
query sets) or breaks it (the profile is query-set-specific) — both are reportable RQ1
results.

## 6. RQ2 — semantic axioms

### 6.1 Two-tier similarity strategy

The semantic axioms need a term-similarity function; ir_axioms defaults to fastText
(`facebook/fasttext-en-vectors`, a measured 7.24 GB download) but ships a WordNet
synonym-set alternative (NLTK data, tens of MB). Phase 0 deferred the choice; the Phase 1
protocol is explicitly staged:

1. **WordNet tier (default, runs first).** `STMC1, STMC2, REG, ANTI-REG` with the WordNet
   synonym-set term similarity on the full grid. This answers RQ2's headline question — do
   semantic axioms add agreement/predictive power over the lexical battery — at negligible
   download cost.
2. **fastText tier (gated).** Fetch the 7.24 GB model and recompute the same four axioms
   *only if* the WordNet tier shows signal worth sharpening (any semantic axiom with
   coverage ≥ 0.05 and |agreement − 0.5| ≥ 0.05, or a joint-fit delta ≥ 1 accuracy point)
   — a blunt similarity function that already moves the needle justifies a better one; if
   WordNet-STMC is pure noise at decent coverage, a denser similarity is unlikely to
   rescue it and the money result is the null. The gate decision and its numbers are
   recorded in §9 either way. (The third option from plan §7 — the ranker's own hidden
   states — is out of reach for an API-served model and stays out of scope.)

Both similarity tiers coexist in one agreement table; the binding of a similarity backend
to the semantic axioms is an implementation detail (`phase1-implementation.md` §3–4).

### 6.2 Analyses

Same analyses as RQ1 (a wider battery), plus the RQ2-specific comparison: joint fit on
lexical-only vs. lexical+semantic feature sets, same folds, reporting Δaccuracy and ΔAUC
per grid cell. This **lexical-vs-semantic delta is [core]**. A clean null — semantics add
nothing that survives CV — is an acceptable, informative RQ2 answer (plan §4.2); with the
top-10 residual as large as Phase 0 measured, even a one-point gain would be noteworthy.

## 7. Open questions Phase 1 must answer (feeding RQ3/RQ4)

- Does the DL19 agreement profile replicate on DL20? (If yes, RQ3 can pool collections;
  if no, the decomposition must be per-collection.)
- Does the gap gradient confirm the pipeline? (Validity of the top-10 null.)
- Does the effectiveness gate pass — is Qwen's Copeland ranking clearly above BM25 on
  nDCG@10 on both collections (§4)? (Validity of treating the top-10 residual as skill.)
- Does relaxing preconditions buy coverage without destroying agreement — i.e. does M-TDC
  stay at ~0.85 when it fires on 10% of pairs instead of 0.8%?
- Do semantic axioms move joint predictive power at all, and is fastText worth 7 GB?
- How large is the *best* achievable axiom feature set's joint accuracy? (This is RQ3's
  starting number; Phase 0's 0.60 on the strict core is the floor.)

## 8. Scientific risks

- **Relaxed M-TDC changes the axiom's meaning**, not just its tolerance — dropping the
  single-difference gate makes it fire on pairs the original authors excluded on purpose.
  Its agreement number is therefore *ours*, not comparable to literature M-TDC; the design
  doc states this at the point of definition (§5.2) and the thesis must too.
- **WordNet similarity is crude** (synonym-set overlap, zero for out-of-vocabulary terms):
  a weak RQ2 null at the WordNet tier alone would be contestable — which is exactly why the
  fastText gate is specified numerically in §6.1 rather than left to taste.
- **Multiple comparisons.** ~25 axiom columns × 12 grid cells invites cherry-picking; the
  CIs and the pre-registered replication targets in §5.3 are the discipline, and RQ3's
  cross-validated joint fit — not any single axiom's agreement — is the number that carries
  weight downstream.

## 9. Outcomes and decisions (2026-07-12)

Phase 1 is complete: all §2 exit criteria hold. The full grid — {DL19, DL20} × {top10,
uniform50} × {Qwen3.6-35B-A3B-AWQ, flan-t5-large}, order swap throughout — is cached in
the preference store; per-cell outputs (agreement profiles with bootstrap CIs, gap
gradients, joint fits, effectiveness) are under `results/`, with the working overview in
`notebooks/p1_overview.ipynb`. All numbers postdate the 2026-07-11 ir_axioms PROX fixes
(PROX2 batch-path sign flip, PROX1 determinism); Phase 0's "PROX2 anti-agreement"
replication target (~0.33) was an artefact of that bug — post-fix PROX2 sits at
0.57–0.65, the mirror image of the pilot number.

### 9.1 Effectiveness gate (§4): PASS

nDCG@10 of Copeland/PRP-allpair over the cached top-10 verdicts vs. the BM25 pool-as-run,
Δ with 95% paired query-bootstrap CIs (10,000 resamples):

| collection | model | BM25 | reranked | Δ nDCG@10 [95% CI] | W/T/L |
|---|---|---|---|---|---|
| DL19 | Qwen3.6-35B | 0.480 | 0.548 | +0.069 [+0.048, +0.090] | 32/5/6 |
| DL20 | Qwen3.6-35B | 0.494 | 0.556 | +0.062 [+0.039, +0.085] | 41/4/9 |
| DL19 | flan-t5-large | 0.480 | 0.529 | +0.050 [+0.027, +0.073] | 30/5/8 |
| DL20 | flan-t5-large | 0.494 | 0.531 | +0.038 [+0.014, +0.061] | 37/4/13 |

MAP moves the same direction everywhere (Qwen +0.018 / +0.026). **Verdict: pass.** Qwen
clearly beats BM25 on both collections with CIs entirely above zero; the top-10 residual
is treated as skill and Phase 1's conclusions are licensed. Context, not gate: reranked
≈ 0.55 sits below the ≈ 0.65 literature anchor for strong PRP rerankers, consistent with
position consistency of only 0.71 / 0.74 collapsing a quarter of the pairs to ties —
worth a sentence in the thesis, not a stop-and-fix.

### 9.2 Findings against the §7 open questions

- **DL20 replicates the DL19 profile.** Agreement correlation over the 19 axiom columns
  with n ≥ 30: r = 0.79 (Qwen) / 0.86 (flan) across collections, r = 0.93 across models
  on DL20 top-10. Post-fix profile shape: AND (0.78–0.83) and LB1 (0.70–0.77) top;
  proximity 0.55–0.68; TFC1 at chance in every cell; DIV at or below chance despite
  > 90% coverage.
- **Gap gradient: partial.** The judge's decisive rate rises cleanly with rank gap
  (~0.55 → 0.75), but per-axiom agreement mostly does not: TFC1 climbs only in the
  widest deciles, and DIV drifts *below* chance as the gap widens (0.55 → 0.33). The §2
  expectation did not appear as designed — per §2 this is the finding to chase before
  RQ3, and it is carried there as an open item.
- **Relaxed preconditions buy coverage, not agreement.** M-TDC: strict 0.83 (12 pairs,
  DL19) falls to 0.52–0.56 at mass margins 0.1/0.3 (274–855 pairs), and on DL20 even
  strict M-TDC is at chance — the Phase 0 number was a small-sample artefact of its
  niche. LNC1 decays to ≤ chance at tf0.5; TF-LNC's relaxations sit at a weak 0.47–0.59. Two levers are
  degenerate: TFC1@len{0.2,0.5} is bit-for-bit identical to strict TFC1 (the LEN margin
  never binds on these pools), and TFC3 stays dead (≤ 4 evaluable pairs) at every margin.
- **Joint fit.** The strict core adds only +0–2 accuracy points over the majority-class
  base rate; the full battery reaches 0.57–0.64 CV accuracy (AUC 0.63–0.67), +4–9 points
  over base — RQ3's starting number. Phase 0's 0.60 floor holds.
- **RQ2 (WordNet tier): null.** Every semantic axiom's CI spans 0.5 on DL20, and the
  combined-vs-lexical joint fit is *negative* in all four top-10 cells (Δaccuracy −0.7
  to −1.6 points, ΔAUC likewise).

### 9.3 Decisions

1. **Battery + margins for RQ3.** Feature set = the full battery as run: strict core +
   AND/DIV/LB1 + relaxed variants (LNC1@tf{0.2,0.5}, TF-LNC@len{0.1,0.3},
   M-TDC@mass{0.1,0.3}), minus the degenerate columns TFC1@len{0.2,0.5} (identical to
   strict) and TFC3 with its variants (≤ 4 evaluable pairs). No relaxed variant is a
   headline axiom — they enter only as joint-fit features, and RelaxedMTdc is reported
   as an inspired variant, not literature M-TDC (§5.2).
2. **Semantic backend: WordNet only — fastText NO-GO.** The §6.1 per-axiom trigger fires
   nominally twice on DL19 (STMC2@wn 0.555 at coverage 0.077; ANTI-REG@wn 0.556 at
   0.619), but neither replicates on DL20 (0.509 / 0.505), both CIs span 0.5, and the
   joint-fit criterion is negative in all four cells. A blunt similarity that moves
   nothing does not justify the 7.24 GB sharper one; RQ2's answer is a null, reported
   with the WordNet-crudeness caveat (§8) alongside.
3. **Collections pool.** The profile replicates (9.2), so RQ3 builds its decomposition
   on DL19+DL20 pooled, reporting per-collection numbers as robustness checks.
4. **RQ3 grid cells.** The decomposition is built on the two top-10 all-pairs cells (the
   primary condition, validated as skill by the gate), with Qwen primary and
   flan-t5-large as replication; the uniform cells remain validity controls and feed the
   gap-gradient follow-up, not the decomposition.
