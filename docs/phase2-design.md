# Phase 2 Design — Decomposition, RQ3 (scientific design)

> Companion document: the engineering plan — pooling mechanics, covariate extraction,
> module architecture, config naming, output inventory, test discipline and runbook —
> lives in `phase2-implementation.md`. This document holds the scientific design.

Phase 2 of the research plan (§5, weeks 10–14) is the decomposition study. **Milestone:
the explained/residual decomposition, and a decision on the emphasis of RQ4.** It builds
directly on the Phase 1 outcomes and decisions (`phase1-design.md` §9); everything below
assumes those numbers, and in particular it consumes the four Phase 1 decisions verbatim
(the battery, the WordNet-only semantic backend, the DL19+DL20 pool, and the two top-10
cells as the decomposition base).

## 1. Framing

RQ3 asks how much of the model's pairwise preferences a combined axiom model can predict,
and how large and how systematic the unexplained remainder is. Phase 1 already fixed the
first half of that answer: the full battery reaches **0.57–0.64 CV accuracy (AUC
0.63–0.67), +4–9 points over the majority-class base rate** per top-10 cell, and RQ2
established that WordNet semantics add nothing that survives cross-validation. Pooled over
DL19+DL20 the interpretable fit sits in the **0.59–0.67** band. That is the starting
number, and it is a *thin* explained part by construction: on top-10 pairs both members
are already lexically strong, so the classical axioms largely re-explain what BM25 decided
and the marginal signal the LLM adds lives in the residual (plan §4.1).

Phase 2 is therefore not a search for a bigger prediction number. It is the disciplined
version of the decomposition the plan promises: fix the combined model, define the
explained/residual split *honestly* (including the part of the residual that is the
model's own noise and can never be explained by anything), and then do the one analysis
that decides the rest of the thesis — **characterise the residual well enough to say
whether it is systematic**. A rich, cross-validated, cross-model-stable residual makes RQ4
(new axioms) the main act, as the plan anticipates; a thin or unsystematic one shifts the
weight to RQ5 and turns RQ4 into an honest boundary result. That go/no-go, made against
criteria fixed *in advance* (§6), is the milestone.

Primary metric: **fidelity** throughout (plan §4.1) — we are characterising the model's
own behaviour, and the effectiveness gate in Phase 1 (§9.1, PASS) already licensed
treating that behaviour as skill rather than noise.

## 2. Objectives and exit criteria

Phase 2 is done when all of the following hold:

1. **Combined model fit.** The interpretable combined axiom model is fit on the pooled
   DL19+DL20 top-10 decisive pairs (Qwen primary, flan-t5-large as replication) with
   query-grouped CV, reported with query-bootstrap CIs, per-collection robustness
   numbers, and a semantic ablation (lexical-only vs. +WordNet) confirming the RQ2 null
   carries into the pooled fit.
2. **Explained/residual decomposition stated two ways.** (a) an **accuracy
   decomposition** — the out-of-fold correct/incorrect split, the intuitive count; and
   (b) an **information decomposition** — the fraction of the model's label entropy the
   axiom model removes (McFadden pseudo-R² / normalised log-loss gain), which is not
   hostage to the 0.5 decision threshold. Both relative to the majority-class baseline.
3. **Noise floor estimated; residual split into stochastic vs. systematic.** An estimate
   of the irreducible (aleatoric) portion of the residual from the model's own
   reproducibility, so the reducible target for RQ4 is quantified rather than assumed to
   be the whole residual.
4. **Residual characterised.** Covariate residual profiles, a cross-validated *residual
   model* over non-axiom covariates, and a qualitative exemplar read of the residual
   pairs — the "what do the mispredicted pairs have in common" the plan asks for (§4.2).
5. **Gap-gradient open item resolved** (§4). The Phase 1 partial gradient (§9.2) is
   carried here per Phase 1 §2; Phase 2 either explains it as a self-consistency artefact
   (top-10 null validated) or surfaces its anomaly (DIV drifting below chance with gap) as
   a piece of the systematic residual.
6. **Cross-model and cross-collection stability** of the decomposition and of any residual
   structure reported.
7. **RQ4-emphasis decision recorded in §7** against the pre-registered systematicity
   criterion of §6.3.

## 3. The decomposition

### 3.1 The combined axiom model

The plan (§4.2) allows "a regularised logistic regression or a shallow tree." We fix the
**L2 logistic regression over the full battery** (Phase 1 `analysis.joint_fit`, graduated)
as the headline combined model, for three reasons: its coefficients are directly
interpretable as per-axiom directions and weights; it is linear and therefore the natural
bridge to the RQ5 Bradley–Terry pointwise surrogate (plan §4.2), so RQ3's fit is not
throwaway; and it is the object Phase 1 already validated and reported, keeping the
starting number comparable.

The feature set is the Phase 1 RQ3 battery (decision 1): strict core + AND/DIV/LB1 +
the surviving relaxed variants (LNC1@tf{0.2,0.5}, TF-LNC@len{0.1,0.3}, M-TDC@mass{0.1,0.3}),
minus the degenerate columns. Semantics enter only as an **ablation** — the headline model
is lexical, and a lexical-vs-(lexical+WordNet) comparison on the pooled fit confirms
whether the RQ2 CV-null (§9.2) survives pooling. RelaxedMTdc remains an inspired variant,
never presented as literature M-TDC (§5.2 of Phase 1).

A **shallow gradient-boosted tree** (depth-limited, same folds) is fit as a *complement,
not a replacement*: if a nonlinear model materially beats the linear one, the gap is
axiom-interaction structure the linear decomposition misses, and it is reported as
**headroom** inside the explained part rather than hidden. If it does not, the linear
model is the whole interpretable story. Either outcome is informative; neither changes the
headline, which stays the interpretable linear fit.

### 3.2 Defining explained vs. residual

Work on the decisive pairs only (position-inconsistent pairs already collapsed to ties and
are out of scope — they are a separate consistency story, not part of the predictable
signal). For each decisive pair the combined model produces an out-of-fold probability
`p̂`; the pair is **explained** if `sign(p̂ − 0.5)` matches the model's verdict, **residual**
otherwise. The **residual set** is the OOF-misclassified decisive pairs — the object of
§3.4.

Two numbers describe the split, both against the majority-class baseline:

- **Accuracy decomposition** — explained fraction = OOF accuracy; residual fraction =
  1 − OOF accuracy. Intuitive, but a pair called correctly at `p̂ = 0.51` counts as fully
  explained, which overstates the explained part.
- **Information decomposition** — the fraction of the model's label entropy removed by the
  axiom model: `1 − CE(axiom model) / CE(base rate)` (McFadden-style pseudo-R² on the OOF
  probabilities). This credits confident-correct predictions and penalises
  confident-wrong ones, and is the honest "how much is explained" figure. Expectation from
  Phase 1: a small positive value (the explained part is thin), which is the point.

### 3.3 The noise floor — what the residual can and cannot contain

The residual is not one thing. It is
- **(a) irreducible / stochastic** — the model's own non-determinism. No axiom, covariate
  or richer model can predict it, and it sets a ceiling on any predictor's accuracy below
  1.0. RQ4 must not chase it.
- **(b) systematic but axiom-invisible** — structure predictable from signals *outside*
  the axiom battery (rank/score gap, document length and verbosity, the model's own
  confidence, exact-match/answer-bearing cues, query type). This is the residual RQ4
  mines.
- **(c) genuinely unexplained** — predictable in principle but captured by none of our
  available signals.

Separating (a) from (b)+(c) is the central methodological move of Phase 2, because RQ4's
target is (b) and its size, not the raw residual. We estimate the noise floor from the
model's **order-swap reproducibility**, which the store already carries at zero extra cost:
position consistency is 0.71/0.74 (§9.1), and the collapsed decisive verdict is by
construction the subset on which the two presentation orders *agreed*, so its reproducible
signal is bounded by that self-agreement. We report the achievable-accuracy ceiling implied
by this reproducibility alongside the fitted accuracies, so the reader sees how much of the
gap-to-1.0 is noise rather than missed structure. This estimate is a **lower bound on
noise / upper bound on reducible structure**, and is stated as such (§8); an optional
temperature re-query on a small pair sample would sharpen it, but is not required and would
cost model calls.

### 3.4 Characterising the residual — the analysis that decides RQ4

Four converging views of the residual set, in increasing commitment:

1. **Covariate residual profiles.** OOF accuracy and the *signed* residual (does the model
   prefer the longer / higher-BM25 / more-confident document among the pairs the axioms get
   wrong?) stratified by non-axiom covariates: BM25 rank gap and score gap, absolute BM25
   scores, both document lengths and their difference/ratio, the model's own
   confidence margin (`prob_a`, `score_a − score_b` from the store), query-term coverage /
   exact-match cues, and query length/type. This is the gap gradient of §4 generalised to
   the full covariate set.
2. **A cross-validated residual model.** Predict, from those non-axiom covariates *and
   conditioned on the axiom prediction*, either "is this pair a residual?" or the model's
   verdict on the residual subset, under the same query-grouped CV. A residual model that
   beats its base rate on **held-out queries** is the operational definition of "the
   residual is systematic." Covariates are chosen orthogonal to the axiom battery so the
   residual model cannot simply re-derive the axioms (leakage guard, §8).
3. **Clustering + qualitative read.** Cluster residual pairs in covariate space and pull
   exemplar pairs — the actual query/passage text — for each cluster. This is what turns a
   statistical residual into a *formalisable* hypothesis for RQ4; a coefficient is not yet
   an axiom.
4. **Cross-model / cross-collection stability.** Any residual structure is re-checked on
   flan-t5-large and per-collection. Structure present in both models and both collections
   is a strong RQ4 seed; Qwen-only structure is model-specific and weaker.

The strongest single residual predictor is likely to be the model's **own confidence
margin** — but confidence is a property of the model, not a retrieval axiom. The design
draws this line explicitly: a residual that is *predictable* from confidence alone is a
calibration finding, not an axiom seed; RQ4 needs residual structure tied to
**document/query content** (length, matching, semantics), which is what views 1–3 isolate.

## 4. The gap-gradient open item (carried from Phase 1 §9.2)

Phase 1 found the gradient **partial**: the judge's decisive rate rises cleanly with rank
gap (~0.55 → 0.75) but per-axiom agreement mostly does not, and DIV drifts *below* chance
as the gap widens (0.55 → 0.33). Phase 1 §2 flagged this to chase before RQ3; Phase 2
resolves it three ways:

1. **Joint-level gradient.** Ask the question the per-axiom view could not: does the
   *combined-model* OOF accuracy rise with gap even where individual axioms are flat?
   `gap.gap_gradient` already carries `joint_cv_accuracy` per bin; the pooled top-10 and
   the wide-gap `uniform50` cells give the full gap range.
2. **Self-consistency decomposition.** Decompose the gradient into a *model
   self-consistency* component (decisive rate / position consistency rising with gap —
   already observed) and an *axiom-alignment* component. If the gradient is mostly the
   former, the pipeline is validated and the top-10 null stands (the wide-gap pairs are
   easier for the *model*, not more axiom-aligned); this is the exploratory decomposition
   Phase 1 deferred, completed here.
3. **The DIV reversal as residual structure.** DIV drifting below chance with gap is a
   concrete anti-signal — DIV may be capturing a verbosity/length preference that reverses
   the axiom's intent on wide-gap pairs. This connects the gap thread directly to §3.4:
   the gradient anomaly is a candidate piece of the systematic residual, and it is followed
   into the residual analysis rather than reported as a loose end.

## 5. Robustness and pooling

Per Phase 1 decision 3 the headline decomposition is on **DL19+DL20 pooled**, with
per-collection fits reported as robustness. Pooling is a concatenation of the two cells'
merged pair frames (verdicts and axiom preferences already cached — no model calls); the
only new engineering is the pooled runner (`phase2-implementation.md` §3). Qwen is primary;
flan-t5-large replicates the whole decomposition, and the **cross-model stability of the
residual structure** (§3.4 view 4) is itself a headline RQ4-emphasis input, not a footnote.

## 6. The RQ4-emphasis decision — criteria fixed in advance

The milestone is a decision, and to keep it honest the criteria are pre-registered here,
before the residual numbers are seen (the discipline the Phase 1 fastText and replication
gates used):

### 6.1 "Rich residual" → RQ4 is the main act

All three hold: (i) the residual model (§3.4 view 2) beats its base rate on held-out
queries by a margin whose 95% query-bootstrap CI is entirely above zero; (ii) at least one
**content-based** covariate (not the model's own confidence alone) carries that signal;
(iii) the structure replicates across the two models *or* the two collections (same sign,
overlapping CIs). Then RQ4 proceeds as the primary contribution, seeded by the
characterised residual clusters (§3.4 view 3).

### 6.2 "Thin/unsystematic residual" → weight shifts to RQ5

The residual model does not beat chance out-of-fold, or its only signal is the model's own
confidence with no content-based structure, or nothing replicates across models. Then RQ4
is reported as an honest boundary result (the model is, to the available signals,
axiom-plus-noise), and the thesis weight moves to the RQ5 surrogate — which the linear
combined model of §3.1 already sets up.

### 6.3 The recorded decision

§7 records which branch fired, the numbers behind it, and — if 6.1 — the ranked list of
residual-cluster hypotheses handed to RQ4.

## 7. Outcomes and decisions (2026-07-12)

Phase 2 is complete: all §2 exit criteria hold. The decomposition runs on the DL19+DL20
top-10 pool, Qwen primary and flan-t5-large replication, from the cached verdicts (zero
model calls); per-cell outputs are under `results/rq3_decomposition/pooled_top10/`, with
the working overview in `notebooks/p2_overview.ipynb`. All numbers postdate the 2026-07-11
ir_axioms PROX fixes (Phase 1 §9).

### 7.1 The decomposition

The interpretable combined model (L2 logistic, full lexical battery) on the pooled top-10
decisive pairs:

| model | n | base | CV acc | gain | AUC | pseudo-R² | noise ceiling | reducible ↑ |
|---|---|---|---|---|---|---|---|---|
| Qwen3.6-35B | 4,330 | 0.559 | **0.629** | +0.070 | 0.659 | **0.057** | 0.839 | 0.210 |
| flan-t5-large | 4,330 | 0.587 | **0.666** | +0.079 | 0.686 | **0.074** | 0.797 | 0.131 |

Qwen lands squarely in the 0.59–0.67 starting band. The **explained part is thin exactly as
the plan expected**: the information decomposition puts pseudo-R² at 0.057 (Qwen) / 0.074
(flan) — the axiom model removes only ~6–7% of the model's label entropy; **the residual is
>93% of the behaviour**. The gradient-boosted complement adds essentially nothing
(headroom +0.013 / +0.002), so the linear fit is the whole interpretable story — the
model's decisive verdicts carry almost no axiom-interaction structure the linear model
misses. Top coefficients are AND and M-TDC (Qwen: AND +1.16; flan: M-TDC +1.06, AND +0.98),
consistent with the Phase 1 finding that AND and the (small-sample) M-TDC were the
best-agreeing axioms.

**Semantic ablation.** Adding the WordNet semantic columns *lowers* pooled CV accuracy
(Qwen 0.629 → 0.623, Δ −0.005; flan 0.666 → 0.652, Δ −0.014) — the RQ2 CV-null (Phase 1
§9.3) carries into the pooled fit. The headline combined model is lexical.

**Per-collection robustness.** The decomposition replicates: Qwen DL19 0.641 / DL20 0.592,
flan DL19 0.669 / DL20 0.641, gains +0.044…+0.068 throughout. The pooled fit is the
headline; the per-collection numbers are the robustness check the plan (§5) asked for.

### 7.2 The residual: noise floor and systematicity

The reliability ceiling (from order-swap consistency ~0.72 / 0.74) is 0.839 (Qwen) / 0.797
(flan): of the gap between the axiom model and perfect prediction, ~0.21 (Qwen) is
*reducible* structure below the noise floor, the rest is the model's own stochasticity. So
the residual is not just noise — there is real headroom for RQ4.

The cross-validated **residual model** (non-axiom covariates conditioned on the axiom
prediction, query-grouped CV, 2,000-draw query-bootstrap CI) finds a small but real
content signal on the primary model:

| model | content-only lift [95% CI] | all-covariate lift [95% CI] |
|---|---|---|
| Qwen (pooled) | **+0.027 [+0.003, +0.052]** | +0.037 [+0.008, +0.065] |
| Qwen DL20 | **+0.039 [+0.004, +0.074]** | +0.056 [+0.019, +0.092] |
| Qwen DL19 | +0.024 [−0.007, +0.057] | +0.013 [−0.026, +0.050] |
| flan (pooled) | +0.007 [−0.011, +0.027] | +0.010 [−0.013, +0.032] |

The content covariates (`d_len`, `d_qcov`) predict the LLM's verdict *above* the axioms on
Qwen — pooled and on DL20, positive in the same direction on DL19. The residual
**clusters** name the structure, and they replicate across both models: a
high-verbosity cluster (Qwen `d_len` +28.9, flan +41.1 — pairs where the model's residual
choice tracks the much longer document) and a high-query-coverage cluster (Qwen `d_qcov`
+0.31, flan +0.29). The lift *magnitude* is Qwen-specific (flan's CI spans zero), but the
*shape* of the residual — length/verbosity and query-term coverage — is the same on both
models. Most of the 0.21 reducible residual remains uncaptured even by these covariates:
the residual is systematic in part and large in whole.

### 7.3 Gap-gradient open item (Phase 1 §9.2) — resolved

At the joint level the gradient is **modest within top-10 and does not extend to wide
gaps**. Within top-10 the combined-model OOF accuracy rises with rank gap (Qwen 0.61→0.72
over gaps 1→9; flan 0.67→0.71) — but so does the model's **decisive rate** (Qwen 0.72→0.86;
flan 0.63→0.75), and on the wide-gap `uniform50` range the joint accuracy is **flat around
0.53–0.62**, reaching ~0.65 only in the widest decile. The expected signature — near-chance
on top-10, high on wide-gap — does **not** appear even for the combined model. The
resolution is Phase 1 §4 interpretation (2): the apparent gradient is largely the model's
own rising self-consistency on wider gaps, not the axioms tracking it better; the top-10
null is **validated**, not a sampling artefact. The Phase 1 DIV-below-chance-with-gap
anomaly is folded into the verbosity residual of §7.2 — DIV's length behaviour is one face
of the length structure the residual model surfaces.

### 7.4 Decision — RQ4 emphasis: **RQ4 is the main act** (§6.1 branch)

The pre-registered §6.1 criteria are met on the primary model: (i) the residual model beats
its base rate out-of-fold with a CI above zero (Qwen pooled, both covariate sets); (ii) the
signal is carried by a **content** covariate set (`d_len`, `d_qcov`), not the model's
confidence; (iii) it replicates **across collections** (positive in DL19 and DL20, same
sign, significant in DL20 and pooled) and the residual *structure* replicates **across
models** (the verbosity and coverage clusters appear in both). RQ4 therefore proceeds as
the primary contribution, seeded by two concrete, ranked residual hypotheses:

1. **A verbosity / length constraint** — the model's residual preference tracks document
   length beyond what LNC/TF-LNC encode; the largest residual cluster on both models is a
   long-document cluster, and it subsumes the Phase 1 DIV-with-gap reversal.
2. **A query-coverage constraint** — a residual preference for broader query-term coverage
   not captured by AND/the TF axioms.

**Caveats carried into RQ4.** The effect is *modest* (a few accuracy points of lift) and
the reducible residual it explains is a fraction of the ~0.21 available, so most of the
residual remains uncharacterised — RQ4 has a clear direction and ample headroom, not a
guaranteed payoff. The lift magnitude is stronger on Qwen than flan (though the structure
is shared), so RQ4 develops its axioms on Qwen and validates cross-model. The linear model's
zero interaction-headroom (§7.1) means the new axioms should be sought as *additional
features*, not as interactions among the existing battery.

### 7.5 Decisions

1. **Combined model = lexical L2 logistic** (semantics excluded — the ablation is
   negative, §7.1); it is also the linear scorer RQ5's Bradley-Terry surrogate builds on.
2. **RQ4 is the main act** (§7.4), seeded by the verbosity/length and query-coverage
   residual hypotheses, developed on Qwen and validated cross-model.
3. **The top-10 null is validated** (§7.3); the uniform cells stay a validity control, and
   the gap-gradient thread is closed.

## 8. Scientific risks

- **Threshold-hacking the residual.** ~10 covariates × several cells invites
  cherry-picking a residual signal. The discipline is the same as Phase 1: query-grouped
  CV (out-of-fold, held-out queries), the pre-registered §6 criterion, query-bootstrap CIs
  on every reported margin, and a qualitative exemplar read that must corroborate any
  statistical cluster before it becomes an RQ4 seed.
- **Leakage — the residual model re-deriving the axioms.** If the residual covariates
  correlate with the axiom columns, the residual model can launder axiom signal back in and
  look systematic spuriously. Mitigation: covariates are chosen orthogonal to the battery,
  the residual model is conditioned on the axiom prediction, and any surviving signal is
  checked to be a *content* dimension the axioms genuinely do not encode.
- **Predictable ≠ axiomatisable.** The model's own confidence margin is likely the
  strongest residual predictor but is not a retrieval axiom; §3.4 keeps confidence-only
  structure out of the RQ4 seed set and requires content-based structure for the 6.1
  branch.
- **The noise floor is an estimate.** Order-swap reproducibility gives a bound, not the
  exact aleatoric share; the split of §3.3 is reported as bound-qualified, and the
  reducible-structure target for RQ4 is stated as an upper estimate.
- **Interpretability vs. accuracy.** A gradient-boosted model beating the linear one means
  the interpretable decomposition is incomplete; that gap is reported as headroom (§3.1),
  not suppressed, and it is itself a (weaker, interaction-shaped) finding.
- **A thin residual is not a failed phase.** Per plan §7 the 6.2 branch is an acceptable,
  informative outcome that redirects rather than blocks the thesis; the design commits to
  reporting it as such.
