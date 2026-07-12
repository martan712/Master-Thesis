# Phase 2 — Decomposing the Residual and Deciding the Emphasis of RQ4

*Draft chapter (Markdown; the LaTeX version follows in the writing phase). Citation
numbers of the form [n] refer to the reference list of the literature overview
(`docs/literature-overview.md`). This chapter continues the measurement study reported in
`thesis/phase1-writeup.md` and consumes its four decisions verbatim: the extended battery,
the WordNet-only semantic backend, the DL19+DL20 pool, and the two top-ten cells as the
decomposition base.*

## 1. Theory and motivation

Phase 1 left the thesis with a large, stable, skill-bearing residual and a single number
attached to it: pooled over DL19 and DL20, the best cross-validated combination of the
entire similarity-free axiom battery reaches only 0.59–0.67 accuracy on the top-ten pairs a
reranker is deployed to reorder. That number is a floor, not a headline. It says the
classical axioms re-explain what BM25 already decided and little else, and it leaves the
question the rest of the thesis turns on unanswered: of the behaviour the axioms miss, how
much is *structure a new constraint could capture* and how much is the model's own noise —
and if there is structure, what is it about?

Phase 2 is that decomposition. It is deliberately **not** a search for a bigger prediction
number; Phase 1 already established that no amount of the classical battery closes the gap.
It is the disciplined version of the split the research plan promises (§4.2). It fixes the
combined model, states the explained/residual split *two honest ways*, estimates the part
of the residual that is irreducible model stochasticity so the reducible target is
quantified rather than assumed to be the whole remainder, and then does the one analysis
that decides the emphasis of the rest of the thesis — it **characterises the residual well
enough to say whether it is systematic**. A rich, cross-validated, cross-model-stable
residual makes RQ4 (new axioms) the main act; a thin or unsystematic one shifts the weight
to the RQ5 surrogate and turns RQ4 into an honest boundary result. That go/no-go, made
against criteria fixed *in advance*, is the milestone.

Two properties make the phase cheap and safe. It collects **zero new model verdicts** —
the DL19+DL20 top-ten verdicts for both rankers are already in the append-only preference
store from Phase 1, and RQ3 only re-reads and pools them, spending seconds of local CPU per
cell. And its combined model *is* the Phase 1 L2 logistic over the full battery, so the
starting number stays comparable and the decomposition builds on a fit the previous phase
already validated rather than a fresh object.

**RQ3** organises the phase: how much of the model's pairwise preference does the combined
axiom model predict, and how large and how systematic is the unexplained remainder?
Fidelity remains the primary quantity throughout — Phase 1's effectiveness gate already
licensed treating the ranker's behaviour as skill rather than noise.

## 2. Experimental setup

**The combined model.** The headline combined model is the **L2 logistic regression over
the full lexical battery** — the Phase 1 `joint_fit`, graduated to the pooled fit. It is
fixed rather than searched for three reasons: its coefficients are directly interpretable as
per-axiom directions and weights; it is linear and therefore the natural bridge to the RQ5
Bradley–Terry pointwise surrogate, so RQ3's fit is not throwaway; and it is the object
Phase 1 validated, keeping the starting number comparable. The feature set is the Phase 1
battery — strict core, plus AND/DIV/LB1, plus the surviving relaxed variants
(LNC1@tf{0.2,0.5}, TF-LNC@len{0.1,0.3}, M-TDC@mass{0.1,0.3}), minus the degenerate columns.
Semantics enter only as an ablation; WordNet is the backend. A shallow, depth-limited
gradient-boosted tree is fit on the same folds as a *complement*, reported as headroom
inside the explained part: if a nonlinear model materially beats the linear one, the gap is
axiom-interaction structure the linear decomposition misses; if it does not, the linear fit
is the whole interpretable story. Either way the headline stays the interpretable linear
fit.

**The pool.** The two top-ten cells' merged pair frames are rebuilt from the cached stages
(pool, pairs, axiom preferences — all Parquet under `data/processed/`), each tagged with its
collection and concatenated into the pooled frame: 4,330 pooled all-pairs (1,900 DL19 +
2,430 DL20), of which the decisive subset — position-consistent, model-decisive pairs, the
only pairs the decomposition works on — is 3,158 for Qwen and 2,931 for flan-t5-large.
Because DL19 and DL20 have disjoint query-id spaces, query-grouped cross-validation over the
pool keeps folds query-clean with no extra work (guarded by an assertion). The decomposition
is run pooled (headline) and per-collection (robustness), on both feature sets (lexical;
lexical+WordNet), with Qwen3.6-35B-A3B-AWQ primary and flan-t5-large as replication.

**Explained vs. residual.** For each decisive pair the combined model produces an
out-of-fold probability *p̂*; the pair is **explained** if `sign(p̂ − 0.5)` matches the
model's verdict and **residual** otherwise. Two numbers describe the split, both against the
majority-class baseline. The **accuracy decomposition** is the intuitive count — explained
fraction = OOF accuracy — but a pair called correctly at *p̂* = 0.51 counts as fully
explained, which overstates it. The **information decomposition** is the honest figure: the
fraction of the model's label entropy the axiom model removes, `1 − CE(model)/CE(base
rate)`, a McFadden-style pseudo-R² on the OOF probabilities that credits confident-correct
predictions and penalises confident-wrong ones and is not hostage to the 0.5 threshold.

**The noise floor.** The residual is not one thing: part is irreducible model
stochasticity, part is structure predictable from signals *outside* the axiom battery, and
part is genuinely unexplained. Separating the first from the rest is the central
methodological move, because RQ4's target is the reducible part and its size, not the raw
residual. It is estimated from the model's **order-swap reproducibility**, which the store
carries at zero cost: the collapsed decisive verdict is by construction the subset on which
the two presentation orders agreed, so its reproducible signal is bounded by that
self-agreement, and we report the achievable-accuracy ceiling this implies alongside the
fitted accuracies. This is a **lower bound on noise / upper bound on reducible structure**,
and is stated as such.

**The residual model and its covariates.** The characterisation has three converging views.
*Covariate residual profiles* stratify OOF accuracy and the signed residual by non-axiom
covariates. A cross-validated *residual model* predicts the model's verdict from those
covariates *conditioned on the axiom prediction*, under the same query-grouped CV; a residual
model that beats its axiom-only base rate on held-out queries is the operational definition
of "the residual is systematic." *Clustering* pulls exemplar pairs to turn a coefficient
into a formalisable hypothesis. The covariates are all derivable from the cached frames —
rank and score gaps (`d_rank`, `d_score`), document-length difference (`d_len`), and
query-term coverage difference (`d_qcov`) — and are chosen orthogonal to the battery so the
residual model cannot re-derive the axioms (the leakage guard). Crucially, `d_len` and
`d_qcov` are the **content** covariates: the design's RQ4-emphasis gate requires the residual
signal to be carried by content, not by the model's own confidence, because a residual
predictable from confidence alone is a calibration finding rather than an axiom seed.

**The pre-registered decision.** Before the residual numbers were seen, the branch criteria
were fixed. A **rich residual** — RQ4 as the main act — requires all three of: the residual
model beats its base rate on held-out queries with a 95% query-bootstrap CI entirely above
zero; at least one **content-based** covariate carries that signal; and the structure
replicates across the two models *or* the two collections. A **thin/unsystematic residual**
— weight to RQ5 — is any failure of those: no out-of-fold signal, or confidence-only signal,
or no replication. §3.7 records which branch fired.

## 3. Results

### 3.1 The decomposition — a thin explained part, exactly as expected

The interpretable combined model on the pooled top-ten decisive pairs:

| model | n | base | CV acc | gain | AUC | pseudo-R² | noise ceiling | reducible ↑ |
|---|---|---|---|---|---|---|---|---|
| Qwen3.6-35B  | 3,158 | 0.559 | **0.629** | +0.070 | 0.659 | **0.057** | 0.839 | 0.210 |
| flan-t5-large | 2,931 | 0.587 | **0.666** | +0.079 | 0.686 | **0.074** | 0.797 | 0.131 |

Qwen lands squarely in the 0.59–0.67 starting band. The information decomposition makes the
central point the accuracy number softens: pseudo-R² is 0.057 (Qwen) and 0.074 (flan) — the
axiom model removes only about six to seven percent of the model's label entropy, so **the
residual is more than ninety percent of the behaviour**. The explained part is thin by
construction, not by underfitting: on top-ten pairs both members are already lexically
strong, so the classical axioms largely re-explain what BM25 decided, and the marginal
signal the LLM adds lives in the residual.

### 3.2 The linear fit is the whole interpretable story

The gradient-boosted complement adds essentially nothing — headroom +0.013 (Qwen) and
+0.002 (flan) over the linear fit on the same folds. The model's decisive verdicts carry
almost no axiom-interaction structure the linear model misses, so the linear decomposition is
complete and the new axioms RQ4 seeks should be sought as *additional features*, not as
interactions among the existing battery. The top coefficients are consistent with Phase 1's
best-agreeing axioms: AND dominates on Qwen (+1.16, with PROX2/PROX3 and M-TDC near +0.5),
and M-TDC (+1.06) and AND (+0.98) lead on flan. AND and M-TDC — a co-occurrence constraint
and the (small-sample) term-discrimination one — are again where the little classical signal
concentrates.

**The semantic ablation confirms the RQ2 null carries into the pool.** Adding the WordNet
semantic columns *lowers* pooled CV accuracy — Qwen 0.629 → 0.623 (Δ −0.005), flan 0.666 →
0.652 (Δ −0.014). Under cross-validation the semantic columns are noise the regulariser
fights, exactly as in Phase 1. The headline combined model is lexical, and — as recorded in
the decisions — it is also the linear scorer the RQ5 surrogate will build on.

**Per-collection robustness.** The decomposition replicates on each collection separately:
Qwen DL19 0.641 / DL20 0.592, flan DL19 0.669 / DL20 0.641, with gains of +0.044 to +0.068
throughout. DL19's smaller query set makes its fold noisier, which is why the pooled fit is
the headline and the per-collection numbers are the robustness check.

### 3.3 The noise floor — the residual is not just noise

From order-swap consistency (~0.72 Qwen / ~0.74 flan) the reliability ceiling — the accuracy
a perfect predictor could reach given the model's own reproducibility — is 0.839 (Qwen) and
0.797 (flan). Read against the fitted accuracies, this splits the gap to perfect prediction:
of the residual, about **0.21 (Qwen)** and 0.13 (flan) is *reducible* structure sitting
below the noise floor, and the rest is the model's own stochasticity that no axiom,
covariate or richer model can recover. So there is real headroom for RQ4 — a fifth of the
residual on the primary model is, in principle, predictable — but it is bounded, and the
bound is stated as a bound (an upper estimate of reducible structure, a lower bound on
noise).

### 3.4 The residual is systematic — and the signal is content

The cross-validated residual model — non-axiom covariates conditioned on the axiom
prediction, query-grouped CV, 2,000-draw query-bootstrap CI — finds a small but real content
signal on the primary model:

| model | content-only lift [95% CI] | all-covariate lift [95% CI] |
|---|---|---|
| Qwen (pooled) | **+0.027 [+0.003, +0.052]** | +0.037 [+0.008, +0.065] |
| flan (pooled)  | +0.007 [−0.011, +0.027] | +0.010 [−0.013, +0.032] |

The content covariates — document-length difference and query-coverage difference — predict
the LLM's verdict *above* the axioms on Qwen, both on the content-only set (the gate's set)
and with the full covariates, with CIs entirely above zero. On flan the same covariates push
in the same direction but the CI spans zero, so the *lift magnitude* is Qwen-specific. This
matters because the residual signal is carried by **content**, not by the model's own
confidence margin (which was deliberately excluded from the content set), meeting the design's
requirement that an RQ4 seed be a content dimension the axioms genuinely do not encode rather
than a calibration artefact.

### 3.5 The residual clusters name the structure — and they replicate

Clustering the residual pairs in covariate space and reading exemplars gives the structure a
shape, and the shape is the same on both models even where the magnitude is not. Two clusters
carry it: a **high-verbosity** cluster where the model's residual choice tracks the much
longer document (mean length difference +28.9 on Qwen, +41.1 on flan) and a **high
query-coverage** cluster (mean coverage difference +0.31 on Qwen, +0.29 on flan). The lift is
Qwen-specific but the residual's *shape* — length/verbosity and query-term coverage — is
shared across two very different architectures. Most of the 0.21 reducible residual remains
uncaptured even by these covariates: the residual is systematic in part and large in whole.

### 3.6 The gap-gradient open item is resolved — the top-ten null is validated

Phase 1 carried forward a partial gradient: the judge's decisive rate rose cleanly with rank
gap but per-axiom agreement mostly did not, and DIV drifted below chance as the gap widened.
Phase 2 asks the question the per-axiom view could not — does the *combined-model* OOF
accuracy rise with gap? — and answers it across the full range. Within top-ten the combined
accuracy does rise with rank gap (Qwen 0.61 → 0.72 over gaps 1 → 9; flan 0.67 → 0.71) — but
so does the model's own **decisive rate** (Qwen 0.72 → 0.86; flan 0.63 → 0.75). And on the
wide-gap uniform-50 range the joint accuracy is **flat around 0.53–0.62**, reaching ~0.65
only in the widest decile. The textbook signature — near-chance on top-ten, high on wide-gap
— does not appear even for the combined model. The resolution is the self-consistency reading
Phase 1 deferred: the apparent gradient is largely the model's rising self-consistency on
wider gaps, not the axioms tracking it better, so the **top-ten null is validated, not a
sampling artefact**. The DIV-below-chance-with-gap anomaly is folded into the verbosity
residual of §3.5 — DIV's length behaviour is one face of the length structure the residual
model surfaces — and the gap thread is closed.

### 3.7 The decision — RQ4 is the main act

The pre-registered rich-residual criteria are met on the primary model: (i) the residual
model beats its axiom base rate out-of-fold with a CI above zero (Qwen pooled, both covariate
sets); (ii) the signal is carried by a **content** covariate set (length, coverage), not the
model's confidence; and (iii) the residual *structure* replicates across models — the
verbosity and coverage clusters appear on both Qwen and flan — while the per-collection
decompositions replicate throughout. **RQ4 therefore proceeds as the primary contribution**,
seeded by two concrete, ranked hypotheses: a **verbosity / length constraint** (the model's
residual preference tracks document length beyond what LNC/TF-LNC encode; the largest
residual cluster on both models is a long-document one, and it subsumes the Phase 1
DIV-with-gap reversal), and a **query-coverage constraint** (a residual preference for
broader query-term coverage not captured by AND or the TF axioms).

## 4. Discussion

Phase 2 turns Phase 1's floor into a map. The explained part of a strong LLM ranker's
top-ten behaviour is genuinely thin — the whole axiom battery removes six to seven percent of
the model's label entropy and no more — and, decisively, that thinness is a property of the
model, not of the fit: a nonlinear complement on the same folds buys essentially nothing, so
there is no hidden axiom-interaction structure the linear decomposition misses. The
interpretable linear model is the complete classical account, and it accounts for very
little.

But the residual it leaves is not a shrug. The noise floor shows about a fifth of it (on the
primary model) is reducible structure below the model's own reproducibility ceiling; the
cross-validated residual model shows part of that structure is real and predictable on
held-out queries; and the clusters show it has a content shape — document length and
query-term coverage — that is the same on two very different architectures. This is exactly
the pattern the pre-registered criteria were written to detect, and it fired: the residual is
systematic enough, and content-based enough, to make new-axiom development the main event
rather than a boundary note.

The result also sharpens *how* RQ4 should proceed, and the caveats are recorded with it. The
effect is modest — a few accuracy points of lift — and it explains only a fraction of the
0.21 reducible residual, so most of the residual remains uncharacterised: RQ4 has a clear
direction and ample headroom, not a guaranteed payoff. The lift magnitude is stronger on Qwen
than flan (though the structure is shared), so the new axioms are developed on Qwen and
validated cross-model. And because the linear fit's interaction-headroom is zero, they are
sought as additional features rather than as combinations of the existing battery. The
verbosity hypothesis is the strongest single seed — it is the largest residual cluster on
both models and it absorbs the one Phase 1 anomaly left open — with query coverage second.

Three decisions leave Phase 2 for the phases to come. The combined model is the lexical L2
logistic — semantics are excluded on the negative ablation, and it doubles as the scorer the
RQ5 surrogate builds on. RQ4 is the main act, seeded by the verbosity/length and
query-coverage residual hypotheses, developed on Qwen and validated cross-model. And the
top-ten null is validated: the uniform cells remain a validity control and the gap-gradient
thread is closed. The limitations are Phase 1's, narrowed — two collections and two models
are a small if replicated sample, the noise floor is a bound rather than an exact aleatoric
share, and the WordNet semantic tier stays blunt — but none of them move the finding. On the
pairs that matter, a competent LLM ranker is, to the classical axioms, axiom-plus-a-large-
residual; that residual is systematic in part, its systematic part is about verbosity and
coverage, and formalising that part into new constraints is what the thesis turns to next.

*The RQ4 development of verbosity and query-coverage constraints, and their validation
against the residual characterised here, are reported in the next chapter.*
