# Phase 1 — Measuring the Axiomatic Agreement Profile of an LLM Pairwise Ranker

*Draft chapter (Markdown; the LaTeX version follows in the writing phase). Citation
numbers of the form [n] refer to the reference list of the literature overview
(`docs/literature-overview.md`). This chapter continues the pilot reported in
`thesis/phase0-writeup.md` and assumes its setup and definitions.*

## 1. Theory and motivation

The pilot (Phase 0) left the thesis with a sharp, uncomfortable headline: on the top-ten
pairs that a reranker is actually deployed to reorder, the classical lexical axioms
explain almost nothing of a strong LLM ranker's pairwise decisions — jointly buying only
one to three accuracy points over guessing the majority class, and a naive vote of the
axioms doing worse than that. A near-null of that importance cannot rest on a single
collection of 43 queries, one prompt family, and a battery run at its strict defaults. If
the rest of the thesis is going to treat the unexplained *residual* of the ranker's
behaviour as its main object of study, that null has to be turned from a pilot observation
into a measured, replicated, error-barred result — and the axioms have to be given their
best possible shot at closing it before it is declared.

Phase 1 is that measurement study. It does four things, each a way of attacking the pilot
null rather than restating it. It **replicates** the profile on a second collection (TREC
Deep Learning 2020), so the numbers are not a property of one query set. It **validates**
the pipeline with a rank-gap gradient — the check that low agreement is concentrated on
the hard, lexically close top-ten pairs rather than leaking from a broken index. It gives
the axioms their **best shot**: three similarity-free axioms the pilot never ran, relaxed
preconditions to recover coverage on the strict axioms that were starved of evidence, and
a first tier of semantic axioms. And it establishes, through an independent effectiveness
gate, that the residual it is about to characterise is *skill* rather than noise — that
the ranker whose decisions the axioms fail to explain is a genuinely good ranker. Only
once all four hold is a null result a foundation the rest of the thesis can stand on.

Two research questions organise the study. **RQ1** asks how much of the LLM's pairwise
preference the *lexical* axioms capture, per collection and per model, with honest error
bars; **RQ2** asks whether adding *semantic* axioms — constraints that reason about term
similarity rather than exact term matching — buys anything the lexical battery does not.
The primary quantity throughout remains fidelity, how well the axioms characterise the
model, not agreement with the human relevance judgements.

## 2. Experimental setup

**The grid.** Everything in Phase 1 is measured on the same 2×2×2 grid: two collections
{DL19, DL20} × two sampling conditions {top-10 all-pairs, a uniform depth-100 control} ×
two rankers {Qwen3.6-35B-A3B-AWQ (primary, an API-served mixture-of-experts chat model),
flan-t5-large (contrast, a 0.8-billion-parameter sequence-to-sequence model run on CPU)}.
Both rankers passed the Phase 0 order-swap sanity gate. The DL19 top-ten verdicts already
existed in the append-only preference store from the pilot and were re-analysed for free;
the three new cells per model — DL20 top-ten, and the two uniform cells — cost about 1.6
hours of Qwen calls and 4.6 hours of flan-t5-large CPU time, all resumable and collected
with order swap mandatory throughout. The MS MARCO passage corpus and the prebuilt Terrier
first-stage index were unchanged from the pilot, so DL20 required only a new pooling pass.

The **top-10 all-pairs** condition is the primary one: all 45 unordered pairs among the ten
top-ranked passages per query, the place reranking decisions are made and where the pilot
found the battery to fail. Across 43 DL19 queries this is 1,900 pairs and across 54 DL20
queries 2,430 pairs, each shown in both orders. The **uniform** condition draws 50 pairs
per query uniformly from the depth-100 pool, spreading the pairs broadly across first-stage
rank gaps; it is the validity control, the set of wide-gap pairs on which classical axioms
should visibly work, and it is what the gap gradient is measured on.

**The extended battery.** RQ1 ran three tiers of similarity-free axioms on every cell.
The *strict core* is the pilot's ten — TFC1, TFC3, M-TDC, LNC1, TF-LNC and the five
proximity constraints PROX1–PROX5 — at ir_axioms defaults, for direct comparability. The
*additions* are three similarity-free axioms the pilot never ran: AND, DIV and LB1. The
*relaxed variants* address the pilot's central operational problem — that the
strict-precondition axioms were nearly dead on natural passages (TFC3 fired on 0.1% of
pairs, M-TDC on 0.8%) because their preconditions demand two real passages match almost
exactly on some dimension. Each relaxed variant loosens exactly one precondition by a
margin: the length-equality tolerance on TFC1/TFC3, the per-term frequency-equality
tolerance on LNC1, an approximate-equality replacement for TF-LNC's exact-length gate, and
— for M-TDC — dropping the "exactly one query term differs" gate entirely while keeping the
per-term-pair validity logic. That last lever is worth flagging at the point of use: an
M-TDC that fires when *more* than one term differs is no longer M-TDC but a variant
inspired by it, so its numbers are ours and are not comparable to the literature axiom.
Strict and relaxed variants coexist as aliased columns (`M-TDC@mass0.3`, `TFC1@len0.5`) in
one agreement table.

**The semantic tier.** RQ2 added four axioms that need a term-similarity function — STMC1,
STMC2, REG and ANTI-REG — computed with ir_axioms' WordNet synonym-set backend (tens of
megabytes of NLTK data). This is the deliberately staged answer to the pilot's deferral of
the 7.24-gigabyte fastText similarity model: measure the semantic question first at
negligible cost with a crude similarity, and fetch the expensive one only if the crude tier
shows signal worth sharpening.

**The effectiveness gate.** Before drawing any axiom conclusion, Phase 1 verifies that the
residual it studies is skill. For each top-10 all-pairs cell, the ranker's cached pairwise
verdicts are aggregated into a ranking by Copeland scoring — a document's score is its
wins minus losses over the pairs it appears in, with position-inconsistent pairs already
collapsed to ties that contribute nothing. Copeland over a complete top-k tournament is
exactly PRP-allpair from the pairwise-reranking literature [1], so the numbers are directly
comparable to published ones. The reranked top-ten block sits above the untouched BM25 tail,
Copeland ties are broken by BM25 rank, and both the BM25 pool-as-run baseline and the
reranked run are scored against the TREC qrels with `ir_measures` — nDCG@10 primary, MAP
secondary — as an honest paired per-query comparison. The check costs zero new model calls:
it only re-reads verdicts the RQ1 top-ten runs already collected.

**Definitions** are carried unchanged from the pilot: an unordered pair's verdict is
derived from its two presentations, with a disagreement recorded as position-inconsistent
and treated as a tie; an axiom's *agreement* is measured over pairs where it is non-neutral
and the model decisive, reported alongside its *coverage*; the *position-consistency rate*
is the fraction of pairs with identical verdicts in both orders; the *non-transitivity
rate* is the fraction of cyclic triangles among triangles decisive on all three edges. New
in Phase 1, every agreement figure carries a 95% query-bootstrap confidence interval
(resample queries with replacement, 2,000 draws), because DL19's 43 queries make per-axiom
agreement noisy and these profiles are the thesis's headline tables.

**A correction to the pilot.** All Phase 1 numbers postdate the 2026-07-11 fixes to two
ir_axioms 1.1.2 bugs — a sign flip in PROX2's batch path and a non-deterministic hash seed
in PROX1. The pilot's striking "PROX2 anti-agreement at 0.33" replication target was an
artefact of the sign-flip bug; post-fix, PROX2 sits at 0.58–0.67, the mirror image of the
pilot number. The pilot's PROX2 finding should be read as retracted, and the profile shape
below is the corrected one.

## 3. Results

### 3.1 The effectiveness gate passes — the residual is skill

Copeland aggregation of Qwen's cached top-ten verdicts beats the BM25 first stage clearly
and on both collections, with paired query-bootstrap confidence intervals (10,000
resamples) entirely above zero.

| collection | model | BM25 nDCG@10 | reranked | Δ nDCG@10 [95% CI] | W/T/L |
|---|---|---|---|---|---|
| DL19 | Qwen3.6-35B | 0.480 | 0.548 | +0.069 [+0.048, +0.090] | 32/5/6 |
| DL20 | Qwen3.6-35B | 0.494 | 0.556 | +0.062 [+0.039, +0.085] | 41/4/9 |
| DL19 | flan-t5-large | 0.480 | 0.529 | +0.050 [+0.027, +0.073] | 30/5/8 |
| DL20 | flan-t5-large | 0.494 | 0.531 | +0.038 [+0.014, +0.061] | 37/4/13 |

MAP moves the same direction in every cell (Qwen +0.018 on DL19, +0.026 on DL20). The gate
is passed: Qwen's pairwise verdicts, aggregated, produce a genuinely better ranking than
BM25 on both collections, so the top-ten residual the rest of the thesis studies is a
property of a competent ranker, not model error. One point of context — not a gate failure:
reranked nDCG@10 ≈ 0.55 sits below the ≈ 0.65 literature anchor for strong PRP rerankers,
consistent with position consistency of only 0.71–0.74 collapsing roughly a quarter of the
pairs to ties before aggregation. It is worth a sentence in the thesis, not a stop-and-fix.

### 3.2 Position consistency and transitivity replicate

The pilot's two structural findings held on DL20. Position consistency stayed moderate and
model-specific — Qwen at 0.714 (DL19) and 0.742 (DL20), flan-t5-large at 0.671 and 0.682 —
confirming that position inconsistency remains the single largest noise source and that
order-swapped collection cannot be replaced by a fixed correction. And once inconsistent
pairs were treated as ties, the decisive preferences were again almost perfectly
transitive: of the sampled triangles decisive on all three edges, Qwen produced 0 cyclic
triangles on DL20 (2 on DL19) and flan-t5-large 3 (6 on DL19), non-transitivity rates at or
below 0.004 everywhere. Non-transitivity is not a practical obstacle to aggregating the
verdicts into an order — which is precisely why the Copeland gate in §3.1 is legitimate.

### 3.3 The lexical agreement profile — replicated, and still near-null

The per-axiom profile for the primary top-ten condition is below: coverage (share of pairs
on which the axiom is non-neutral) for each collection, and agreement for each model ×
collection. Coverage is a property of the pairs and the axiom, so it is shared across the
two models within a collection.

| axiom | cov DL19 | cov DL20 | Qwen DL19 | flan DL19 | Qwen DL20 | flan DL20 |
|---|---|---|---|---|---|---|
| TFC1   | 0.757 | 0.784 | 0.478 | 0.466 | 0.513 | 0.520 |
| PROX1  | 0.403 | 0.353 | 0.579 | 0.613 | 0.507 | 0.584 |
| PROX2  | 0.498 | 0.414 | 0.646 | 0.672 | 0.575 | 0.648 |
| PROX3  | 0.128 | 0.069 | 0.675 | 0.707 | 0.678 | 0.689 |
| PROX4  | 0.205 | 0.203 | 0.639 | 0.595 | 0.544 | 0.546 |
| PROX5  | 0.242 | 0.231 | 0.560 | 0.536 | 0.550 | 0.546 |
| AND    | 0.214 | 0.202 | 0.780 | 0.773 | 0.798 | 0.788 |
| DIV    | 0.934 | 0.967 | 0.522 | 0.511 | 0.457 | 0.443 |
| LB1    | 0.193 | 0.184 | 0.765 | 0.804 | 0.721 | 0.713 |
| LNC1   | 0.063 | 0.037 | 0.639 | 0.539 | 0.451 | 0.391 |
| TF-LNC | 0.049 | 0.050 | 0.547 | 0.576 | 0.557 | 0.638 |
| M-TDC  | 0.008 | 0.008 | 0.833 | 0.889 | 0.462 | 0.750 |
| TFC3   | 0.001 | 0.000 | 0.500 | 1.000 | 1.000 | 1.000 |

Three things stand out. **First, the profile replicates.** The agreement vectors correlate
across collections (r = 0.79 for Qwen, 0.86 for flan over the axiom columns with n ≥ 30
evaluable pairs) and across models (r = 0.93 on DL20 top-ten). The axiomatic
characterisation is a property of LLM pairwise ranking in general on this task, not of one
checkpoint or one query set. **Second, the newly added axioms carry the signal.** AND
(0.77–0.80) and LB1 (0.71–0.80) are the strongest-agreeing axioms in every cell, with
bootstrap CIs comfortably above 0.5, joined by the proximity family in the 0.55–0.68 band;
these are the axioms whose agreement is real. **Third, the workhorse lexical axioms are at
chance.** TFC1 — the term-frequency constraint, non-neutral on three quarters of pairs —
sits at 0.47–0.52 in every cell, its CI straddling 0.5 throughout: the single most
influential classical axiom is indistinguishable from a coin flip on the top-ten pairs.
DIV, non-neutral on more than 90% of pairs, is at or below chance (0.44–0.52). The strict
core is not merely low-coverage; where it has coverage, it has no signal.

The pilot's high M-TDC number (0.83–0.89) did *not* survive: on DL20 top-ten it is 0.46
(Qwen) / 0.75 (flan) on 12–13 pairs, its CI spanning everything. As anticipated in the
design, that number was a small-sample artefact of M-TDC's ~12-pair niche, and it is
retracted along with PROX2's pilot value. Coverage was again the binding constraint on the
strict axioms — but §3.5 shows relaxing it does not manufacture the missing agreement.

### 3.4 The gap gradient is only partial

The validity check was the rank-gap gradient on the uniform control: agreement should rise
as the two documents in a pair grow farther apart in the first-stage ranking, from
near-chance on adjacent pairs to clear agreement on wide-gap pairs, confirming both that
the pipeline is sound and that the top-ten null is a genuine property of hard pairs. What
appeared was a gradient in the *model* but not, mostly, in the *axioms*. The judge's
decisive rate climbs cleanly with gap — from ~0.55 on the narrowest bins to ~0.70 on the
widest — so the model does get more confident as the pair gets easier, exactly as expected.
But per-axiom agreement mostly stays flat: TFC1 hugs chance across the range and only lifts
(to ~0.69) in the very widest decile, and DIV drifts the *wrong* way, from ~0.55 down to
~0.36 as the gap widens. The clean textbook gradient did not appear. Per the design's own
protocol, this is a finding to chase before RQ3, not one to skip: the model-side gradient
confirms the pipeline is not broken, but the axiom-side flatness says the classical
constraints do not track the ranker's growing confidence even on easy pairs, which is
carried into RQ3 as an open item.

### 3.5 Relaxing preconditions buys coverage, not agreement

Loosening the strict axioms' preconditions recovered the coverage the pilot lacked and, in
doing so, dissolved the apparent agreement. M-TDC is the clean case: relaxing its
single-difference gate takes DL19-Qwen coverage from 12 pairs to 274 (`@mass0.1`) and 725
(`@mass0.3`) pairs, and agreement falls from 0.83 to 0.54 and 0.52 — right onto chance —
with CIs that now span 0.5 because there is finally enough data to see that they do. LNC1
decays to at or below chance once its frequency margin is loosened (0.45 at `@tf0.5`), and
TF-LNC's relaxations sit at a weak 0.47–0.59. Two levers turned out degenerate and are
dropped: TFC1's length-margin relaxations (`@len0.2`, `@len0.5`) are bit-for-bit identical
to strict TFC1 because the length precondition never binds on these pools, and TFC3 stays
dead (≤ 4 evaluable pairs) at every margin. The interpretation is decisive: the strict
axioms' high pilot agreement was a property of their tiny, unrepresentative niches, and
buying coverage reveals chance-level behaviour underneath. Coverage was never hiding
signal.

### 3.6 Joint predictive power — the extensions add a little

Combining the axioms improves on the pilot but does not rescue them. A simple majority vote
of the axioms still scores at or below the majority-class base rate in every cell. A
query-grouped cross-validated L2 logistic regression, free to invert the anti-agreeing
axioms and weight the rest, does better on the *full* battery than on the strict core: on
DL19-Qwen it reaches 0.639 CV accuracy (AUC 0.670) against a 0.573 base rate, where the
strict core reached only 0.591; across the four top-ten cells the full battery lands at
0.59–0.67 accuracy (AUC 0.63–0.69), roughly four to seven points above base rate, versus the
strict core's zero-to-two. The additions (AND, LB1) and the relaxed features together buy
the extra points — but the ceiling is still low. This 0.59–0.67 is RQ3's starting number:
the best a cross-validated combination of the entire similarity-free battery achieves, and
the floor RQ3's decomposition must build up from.

### 3.7 RQ2 — the semantic tier is a null, and fastText is a no-go

The WordNet semantic axioms added nothing. STMC1 (non-neutral on ~96% of pairs) sits at
0.52–0.55 across cells; STMC2, REG and ANTI-REG hover at chance with CIs spanning 0.5 in
every top-ten cell on both collections. The design's numerical trigger for fetching the
7.24-gigabyte fastText model fired nominally twice on DL19 (STMC2@wn 0.555 at 0.077
coverage; ANTI-REG@wn 0.556 at 0.619) but neither replicated on DL20 (0.509 / 0.505), and —
the decisive test — adding the semantic features to the joint fit made it *worse* in all
four top-ten cells (Δaccuracy −0.7 to −1.6 points, ΔAUC likewise): under cross-validation
the semantic columns are noise the regulariser has to fight. A crude similarity that moves
nothing does not justify a sharper one, so **fastText is a no-go** and RQ2's answer is a
null, reported with the caveat that WordNet similarity is genuinely blunt (synonym-set
overlap, zero for out-of-vocabulary terms). The null is informative rather than
disappointing: with a top-ten residual this large, even a one-point semantic gain would
have been worth chasing, and there was not one.

## 4. Discussion

Phase 1 confirms and hardens the pilot's null. The classical axiomatic account of retrieval
— lexical *and*, at the WordNet tier, semantic — captures little of what a strong LLM
ranker does on the top-ten pairs where reranking happens, and it now survives the four
checks that could have dismissed it. It replicates across a second collection and across
two very different architectures (the cross-model, cross-collection agreement correlations
of 0.79–0.93 are the strongest evidence yet that this is a property of LLM pairwise ranking,
not of one model). It is not an artefact of a broken pipeline: the effectiveness gate shows
the ranker whose decisions the axioms miss genuinely beats BM25 on both collections, so the
residual is skill. It is not an artefact of starved coverage: relaxing the strict axioms'
preconditions recovers coverage and reveals chance-level agreement underneath, retracting
the pilot's two headline high numbers (M-TDC's 0.83, PROX2's 0.33) as small-sample and
buggy respectively. And it is not closed by adding axioms: the full similarity-free battery
plus WordNet semantics reaches only 0.59–0.67 cross-validated accuracy, and semantics
subtract under cross-validation.

The one result that did not land as designed — the gap gradient — is the most interesting
loose end. The model gets more decisive on wide-gap pairs but the axioms do not track that
confidence, and DIV actively anti-correlates with gap. The pilot's tidy story was that low
top-ten agreement would be explained by a rising gradient showing the axioms work fine once
documents are well separated; that story is only half-true here, and the half that failed
says the classical constraints are not simply "right on easy pairs, silent on hard ones"
but are weakly tracking the model's ordering even where it should be easiest for them. That
is carried into RQ3 as an open item rather than smoothed over.

What the profile *does* show is where the little signal there is lives. AND and LB1 — the
axioms the pilot never ran — are the consistently strongest, together with the proximity
family, while the term-frequency workhorses TFC1 and DIV are at chance. If any classical
structure survives into the LLM's top-ten behaviour, it is about term co-occurrence,
proximity and lower-bounding, not raw term-frequency counting; these are the axioms RQ3's
decomposition should weight, and the evidence that LLM rankers are *not* term-frequency
rankers in disguise is now much firmer than in the pilot.

Phase 1 fixes the inputs to the rest of the thesis. The RQ3 feature set is the full battery
as run — strict core, plus AND/DIV/LB1, plus the non-degenerate relaxed variants — with the
degenerate columns (TFC1's length relaxations, all of TFC3) dropped and RelaxedMTdc labelled
an inspired variant rather than literature M-TDC. Because the profile replicates, RQ3 builds
its decomposition on DL19+DL20 pooled, with per-collection numbers as robustness checks, on
the two top-ten cells (Qwen primary, flan-t5-large replication); the uniform cells stay as
validity controls feeding the gap-gradient follow-up. The semantic backend is WordNet only.
And RQ3's starting number — the best joint accuracy the whole axiom battery achieves — is
0.59–0.67, a low ceiling that makes the size of the residual, and the task of characterising
it, the main event of the phases to come.

The limitations are the pilot's, narrowed. Two collections and two models are still a small
sample, though a replicated one; the semantic null rests on a deliberately crude similarity,
which is why the fastText no-go is recorded with its numbers rather than as a verdict of
taste; and the WordNet tier cannot see term relationships absent from its synonym sets. The
gap-gradient anomaly is an acknowledged open thread. None of these move the central finding:
on the pairs that matter, the classical axioms — old and new, lexical and semantic — explain
little of what a competent LLM ranker does, and that large, stable, skill-bearing residual
is what the thesis turns to next.

*The RQ3 decomposition of the residual, and the search for constraints that discriminate
between lexically close documents, are reported in the next chapter.*
