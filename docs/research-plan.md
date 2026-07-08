# Research Proposal
### Axiomatic reduction of generative-LLM pairwise rankers: explanation, new axioms, and acceleration

This document sets out the plan for the thesis. The background, the theoretical framework
and the full reference list live in `literature-overview.md`, in the same folder, and the
citation numbers of the form [n] used below point to its reference list. Here we assume
that context and concentrate on what we intend to do, why, and in what order.

## 1. Problem statement

Generative LLMs used as pairwise rankers (PRP [2]) and as listwise rankers (RankGPT [1])
currently produce some of the strongest rankings available, but their effectiveness comes
at a price. They are slow, because a pairwise ranker compares documents two at a time and
so needs on the order of N² calls to the model, each of which is a full forward pass. They
are opaque, because the model simply states that one document beats another and offers no
account of why. And they are inconsistent, because their pairwise preferences have been
shown to be non-transitive and sensitive to the order in which the two documents are
presented [3, 4]. Axiomatic information retrieval offers what these models lack, namely a
set of pairwise constraints on a pointwise and therefore transitive scoring function
[7, 8]. Recent work has started to look inside LLM rankers [5, 6], but none of it treats
the model's pairwise preferences as the satisfaction of retrieval axioms, none derives new
axioms from the cases where the model departs from the classical ones, and none uses
axioms as a cheap stand-in for the model. It is this gap that we address.

We can phrase the aim of the thesis as a single question. How far can the pairwise
preferences of a generative LLM ranker be reduced to an interpretable set of retrieval
axioms, and what is the nature of the part that cannot? The value of the question does not
depend on the answer going one way. To the extent that the model is axiom-reducible, we
obtain an interpretable, transitive and fast surrogate for it. To the extent that it is
not, we have isolated the behaviour that the classical axioms miss and can try to formalise
it into new axioms.

## 2. Research questions

We break the central question into six smaller questions. The first three measure how far
the existing axioms take us, the fourth turns whatever is left over into new axioms, and
the last two try to cash the explanation out as a faster ranker. The questions depend on
one another in that order, so that a weak result on one becomes the starting point for the
next rather than a dead end.

- RQ1 asks which of the classical lexical axioms, that is the term-frequency,
term-discrimination, length-normalisation and proximity constraints, agree with a pairwise
LLM ranker's verdicts on document pairs, and how often each of them does so.

- RQ2 asks whether adding the semantic axioms, the semantic term-matching constraints [10]
and their embedding-based variants [11], raises this agreement beyond what the lexical
axioms achieve on their own, and by how much.

- RQ3 asks how much of the model's pairwise preferences a combined lexical and semantic
axiom model can predict, and how large and how systematic the unexplained remainder is.
This question produces the decomposition of the model's behaviour into an axiom-explained
part and a residual, and the residual is what the next question works on.

- RQ4, which we regard as the main contribution, asks whether that systematic residual can
be formalised into one or more new axioms, and whether adding those axioms improves
axiomatic re-ranking over the classical set. This is deliberately placed so that the worse
the earlier questions turn out for the axioms, the more there is to discover here.

- RQ5 asks whether an axiom-based pointwise scoring function can reproduce the model's
ranking, and what the trade-off is between the effectiveness we give up and the cost we
save by moving from an N² model to a linear-time scorer.

- RQ6, which we treat as a stretch goal, asks whether axiom disagreement can flag the
comparisons on which the model is inconsistent, and whether that signal can be used to
build a cascade that calls the model only where it is needed, at a bounded loss in
effectiveness.

## 3. Background and positioning

The full treatment is given in `literature-overview.md`; we recall only what is needed to
position the plan. Axioms are well established for sparse rankers [7, 8], are becoming
established for dense and neural rankers as tools for diagnosis and regularisation
[14, 15, 16, 17], and have been shown to be latent inside neural rankers even when nobody
put them there [20, 21, 23], which is our main reason for expecting to recover them in LLM
rankers as well. The nearest prior work analyses the internals of LLM rankers, either
mechanistically [5] or through probing [6], but builds neither an interpretable functional
model, nor new axioms, nor an efficiency method on top. The tooling we need already exists
in `ir_axioms` [19] and `ir_explain` [36], and the bridge from pairwise preferences to a
pointwise score is provided by the Bradley-Terry model [28]. In short, we extend the
axiomatic line of work from neural cross-encoders to generative LLM rankers, and we add the
efficiency payoff that the existing work does not pursue.

## 4. Methodology

### 4.1 Shared setup

All six questions draw on a common experimental setup, which we describe once here. For
collections we use the TREC Deep Learning passage tasks and MS MARCO for the core study,
and a selection of BEIR datasets for generalisation, with the Touché 2020 argument data
serving as a semantic stress test [46]. For the ranker we use an open and reproducible
pairwise model in the style of PRP [2], with RankVicuna [41] and RankZephyr [42] available
as listwise references, and we retrieve a first-stage pool with BM25 before sampling
document pairs from it. Because collecting the model's verdicts is the expensive part, we
run the model once over the sampled pairs and cache the results, which gives us a reusable
preference dataset for the whole thesis. For the axioms we use the battery described in the
literature overview, implemented through `ir_axioms` [19] and `ir_explain` [36], where each
axiom returns a preference for one document, the other, or neither.

We rely on a small set of metrics throughout. Agreement is the fraction of pairs on which
an axiom and the model give the same verdict, setting aside the pairs the axiom calls
neutral. Predictive power is the accuracy or area under the curve of a model that predicts
the LLM's verdict from the axiom features. Effectiveness is measured with nDCG at ten and
with MAP, efficiency with the FLOP-based RPP and QPP metrics [24] together with the count
of model calls and the latency, fidelity with the rank correlation to the model's own
ranking, and consistency with the rate of non-transitive triples. One point of discipline
matters here and recurs below: each question should say in advance whether it is
optimising fidelity, meaning how well we reproduce the model, or effectiveness, meaning how
well we match the ground-truth judgements, because the two can diverge.

### 4.2 How each question is answered

For RQ1 we compute, over the cached pair set, the agreement of each lexical axiom with the
model's verdicts, which gives us a per-axiom profile of where the model already behaves
like classical IR. This is descriptive and so always yields a result. For RQ2 we add the
semantic axioms and compare the lexical-only battery against the combined one, reporting
the change in agreement and in predictive power; here a clean null result, showing that
semantics add nothing, would be as informative as a positive one.

For RQ3 we fit an interpretable model, such as a regularised logistic regression or a
shallow tree, that predicts the model's verdict from all the axiom votes together, and we
report both how much of the behaviour it captures and what the mispredicted pairs have in
common. The analysis of those pairs is what decides the emphasis of RQ4: a rich and
systematic residual points towards a new-axiom study, whereas a thin residual would tell us
that the model is largely axiom-reducible and would shift the weight of the thesis towards
RQ5. In RQ4 we take the systematic residual, try to characterise it, for instance as
contextual behaviour, a form of semantic mismatch, or an effect that depends on the other
documents present, and formalise it as one or more new axioms, which we then add to the
re-ranker and evaluate against the classical set with nDCG. If the residual turns out to be
unsystematic, the honest boundary result is itself a contribution.

For RQ5 we fit a Bradley-Terry style pointwise score over the axiom features against the
model's pairwise labels [28], deploy it in linear time, and compare its ranking to the
model's both in effectiveness and on the efficiency frontier, so that the outcome is either
a surrogate that reproduces the model within a stated margin at a large saving, or a
characterisation of how much effectiveness a linear-time axiomatic scorer must give up.
For RQ6 we test whether the pairs on which the axioms disagree are the pairs on which the
model is inconsistent, and if so we build a cascade that lets the axioms settle the
confident pairs and calls the model only on the rest, measuring how many model calls we
save at a fixed level of effectiveness.

## 5. Timeline and milestones

The schedule below assumes a full-time span of about twenty-six weeks, which is the usual
length of a master's project; it should be stretched or compressed to match the actual
duration. We work in phases that follow the dependency order of the questions, and each
phase ends in a concrete milestone.

| Phase | Weeks | Focus | Milestone |
|---|---|---|---|
| 0 — Setup and pilot | 1–4 | Data, the ranker harness, the axiom toolkit, and the preference-logging pipeline, piloted on DL19 | A cached preference dataset and a working agreement pipeline |
| 1 — Measurement (RQ1–2) | 5–9 | Lexical and then semantic agreement studies | The per-axiom and semantic agreement profiles |
| 2 — Decomposition (RQ3) | 10–14 | The combined axiom model and the residual analysis | The explained/residual decomposition, and a decision on the emphasis of RQ4 |
| 3 — New axioms (RQ4) | 15–19 | Deriving and validating new axioms | One or more new axioms with a re-ranking evaluation |
| 4 — Acceleration (RQ5, and RQ6 if time) | 20–23 | The pointwise surrogate and the efficiency study, then the cascade | The effectiveness–efficiency results |
| 5 — Writing and buffer | 24–26 | Writing up, with slack for overruns | A thesis draft |

## 6. Deliverables

By the end of the project we expect to produce a reusable dataset of the model's pairwise
preferences together with the harness that evaluates axioms against the model; the
decomposition of a pairwise LLM ranker into an axiom-explained part and a residual, from
RQ1 to RQ3; one or more new retrieval axioms with a re-ranking validation, from RQ4, which
we regard as the primary novel contribution; an axiom-based pointwise surrogate with a
benchmark on the effectiveness–efficiency frontier, from RQ5, and where time allows a
selective cascade from RQ6; and, finally, the thesis itself along with a reproducibility
package.

## 7. Risks and mitigations

Several things could go wrong, and it is worth saying in advance how we would respond. The
most obvious risk is that the axioms explain rather little of the model, giving a weak
result on RQ3. This is less damaging than it looks, because a large residual simply gives
RQ4 more to work with; only if the residual is also unsystematic do we fall back on
reporting a boundary result on how far axioms can go. A second risk is that running the
model over enough pairs is computationally heavy, which we address by choosing a small open
model, caching every verdict once, and sampling pairs from the BM25 pool rather than taking
all of them. A third is that a new axiom fails to generalise, in which case we report the
negative result and the analysis of why, which is still of value. A fourth is that the
pointwise surrogate underperforms in RQ5, which is exactly why RQ6 exists: a partial
replacement that routes only the hard pairs to the model still yields an efficiency result.
There is also the open question of which similarity function should drive the semantic
axioms, which we treat as an ablation over static embeddings, an external encoder, and the
model's own hidden states, and the risk that fidelity and effectiveness point in different
directions, which we handle by fixing the primary metric of each question in advance, as
described in Section 4.

## 8. Scope and assumptions

We limit ourselves to English passage ranking, to one or two open LLM rankers, and to the
pointwise and pairwise axis rather than the internals of listwise generation. Cross-lingual
retrieval, the effect of document length, and production-latency concerns are out of scope.
A handful of design choices are genuinely open at the start and are best settled by the
Phase 0 pilot rather than argued here in advance: whether a pointwise score is expressive
enough to reproduce the model's order at all, which similarity function the semantic axioms
should use, whether the pairs that the axioms explain well are the same pairs on which the
model is confident, and which of fidelity or effectiveness each experiment should treat as
primary. These are the same open questions recorded in `literature-overview.md`.
