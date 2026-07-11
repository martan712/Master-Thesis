# Phase 0 — Piloting the Axiomatic Analysis of an LLM Pairwise Ranker

*Draft chapter (Markdown; the LaTeX version follows in the writing phase). Citation
numbers of the form [n] refer to the reference list of the literature overview
(`docs/literature-overview.md`).*

## 1. Theory and motivation

Generative large language models used as pairwise rankers currently produce some of the
strongest document orderings available [1, 2], but their effectiveness comes at a price
that motivates the whole of this thesis. Such a ranker compares documents two at a time,
so ordering a list of length N costs on the order of N² comparisons, each a full forward
pass through a large model; it is therefore slow. It states only that one document beats
another and offers no account of why; it is therefore opaque. And its pairwise preferences
have been shown to be non-transitive and sensitive to the order in which the two documents
are presented [3, 4]; it is therefore inconsistent.

Axiomatic information retrieval offers precisely what these models lack. An axiom is a
common-sense constraint on a scoring function, stated as a thought experiment about two
documents that differ in one controlled way — for instance, that a document containing a
query term more often than an otherwise identical document should not be ranked below it.
Such constraints have been developed over two decades for sparse lexical rankers [7, 8]
and organised into a broader programme covering term-frequency, term-discrimination,
length-normalisation, proximity and semantic families of constraint. Crucially, these
constraints are pairwise statements about a *pointwise* and therefore transitive scoring
function: the pairwise reasoning is a design device, and the object actually deployed is a
per-document score, which is why classical axiomatic rankers are both transitive and
linear-time. A generative pairwise ranker inverts this — it answers "is one document
better than the other?" directly and keeps no underlying score — which is exactly the
source of its non-transitivity and its quadratic cost.

There is reason to expect that axiomatic structure can be recovered from a neural ranker
even when nobody put it there: cross-encoders have been shown to reimplement recognisable
variants of classical retrieval behaviour, and causal interventions have isolated
components that behave like term-frequency counting [20, 21, 23]. Recent work has begun to
look inside LLM rankers mechanistically and through probing [5, 6], but none of it frames
the model's pairwise preferences as the satisfaction of retrieval axioms, derives new
axioms from where the model departs from the classical ones, or uses axioms as a cheap
stand-in for the model. The thesis addresses that gap with a single question: how far can
the pairwise preferences of a generative LLM ranker be reduced to an interpretable set of
retrieval axioms, and what is the nature of the part that cannot? To the extent the model
is axiom-reducible, one obtains an interpretable, transitive and fast surrogate; to the
extent it is not, one has isolated the behaviour the classical axioms miss and can try to
formalise it into new axioms.

This chapter reports the pilot that made those questions measurable. Its purpose was not
to answer the research questions but to fix the experimental machinery — the collections,
the pair sampling, the ranker prompt, the axiom battery, and the definitions of agreement,
consistency and transitivity that every later phase reports against — and to produce the
first real measurements on which the rest of the study is planned.

## 2. Experimental setup

**Collection and pairs.** The pilot used the judged queries of the TREC Deep Learning 2019
passage task over the MS MARCO passage corpus — 43 queries with relevance judgements. For
each query, a BM25 first-stage retrieval produced a pool, and the ten top-ranked passages
were taken all-pairs: every one of the forty-five unordered document pairs among the top
ten. This top-ten condition was chosen deliberately, because it is where reranking
decisions are actually made, and because complete pairwise coverage of a set of documents
is required to measure non-transitivity through closed triangles. Across the 43 queries
this gave 1,900 unordered pairs. Each pair was presented to the ranker in both of its two
possible orders, giving 3,800 presentations per model; the presentation order was treated
throughout as an explicit experimental dimension rather than a nuisance to be averaged
away.

**Ranker and the sanity gate.** The ranker followed the pairwise-ranking-prompt style of
prior work [2]: the model was shown two passages labelled A and B and asked which better
answered the query, and its verdict was read from the likelihood it assigned to the two
label continuations rather than from free text. This scoring is deterministic, needs no
sampling temperature, and yields the probability mass on "A" as a natural confidence
signal. Before any full collection, every candidate model had to pass a four-way
order-swap sanity check on two obvious relevant-versus-irrelevant pairs, each shown in both
orders; a model that could rank pairwise at all should answer consistently when the
passages are swapped. The smallest sequence-to-sequence model failed completely,
answering "A" in all four presentations regardless of content — a pure position bias
consistent with the original finding that only larger Flan-T5 variants can rank pairwise
[2]. The base-sized model also failed, correct in three of four cases but with
probabilities hovering around one half. Two models passed cleanly and were carried forward:
a 0.8-billion-parameter sequence-to-sequence model (Flan-T5-large), which was decisive
with label probabilities at least 0.998, and a 35-billion-parameter mixture-of-experts
chat model (Qwen). For the larger chat model, the model's internal "thinking" was disabled
and verified to produce bare "A"/"B" answers, since intervening reasoning tokens would
break the label-likelihood scoring.

**Preference collection.** Because collecting the model's verdicts is the expensive part of
the whole thesis, every verdict was computed once and cached in an append-only store keyed
by collection, query, ordered document pair, model and prompt version, and never
recomputed. This gives a reusable preference dataset that all later phases draw on without
paying again. The larger chat model was served behind an inference endpoint and its verdict
read from a single output token's log-probabilities; the sequence-to-sequence model was run
locally on CPU.

**Axiom battery.** The pilot ran the similarity-free lexical battery available in the
tooling: the term-frequency constraint TFC1 and the related TFC3, the term-discrimination
constraint M-TDC, the length-normalisation constraints LNC1 and TF-LNC, and the five
proximity constraints PROX1 through PROX5. The embedding-based semantic constraints were
held back after their similarity backend was measured to require a 7.24-gigabyte download,
which was judged out of scope for a pilot and deferred to the semantic study.

**Definitions.** Four definitions were fixed at this point so that all later phases would
be comparable. The *model's verdict on an unordered pair* was derived from its two
presentations: if the two orders disagreed, the pair was recorded as position-inconsistent
and its verdict treated as a tie. The *agreement* of an axiom was measured over those pairs
where the axiom is non-neutral and the model is decisive, as the fraction on which the two
point in the same direction, and was always reported alongside the axiom's *coverage*, the
fraction of all pairs on which it is non-neutral. The *position-consistency rate* was the
fraction of pairs on which the model gave identical verdicts in both orders. The
*non-transitivity rate* was the fraction of cyclic triangles among those triangles on
which the model was decisive on all three edges. Throughout the pilot the primary quantity
was fidelity — how well the axioms characterise the model — rather than agreement with the
human relevance judgements.

## 3. Results

**Position consistency and its direction.** The model's verdicts were only moderately
stable under order swap: the chat model repeated its verdict on 0.714 of pairs and the
sequence-to-sequence model on 0.671, making position inconsistency the single largest
source of noise in the data. More informative than the rate was its direction. The two
models were biased in *opposite* directions — the chat model tended toward the
second-shown passage and the sequence-to-sequence model toward the first — which shows
that position bias is not a fixed artefact of the prompt template but a model-specific
property. The underlying verdict counts made the same point: the chat model's raw split
was 1,448 first-passage / 2,276 second-passage / 76 ties, while the local model's was
2,484 / 1,316 / 0.

**Transitivity.** Once position-inconsistent pairs were treated as ties, the decisive
preferences were almost perfectly transitive. Of the 5,050 sampled triangles, 2,096
(41.5%) were decisive on all three edges for the chat model and 1,699 (33.6%) for the
sequence-to-sequence model; among these, only 2 and 6 respectively were cyclic, giving
non-transitivity rates of 0.001 and 0.004. Non-transitivity is therefore not a practical
obstacle to rank aggregation here. The number of complete triangles is itself a
consistency signal: because a triangle survives only if the model is decisive on all its
edges, a model that flip-flops more under order swap contributes fewer usable triangles,
which is why the less consistent model also had the smaller complete-triangle count.

**Per-axiom agreement and the coverage collapse.** The per-axiom profile is given below,
with coverage over all 1,900 pairs and agreement over the evaluable pairs for each model.

| axiom | coverage | agreement (chat) | agreement (seq2seq) |
|---|---|---|---|
| TFC1 | 0.757 | 0.478 | 0.466 |
| PROX1 | 0.395 | 0.597 | 0.622 |
| PROX2 | 0.498 | 0.354 | 0.328 |
| PROX3 | 0.128 | 0.675 | 0.707 |
| PROX4 | 0.205 | 0.639 | 0.595 |
| PROX5 | 0.242 | 0.560 | 0.536 |
| LNC1 | 0.063 | 0.639 | 0.539 |
| TF-LNC | 0.049 | 0.547 | 0.576 |
| M-TDC | 0.008 | 0.833 | 0.889 |
| TFC3 | 0.001 | 0.500 | 0.500 |

Two features stand out. First, the strict-precondition axioms were almost entirely dead:
TFC3 fired on 0.1% of pairs, M-TDC on 0.8%, TF-LNC on 4.9% and LNC1 on 6.3%. Their
equal-length and equal-frequency preconditions, which require two natural passages to match
almost exactly on some dimension, are rarely satisfied. The usable lexical core was
therefore TFC1 together with the five proximity axioms, spanning 13% to 76% coverage; taken
together, 94.6% of pairs had at least one non-neutral axiom, so the low per-axiom coverage
was not a failure to reach the pairs. Second, the profile *shape* was strikingly consistent
across two very different architectures. Both a 0.8-billion-parameter sequence-to-sequence
model and a 35-billion-parameter mixture-of-experts model placed TFC1 *below* chance (about
0.47), PROX2 markedly anti-agreeing (0.33–0.35), and PROX3 highest (0.68–0.71). Where M-TDC
fired at all it was the best-agreeing axiom (0.83–0.89), but on well under one percent of
pairs.

**Joint predictive power.** Combining the axioms did not rescue them. Predicting each
model's decisive verdict from all ten axiom preferences at once, a simple majority vote of
the axioms scored *below* the majority-class base rate — 0.488 against a base rate of 0.573
for the chat model, and 0.470 against 0.600 for the sequence-to-sequence model. A
cross-validated regularised logistic model, free to invert the anti-agreeing axioms and
weight the rest, reached only 0.601 and 0.611 respectively, roughly one to three accuracy
points above the base rate. The fitted structure was itself stable across the two models —
the term-discrimination constraint strongly positive, the second proximity constraint
negative — so the battery failed in a systematic, model-independent way rather than
noisily.

**Confidence and coverage were orthogonal.** The correlation between the model's confidence
and axiom coverage was 0.004 for the chat model and −0.131 for the sequence-to-sequence
model. The axioms do not simply fire on the pairs the model is sure about, which means that
coverage-weighted fidelity and confidence-stratified analyses measure genuinely different
things.

## 4. Discussion

The central pilot result is a near-null: on the top-ten pairs that a reranker is actually
deployed to reorder, the classical lexical axioms explain almost nothing of a strong LLM
ranker's decisions, jointly buying only one to three accuracy points over guessing the
majority class, and a naive vote of them does worse than that. The natural reading is not
that the axioms are wrong but that they are redundant with the first stage: both members of
a top-ten pair are already lexically strong, so classical lexical constraints largely
re-explain what BM25 has already decided, and the marginal value the LLM adds over BM25 —
the very reordering one runs the reranker for — sits almost entirely in the residual that
the axioms do not touch. This distinction matters for interpreting the number: low
agreement on easy, wide-gap pairs would indicate a broken pipeline, whereas low agreement
concentrated on the hard top-ten pairs is the phenomenon under study.

That reading also frames what the next phase must check. If the account is right, agreement
should rise as the two documents in a pair become more separated in the first-stage
ranking, producing a gradient from near-chance on adjacent pairs to clear agreement on
wide-gap pairs; observing that gradient would show the pipeline is sound and the top-ten
null is a property of LLM reranking rather than an artefact. The measurement phase is
therefore built around validity checks of this kind — the gap gradient, a second
collection for replication, relaxed axiom preconditions to recover coverage on the hard
pairs where the one high-agreement axiom is starved of evidence, and a first semantic tier —
rather than around a search for a larger agreement number. More broadly, a large and stable
residual shifts the weight of the thesis toward its later questions: characterising that
residual and formalising new constraints that discriminate between lexically close
documents become the main event, rather than the contingency they were planned as.

Several encouraging secondary findings emerged. The replication of the agreement profile
across a sub-billion-parameter sequence-to-sequence model and a 35-billion-parameter
mixture-of-experts model is early evidence that the axiomatic account characterises LLM
pairwise ranking in general, not one checkpoint, and that these rankers are not merely
term-frequency rankers in disguise. The near-perfect transitivity of decisive preferences
means non-transitivity is not a practical obstacle to aggregating the pairwise verdicts
into an order. And the opposite-direction position biases confirm that order-swapped
collection must remain mandatory, since averaging over orders cannot be replaced by a fixed
correction that would work for every model.

The limitations should be stated plainly. The pilot rested on a single collection of
43 queries, two models, and one prompt family, and it used a prebuilt first-stage index
whose tokenisation need not match what the axioms assume, any of which could move the
numbers. The strict-precondition axioms were measured essentially at zero coverage, so
their high apparent agreement rests on a handful of pairs and may not survive when their
preconditions are relaxed. These are exactly the threats the measurement phase is designed
to probe.

*The Phase 1 measurement study (the lexical and semantic agreement profiles) is reported
separately once its results are in.*
