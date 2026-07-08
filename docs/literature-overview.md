# Literature Overview and Theoretical Framework
### Axiomatic explanation and acceleration of generative-LLM pairwise rankers

In this document we bring together the literature that underpins the thesis and lay out
the theoretical framework that we build on in the research plan. Citation numbers of the
form [n] refer to the References at the end, which mirror the Zotero "Master Thesis"
collection. The metadata there is best-effort, so venues and identifiers should be
checked before they are used in the thesis proper.

## 1. Thesis problem description

Axiomatic information retrieval starts from a simple idea. A good ranking function ought
to obey a handful of common-sense constraints, which we call axioms. If, for example, one
document contains a query term more often than another and the two are otherwise
identical, then the first should not be ranked below the second. Over the past two decades
a fairly large collection of such constraints has been proposed and used to build, to
diagnose, and to explain retrieval models. The programme is well developed for sparse
lexical rankers such as BM25 and query likelihood [7, 8], it has begun to be applied to
dense neural rankers, and it has barely been applied to the generative decoder LLMs that
are increasingly used as rankers. It is this last setting that we are interested in.

LLMs used as pairwise or listwise rankers currently produce some of the strongest
rankings available (RankGPT [1], PRP [2]). Their effectiveness, however, comes with three
problems. They are slow: a pairwise ranker compares documents two at a time, so ordering
a list of N documents costs on the order of N² comparisons, or N log N if the comparisons
are arranged as a sort, and every comparison is a full forward pass through the model.
They are opaque: the model simply asserts that one document beats another, and gives us no
account of why. And they are inconsistent: their pairwise preferences have been shown to
be non-transitive, so that a model may prefer A over B, B over C, and yet C over A, and to
depend on the order in which the two documents are presented [3, 4].

Axiomatic IR offers precisely what these models lack, namely a faithful, cheap and
readable account of why one document should be preferred to another. Recent work has begun
to look inside LLM rankers, either mechanistically [5] or through probing [6], but none of
it frames the model's pairwise preferences as the satisfaction of retrieval axioms, none
derives new axioms from the cases where the model departs from the classical ones, and
none uses axioms as a cheap substitute for the expensive model. This is the gap we set out
to address.

We can state the aim of the thesis as a single question. How far can the pairwise
preferences of a generative LLM ranker be reduced to an interpretable set of retrieval
axioms, and what is the nature of the part that cannot? The question is worth asking
whichever way it turns out. In so far as the model is axiom-reducible, we obtain an
interpretable, transitive and fast surrogate for it. In so far as it is not, we have
isolated the behaviour that the classical axioms miss, and we can try to formalise that
behaviour into new axioms. To make this tractable we break the question down into six
smaller research questions, which we return to in Section 3 and which drive the research
plan:

- RQ1 asks which of the classical lexical axioms agree with an LLM ranker's pairwise
  verdicts, and how often.
- RQ2 asks whether adding semantic axioms raises that agreement beyond the lexical
  axioms alone.
- RQ3 asks how much of the model's preferences a combined axiom model can predict, and
  how large and how systematic the unexplained remainder is.
- RQ4 asks whether that remainder can be formalised into one or more new axioms that
  improve axiomatic re-ranking.
- RQ5 asks whether an axiom-based pointwise scorer can reproduce the model's ranking, and
  at what trade-off between effectiveness and cost.
- RQ6, a stretch question, asks whether axiom disagreement can flag the model's
  inconsistent comparisons and so drive a selective, cheaper cascade.

## 2. Background and theoretical framework

### 2.1 Axiomatic IR and the constraint view

An axiom is a constraint stated as a thought experiment about two documents. We take two
documents that differ in one controlled way, for instance one contains a query term once
more than the other while everything else is held equal, and we require that the scoring
function `s(q,d)` order them in the expected direction. This way of thinking began with
the sparse lexical models [7, 8] and has since been organised into a broader axiomatic
programme [9]. The constraints come in families. There are term-frequency constraints,
which say that seeing a query term more often should help; term-discrimination
constraints, which say that matching a rare term should count for more than matching a
common one; length-normalisation constraints, which keep long documents from winning
simply by being long; and proximity constraints, which reward query terms that appear
close together. Alongside these lexical families there are semantic constraints, which
allow related but non-identical terms to match [10] and which have later been generalised
to embedding-based similarity [11], and there are domain-specific constraints tailored to
particular tasks such as argument retrieval [12] and retrieval-augmented generation [13].

Axioms play four roles in the literature, and it is worth separating them. They can be
used to construct or guide a retrieval function; they can be used to regularise or
supervise a neural ranker during training [14]; they can be used to diagnose or explain
the behaviour of a black-box ranker after the fact [15, 16, 17]; and they can be used to
re-rank an initial result list, treating each axiom as a feature [18, 19]. The work we
propose lives mainly in the third and fourth of these roles, extended for the first time
to generative LLM rankers.

### 2.2 Where axioms have reached across ranker families

It helps to see where the axiomatic programme currently stands for each kind of ranker.
For sparse lexical rankers such as BM25 the axioms are native, since these models were in
part designed around them. For dense and neural rankers the picture is less settled but
developing quickly, with axioms used for diagnosis, for regularisation, and more recently
as a lens for mechanistic interpretability [20, 21]. For the generative LLM rankers that
concern us, axiomatic coverage is essentially absent, which is the gap the thesis targets.

There is, however, encouraging evidence that axiomatic structure is already present inside
neural rankers even when it was never put there deliberately. A MiniLM cross-encoder has
been shown to reimplement a recognisable variant of BM25 [20], causal interventions have
isolated components that behave like term-frequency counting [21], and language models
have been found to carry out simple, interpretable vector arithmetic [23]. If such
structure is latent in neural rankers, it is reasonable to expect that it can also be
recovered in LLM rankers, which is the premise on which our explanation questions rest.

### 2.3 Generative LLM rankers and the efficiency problem

The way an LLM is prompted to rank sits on a trade-off between cost and reliability. A
pointwise prompt asks the model to score each document on its own, which is cheap but
gives unreliable absolute numbers. A pairwise prompt asks it to compare two documents,
which yields the best relative judgements but costs on the order of N² comparisons [2]. A
listwise prompt asks it to order a whole window of documents at once, as RankGPT does with
a sliding window [1], and a setwise prompt compares small sets and sorts, which reduces
the number of calls further [22]. Because these methods are expensive, their efficiency is
now measured in a hardware-independent way through FLOP-based metrics such as RPP and QPP
[24], and the common ways of making them faster are distillation into a cheaper model [25]
and single-pass aggregation [26]. There is also the question of where an expensive ranker
should sit within a larger pipeline, which has been studied under the heading of compound
retrieval systems [27]. What none of these approaches provides is an interpretable
efficiency lever, which is one of the openings we pursue.

### 2.4 From pairwise comparisons to a ranking

At the centre of the thesis lies a question that is easy to overlook: once we have a set of
axioms that each compare two documents, how does that turn into a ranking? This is the
classical problem of turning pairwise preferences into a total order, and it has a mature
theory behind it.

The most useful part of that theory for us is the family of scoring-function models, of
which the Bradley-Terry model is the canonical example [28]. Such a model assumes a hidden
score for each document and explains the probability that one document is preferred over
another in terms of the difference in their scores. Fitting the model recovers a single
score per document, and sorting by that score gives a ranking that is transitive by
construction and can be produced in linear time once the scores are known. Bradley-Terry
is therefore the bridge that takes us from pairwise preferences back to a pointwise score.
There are also voting and tournament rules that produce an order directly from pairwise
outcomes, such as Copeland, Condorcet, Borda and the Kemeny-Young rule, as well as
Markov-chain methods that rank documents by the stationary distribution of a random walk
over the comparison graph [29]. And there is a body of work on efficient and noisy sorting
which shows that a good order can be recovered from far fewer than all N² comparisons,
which matters if we keep some model calls in the loop.

This same view resolves the apparent tension between axioms and LLM rankers. The classical
axioms are pairwise constraints placed on a pointwise scoring function. The pairwise
reasoning is only a device for reasoning about the design; the object that is actually
deployed is a score, which is why classical axiomatic rankers are transitive and run in
linear time. A generative LLM pairwise ranker inverts this. It answers "is d₁ better than
d₂?" directly and keeps no underlying score, which is exactly why it can be non-transitive
and why it costs N² comparisons [3]. In the language of learning to rank, which separates
pointwise, pairwise and listwise approaches [30, 31], the axioms map most naturally onto
the pairwise approach, since a pairwise constraint is precisely what trains a scoring
function.

It is worth being careful here about where the cost of ranking actually lives, because it
is tempting to claim too much for axioms. Replacing an expensive model comparison with a
cheap axiom test does not change the number of comparisons; a pairwise scheme still makes
N² of them. What it changes is the cost of each comparison and, in particular, the number
of times we have to call the model. The asymptotic cost only falls to linear when the
axioms combine into a pointwise score, so that we can score each document once and sort.
There are therefore three regimes worth keeping apart. In the first, we run the pairwise
LLM directly, which is the most expensive and offers no guarantee of transitivity. In the
second, we use cheap axioms to settle the easy comparisons and call the model only on the
ones the axioms cannot resolve, which cuts the number of model calls sharply but leaves
the number of comparisons, and the model's own consistency, unchanged. In the third, we fit
a pointwise axiomatic score and drop the model at inference entirely, which is transitive
and linear but only as good as the score we can fit. The third regime is the most valuable
target, because it buys explanation, consistency and speed at the same time, and much of
the thesis is about how far we can push towards it.

### 2.5 Which axioms, and the semantic question

The set of axioms we start from combines the classical lexical constraints with semantic
ones. On the lexical side we have the term-frequency, term-discrimination,
length-normalisation and proximity constraints described above, all of which decompose
into a pointwise score and each of which captures a familiar aspect of relevance, such as
lexical overlap, the weighting of rare terms, control for verbosity, and the value of
query terms appearing close together. On the semantic side we have the semantic
term-matching constraints [10] and their embedding-based generalisations [11], which
capture paraphrase and vocabulary mismatch and which can still be written as a pointwise
score once a similarity function is fixed. To these we may add the domain-specific
constraints for argument retrieval [12] and retrieval-augmented generation [13] where the
task calls for them.

The interesting category is the one we do not yet have. If the advantage that an LLM ranker
holds over BM25 lies in genuinely contextual behaviour, such as comparing documents against
one another or rewarding novelty, then no pointwise axiom set can fully explain it, because
such behaviour does not decompose into a per-document score. This is the boundary that RQ3
is meant to locate and that RQ4 is meant to push against, either by formalising new
semantic and contextual constraints or by reporting an honest limit on what axioms can
capture. That classical axioms are already known to be incomplete for modern models is not
merely a supposition; recently identified failures such as the over-penalisation of extra
information make the point concretely [32].

### 2.6 Explainability neighbours and the competitive landscape

The thesis sits within the broader field of explainable information retrieval [33]. Two
lines of work are close enough to serve as points of comparison. The first is feature
attribution, where RankingSHAP attributes a ranking to input features in a listwise way
[34] and LiEGe generates listwise aspect-based explanations [35]; both offer a contrast to
the kind of explanation an axiomatic account provides. The second is the tooling that makes
axiomatic experimentation practical, namely `ir_axioms`, which brings axioms into PyTerrier
and Pyserini [19], and its broader successor `ir_explain` [36], either of which can serve
as an experimental backbone.

Most of the directly relevant recent activity comes from a single, fast-moving group around
Chen and Eickhoff, which has produced a mechanistic-interpretability tool for IR [37], a set
of best practices for axiomatic activation patching [38], a tutorial [39], probing and
mechanistic analyses of LLM rankers [5, 6], and a study of how role-play prompts move an
LLM ranker's judgements [40]. The two nearest pieces of prior work are the mechanistic
account of how LLMs form pointwise and pairwise relevance judgements [5] and the probing
study that asks whether an LLM reranker's internal features line up with human-engineered
ones [6]. Neither of them, however, frames the model's pairwise preferences as axiom
satisfaction, and neither builds an efficiency method on top. Our position is therefore to
extend this axiomatic-intervention line from neural cross-encoders to generative LLM
rankers, and to add the efficiency payoff that this group has not pursued. We should note
that the area is moving quickly, with a steady stream of 2025 and 2026 work, so the
pairwise-efficiency angle is open now but may not stay open for long.

## 3. Broad methodology and justification

The framing we adopt is to explain a pairwise LLM ranker with a combined lexical and
semantic axiom set and then to distil those axioms into a pointwise surrogate. We prefer
this framing because it delivers interpretability and a genuine reduction in asymptotic
cost within a single story, and because it occupies ground that neither of the nearest
prior works [5, 6] has taken.

In outline the work proceeds as follows, and the research plan gives the detailed version.
We first assemble the axiom battery, combining the lexical constraints with the semantic
term-matching constraints [10] and their embedding variants [11], and we implement it using
the existing tooling [19, 36] so that our axioms remain comparable with earlier work. We
then choose an open and reproducible pairwise ranker in the style of PRP [2], with
RankVicuna [41] and RankZephyr [42] available as listwise references, and we use it to
label document pairs drawn from standard collections such as TREC Deep Learning and BEIR,
with the Touché argument data serving as a semantic stress test. With this data in hand we
measure, for each axiom and then for a combined model, how well the axioms predict the
model's pairwise decisions, which answers the explanation questions and, through the
residual, tells us what the axioms are missing. We then try to formalise that residual into
new axioms and test whether they improve axiomatic re-ranking. Finally we fit a
Bradley-Terry style pointwise score over the axiom features [28], deploy it in linear time,
and report the gap in effectiveness against the model alongside the gain in efficiency on
the RPP and QPP frontier [24]; where full replacement falls short we fall back to using
axiom agreement to decide which comparisons still need the model, and we test whether the
model's inconsistent comparisons [3] coincide with axiom disagreement.

Two points of experimental discipline run through all of this. The first is that we must
distinguish fidelity, meaning how well we reproduce the model's own ranking, from
effectiveness, meaning how well we match the ground-truth relevance judgements, because the
two can pull in different directions and each research question should say which it is
optimising. The second is that several design choices are genuinely open at the outset and
are best settled by a pilot rather than argued in advance: whether a pointwise score is
expressive enough to reproduce the model's order at all, which similarity function should
drive the semantic axioms, whether the pairs that axioms explain well are the same pairs on
which the model is confident, and which of fidelity or effectiveness a given experiment
should treat as primary.

## 4. In-collection references not cited above

The following references are in the Zotero collection but are not cited in Sections 1 to 3.
We keep them for orientation and for possible use in later parts of the thesis, and give a
short reason for each.

| # | Reference | Why not cited |
|---|---|---|
| 43 | Interpretability in Neural IR (thesis) | A sibling thesis kept for orientation; no specific claim is drawn from it. |
| 44 | catherineschen/axiomatic-ir-interventions (code) | The software behind [21], cited through the paper rather than separately. |
| 45 | Mueller & Macdonald, Semantic nDCG for ColBERT | A dense-ranker explanation metric, tangential to the pairwise-LLM focus. |
| 46 | Thakur et al., Touché 2020 / BEIR | Dataset context for argument retrieval rather than a framework claim; the argument-axiom point is carried by [12]. |
| 47 | Reusch & Belinkov, Reverse-Engineering GenIR | Generative retrieval of the DSI kind, a different architecture from LLM re-rankers. |
| 48 | Wallat et al., Causal Probing for Dual Encoders | A method for dense bi-encoders that is not part of the chosen pipeline. |
| 49 | Joho & Jose, Instruction-Response Perspective | General context on the transparency of LLM-based IR; no specific claim used. |
| 50 | ELSPR | A secondary non-transitivity cure; the point is carried by [3] and [4]. |
| 51 | TrustJudge | A secondary account of LLM-as-judge inconsistency, redundant with [3]. |
| 52 | Anand et al., Explainable IR (book) | An umbrella reference; the survey [33] serves the same purpose inline. |

The one untitled item in the collection is omitted entirely, as it carries no metadata, and
the duplicate Völske 2021 entry is merged into [15].

## References

Numbered in order of first appearance. Entries [1] to [42] are cited above; [43] to [52]
are in the collection for context but not cited, as explained in Section 4. The metadata is
best-effort and should be verified before use in the thesis proper.

1. Sun, W., Yan, L., Ma, X., et al. (2023). *Is ChatGPT Good at Search? Investigating LLMs as Re-Ranking Agents.* EMNLP. doi:10.18653/v1/2023.emnlp-main.923.
2. Qin, Z., Jagerman, R., Hui, K., et al. (2024). *Large Language Models are Effective Text Rankers with Pairwise Ranking Prompting.* NAACL Findings. doi:10.18653/v1/2024.findings-naacl.97.
3. *Investigating Non-Transitivity in LLM-as-a-Judge.* (2025). arXiv:2502.14074.
4. Zeng, Y., Tendolkar, O., Baartmans, R., et al. (2024). *LLM-RankFusion: Mitigating Intrinsic Inconsistency in LLM-based Ranking.* arXiv:2406.00231.
5. Liu, Q., Duan, H., Mao, J., Wen, J.-R. (2025). *How Do Large Language Models Understand Relevance? A Mechanistic Interpretability Perspective.* ACM TOIS. arXiv:2504.07898.
6. Chowdhury, T., Nijasure, A., Allan, J. (2025). *Probing Ranking LLMs: A Mechanistic Analysis for Information Retrieval.* ICTIR. doi:10.1145/3731120.3744603.
7. Fang, H., Tao, T., Zhai, C. (2004). *A Formal Study of Information Retrieval Heuristics.* SIGIR.
8. Fang, H., Zhai, C. (2005). *An Exploration of Axiomatic Approaches to Information Retrieval.* SIGIR.
9. Amigó, E., Fang, H., Mizzaro, S., Zhai, C. (2017). *Report on the SIGIR 2017 Workshop on Axiomatic Thinking for IR (ATIR).* ACM SIGIR Forum.
10. Fang, H., Zhai, C. (2006). *Semantic Term Matching in Axiomatic Approaches to Information Retrieval.* SIGIR. doi:10.1145/1148170.1148193.
11. Yang, P., Lin, J. (2019). *Reproducing and Generalizing Semantic Term Matching in Axiomatic Information Retrieval.* ECIR. doi:10.1007/978-3-030-15712-8_24.
12. Heinrich, M., Vogel, M., Bondarenko, A., Hagen, M. (2025). *Axiomatic Re-Ranking for Argument Retrieval.*
13. Reimer, J.H., Fröbe, M., Stein, B., Potthast, M. (2025). *Axioms for Retrieval-Augmented Generation.*
14. Rosset, C., Mitra, B., Xiong, C., et al. (2019). *An Axiomatic Approach to Regularizing Neural Ranking Models.* SIGIR.
15. Völske, M., Bondarenko, A., Fröbe, M., et al. (2021). *Towards Axiomatic Explanations for Neural Ranking Models.* ICTIR.
16. Câmara, A., Hauff, C. (2020). *Diagnosing BERT with Retrieval Heuristics.* ECIR.
17. MacAvaney, S., Feldman, S., Goharian, N., Downey, D., Cohan, A. (2022). *ABNIRML: Analyzing the Behavior of Neural IR Models.* TACL. doi:10.1162/tacl_a_00457.
18. Hagen, M., Völske, M., Göring, S., Stein, B. (2016). *Axiomatic Result Re-Ranking.* CIKM.
19. Bondarenko, A., Fröbe, M., Reimer, J.H., et al. (2022). *Axiomatic Retrieval Experimentation with ir_axioms.* SIGIR (demo).
20. Lu, C., Chen, C., Eickhoff, C. (2025). *Cross-Encoder Rediscovers a Semantic Variant of BM25.* EMNLP.
21. Chen, C., Merullo, J., Eickhoff, C. (2025). *Axiomatic Causal Interventions for Reverse Engineering Relevance Computation in Neural Retrieval Models.* SIGIR.
22. Zhuang, S., Zhuang, H., Koopman, B., Zuccon, G. (2024). *A Setwise Approach for Effective and Highly Efficient Zero-shot Ranking with LLMs.* SIGIR. doi:10.1145/3626772.3657813.
23. Merullo, J., Eickhoff, C., Pavlick, E. (2024). *Language Models Implement Simple Word2Vec-style Vector Arithmetic.* NAACL.
24. Peng, Z., Wei, T., Song, T., Zhao, Y. (2025). *Efficiency-Effectiveness Reranking FLOPs for LLM-based Rerankers.* EMNLP Industry. arXiv:2507.06223.
25. Sun, W., Chen, Z., Ma, X., et al. (2023). *Instruction Distillation Makes LLMs Efficient Zero-shot Rankers.* arXiv:2311.01555.
26. Dedov, E. (2025). *JointRank: Rank Large Set with Single Pass.* ICTIR. doi:10.1145/3731120.3744587.
27. Oosterhuis, H., Jagerman, R., Qin, Z., Wang, X. (2025). *Optimizing Compound Retrieval Systems.* SIGIR.
28. Bradley, R.A., Terry, M.E. (1952). *Rank Analysis of Incomplete Block Designs: I. The Method of Paired Comparisons.* Biometrika. doi:10.2307/2334029.
29. Negahban, S., Oh, S., Shah, D. (2017). *Rank Centrality: Ranking from Pairwise Comparisons.* Operations Research. doi:10.1287/opre.2016.1534.
30. Cao, Z., Qin, T., Liu, T.-Y., Tsai, M.-F., Li, H. (2007). *Learning to Rank: From Pairwise Approach to Listwise Approach.* ICML. doi:10.1145/1273496.1273513.
31. Liu, T.-Y. (2009). *Learning to Rank for Information Retrieval.* Foundations and Trends in IR. doi:10.1561/1500000016.
32. Usuha, K., Kato, M.P., Fujita, S. (2024). *Over-penalization for Extra Information in Neural IR Models.* CIKM.
33. Anand, A., Lyu, L., Idahl, M., Wang, Y., Wallat, J., Zhang, Z. (2022). *Explainable Information Retrieval: A Survey.* arXiv:2211.02405.
34. Heuss, M., de Rijke, M., Anand, A. (2025). *RankingSHAP: Faithful Listwise Feature Attribution Explanations for Ranking Models.*
35. Yu, P., Rahimi, R., Allan, J. (2022). *Towards Explainable Search Results: A Listwise Explanation Generator (LiEGe).* SIGIR.
36. Saha, S., Agarwal, H., Venktesh, V., Anand, A., et al. (2024). *ir_explain: A Python Library of Explainable IR Methods.* arXiv:2404.18546.
37. Parry, A., Chen, C., Eickhoff, C., MacAvaney, S. (2025). *MechIR: A Mechanistic Interpretability Framework for Information Retrieval.* ECIR. doi:10.1007/978-3-031-88720-8_16.
38. Polyakov, G., Chen, C., Eickhoff, C. (2025). *Towards Best Practices of Axiomatic Activation Patching in Information Retrieval.* SIGIR. doi:10.1145/3726302.3730256.
39. Chen, C., Heuss, M., Eickhoff, C. (2026). *Tutorial on Mechanistic Interpretability.*
40. Wang, Y., Qi, J., Chen, C., Eustratiadis, P., Verberne, S. (2025). *How Role-Play Shapes Relevance Judgment in Zero-Shot LLM Rankers.* arXiv:2510.17535.
41. Pradeep, R., Sharifymoghaddam, S., Lin, J. (2023). *RankVicuna: Zero-Shot Listwise Document Reranking with Open-Source LLMs.* arXiv:2309.15088.
42. Pradeep, R., Sharifymoghaddam, S., Lin, J. (2023). *RankZephyr: Effective and Robust Zero-Shot Listwise Reranking is a Breeze!* arXiv:2312.02724.
43. *Interpretability in Neural Information Retrieval* (2025). MSc/PhD thesis.
44. Chen, C. (2025). *catherineschen/axiomatic-ir-interventions* (software).
45. Mueller, A., Macdonald, C. (2025). *Semantically Proportioned nDCG for Explaining ColBERT's Learning Process.* ECIR. doi:10.1007/978-3-031-88708-6_22.
46. Thakur, N., Bonifacio, L., Fröbe, M., et al. (2024). *Systematic Evaluation of Neural Retrieval Models on the Touché 2020 Argument Retrieval Subset of BEIR.* SIGIR. doi:10.1145/3626772.3657861.
47. Reusch, A., Belinkov, Y. (2025). *Reverse-Engineering the Retrieval Process in GenIR Models.* doi:10.1145/3726302.3730076.
48. Wallat, J., Hinrichs, H., Anand, A. (2024). *Causal Probing for Dual Encoders.* CIKM.
49. Joho, H., Jose, J.M. (2025). *An Instruction-Response Perspective on LLMs in IR Tasks.*
50. Yu, Y., Liu, Y., He, M., et al. (2026). *ELSPR: Evaluator LLM Self-Purification on Non-Transitive Preferences via Tournament Graph Reconstruction.* AAAI. arXiv:2505.17691.
51. Wang, Y., Song, Y., Zhu, T., et al. (2025). *TrustJudge: Inconsistencies of LLM-as-a-Judge and How to Alleviate Them.* arXiv:2509.21117.
52. Anand, A., Saha, S., Venktesh, V. (2025). *Explainable Information Retrieval* (book). Springer.
