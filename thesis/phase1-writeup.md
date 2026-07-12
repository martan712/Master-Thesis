# Phase 1 — Corrected Lexical and WordNet-Semantic Baselines

*Detailed analysis record for RQ1–RQ2. The final manuscript consolidates it in
`foundation-writeup.md`; chronology and superseded interpretations are in
`docs/research-logbook.md`. Citation numbers refer to `docs/literature-overview.md`.*

## 1. Purpose

Phase 1 establishes the corrected classical axiomatic-IR baseline [7, 8, 19] that RQ4 must explain or improve. It
repeats measurement across DL19 and DL20, adds query-level uncertainty, broadens the lexical
battery, tests relaxed preconditions and a scoped WordNet semantic tier, and measures whether the
target LLM systems improve BM25 at the same rerank depth.

## 2. Experimental design

The grid crosses DL19/DL20, BM25 top-10 all-pairs/uniform depth-100 samples, and two ranker
systems: Qwen chat/log-probability v1 and FLAN-T5-large label-likelihood v0. Each pair is presented
in both orders. The top-10 condition is primary; uniform samples describe scope across rank gaps.

The query is the inferential unit. Agreement intervals resample queries. Joint prediction uses
query-grouped folds. Qwen and FLAN differ in prompt and scoring as well as model, so cross-system
similarity is robustness evidence, not a controlled architecture effect.

## 3. Batteries

The strict lexical battery contains TFC1, TFC3, M-TDC, LNC1, TF-LNC and corrected PROX1–PROX5.
AND, DIV and LB1 extend it. Relaxed LNC1, TF-LNC and M-TDC variants test whether strict natural-
passage preconditions suppress coverage. Relaxed M-TDC changes the original axiom's semantics and
is labelled an inspired variant.

The semantic tier contains STMC1, STMC2, REG and ANTI-REG, including the semantic term-matching
family [10, 11], with WordNet synonym-set similarity.
Dense embedding semantics was not evaluated.

## 4. Effectiveness reference

Pairwise LLM preferences are Copeland-aggregated over the same BM25 top ten, matching the
wins-minus-losses PRP-allpair aggregation family [2]; order-inconsistent
pairs contribute ties and BM25 breaks aggregate ties. Results use 10,000-draw paired query-
bootstrap intervals.

| query set | ranker system | BM25 nDCG@10 | LLM reranked | Δ [95% CI] | W/T/L |
|---|---|---:|---:|---|---|
| DL19 | Qwen | 0.480 | 0.548 | +0.069 [+0.048,+0.090] | 32/5/6 |
| DL20 | Qwen | 0.494 | 0.556 | +0.062 [+0.039,+0.085] | 41/4/9 |
| DL19 | FLAN-T5-large | 0.480 | 0.529 | +0.050 [+0.027,+0.073] | 30/5/8 |
| DL20 | FLAN-T5-large | 0.494 | 0.531 | +0.038 [+0.014,+0.061] | 37/4/13 |

MAP moves in the same direction. Aggregated preferences therefore add relevance signal at depth
ten. This does not imply that every axiom-model error is ranker skill. Published deeper-pool PRP
scores are not direct absolute anchors.

## 5. Corrected lexical profile

| axiom | coverage DL19 | coverage DL20 | Qwen DL19 | FLAN DL19 | Qwen DL20 | FLAN DL20 |
|---|---:|---:|---:|---:|---:|---:|
| TFC1 | 0.757 | 0.784 | 0.478 | 0.466 | 0.513 | 0.520 |
| PROX1 | 0.403 | 0.353 | 0.579 | 0.613 | 0.507 | 0.584 |
| PROX2 | 0.498 | 0.414 | 0.646 | 0.672 | 0.575 | 0.648 |
| PROX3 | 0.128 | 0.069 | 0.675 | 0.707 | 0.678 | 0.689 |
| PROX4 | 0.205 | 0.203 | 0.639 | 0.595 | 0.544 | 0.546 |
| PROX5 | 0.242 | 0.231 | 0.560 | 0.536 | 0.550 | 0.546 |
| AND | 0.214 | 0.202 | 0.780 | 0.773 | 0.798 | 0.788 |
| DIV | 0.934 | 0.967 | 0.522 | 0.511 | 0.457 | 0.443 |
| LB1 | 0.193 | 0.184 | 0.765 | 0.804 | 0.721 | 0.713 |

Agreement is conditional on axiom coverage and a decisive canonical model label; full tables also
report evaluable counts and query-bootstrap intervals. AND and LB1 are strongest in the primary
condition. TFC1 remains near chance despite broad coverage, and DIV is at or below chance despite
over 90% coverage.

Agreement-profile correlations are 0.79 for Qwen and 0.86 for FLAN across query sets, and 0.93
across systems on DL20. These support within-study stability only.

## 6. Relaxed preconditions

Relaxation increases coverage but does not preserve the pilot's apparent high M-TDC agreement.
M-TDC falls toward approximately 0.52–0.56 at broader margins; strict DL20 M-TDC is also
uncertain. LNC1 and TF-LNC relaxations are weak. TFC1 length relaxations are identical to strict
TFC1, while TFC3 remains effectively inactive. The result separates coverage failure from useful
predictive signal.

## 7. Joint prediction

The strict lexical core adds approximately 0–2 accuracy points over majority baselines. The full
lexical battery reaches approximately 0.57–0.64 accuracy and 0.63–0.67 AUC, gains of roughly 4–9
points depending on cell. This is the fitted classical baseline for Phase 2/RQ4, not a mechanistic
explained fraction.

## 8. WordNet semantic result

STMC1 is broadly non-neutral but remains around 0.52–0.55; the other WordNet semantic axioms have
intervals spanning chance on DL20. Adding WordNet columns lowers top-10 joint accuracy by about
0.7–1.6 points across the four cells. FastText was not run. The result is a null for this WordNet
operationalisation, not for semantics generally.

## 9. Rank-gap scope

Model decisiveness rises with BM25 rank gap, but most axiom agreement does not. TFC1 improves only
in the widest bins and DIV becomes more negatively aligned. The expected “axioms work on easy,
wide-gap pairs” gradient is unsupported. Uniform-depth results therefore map scope; they do not
validate a causal explanation of the top-10 baseline.

## 10. Handoff

Phase 1 fixes the classical fitted baseline and a scoped WordNet null. Phase 2 compactly measures
the baseline's pooled predictive limit and grades candidate leads. RQ4 remains the main
contribution and owns all new-axiom fitting, effectiveness, ablation and external confirmation.
