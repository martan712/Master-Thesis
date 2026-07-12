# Phase 1 Design — Corrected Lexical and Semantic Baselines (RQ1–RQ2)

Engineering details are in `phase1-implementation.md`; provenance and superseded interpretations
are in `research-logbook.md`.

## 1. Purpose

Phase 1 produces the corrected baseline battery and the depth-matched LLM effectiveness reference
that RQ4 must improve on or explain. It answers RQ1 and the tested portion of RQ2; it does not
discover new axioms.

## 2. Experimental grid

| dimension | levels |
|---|---|
| query set | TREC DL19 (43), TREC DL20 (54) |
| pair condition | BM25 top-10 all-pairs; 50 uniform pairs/query from depth 100 |
| ranker system | Qwen chat/log-probability v1; FLAN-T5-large label-likelihood v0 |

Order swap is mandatory. Top-10 is the primary conditional population. Uniform-depth pairs map
scope across rank gaps; they are not a pass/fail validity test. Qwen and FLAN differ in model,
prompt and scoring, so cross-system similarity is robustness evidence rather than an isolated
architecture comparison.

## 3. Axiom batteries

### 3.1 Lexical battery

- Strict pilot set: TFC1, TFC3, M-TDC, LNC1, TF-LNC, PROX1–PROX5.
- Additions: AND, DIV and LB1.
- Relaxed development variants: LNC1@tf{0.2,0.5}, TF-LNC@len{0.1,0.3} and
  M-TDC@mass{0.1,0.3}.

Relaxed M-TDC drops the original “exactly one differing query term” gate and is therefore an
inspired variant, not literature M-TDC. Degenerate variants are excluded from fitted models but
retained in the result record.

### 3.2 Semantic battery

STMC1, STMC2, REG and ANTI-REG are evaluated with the WordNet synonym-set similarity backend.
This operationalisation is intentionally scoped. Dense embedding semantics and ranker hidden
states are not tested in Phase 1; a WordNet null cannot be generalised to them.

## 4. Estimands and analyses

### 4.1 Per-axiom profiles

For each cell, report coverage, evaluable count and agreement with a 95% query-bootstrap interval.
The query is resampled, preserving within-query pair dependence.

### 4.2 Joint prediction

Fit L2 logistic regression under query-grouped folds. Report majority baseline, out-of-fold
accuracy, ROC-AUC and log loss. Compare strict lexical, full lexical and lexical+WordNet on
identical folds. Coefficients are descriptive unless stability is evaluated separately.

### 4.3 Rank-gap scope

Report model decisiveness and axiom agreement/prediction by BM25 rank-gap bins. No expected shape
is required for the pipeline to be considered valid.

### 4.4 Effectiveness reference

Aggregate the cached top-10 LLM pair preferences with Copeland wins-minus-losses; order-
inconsistent pairs contribute ties and BM25 breaks equal aggregate scores. Compare the reranked
top-ten block with BM25 using nDCG@10 (primary), MAP, paired query-bootstrap intervals and
win/tie/loss counts. This shows whether the target ranker adds aggregate relevance signal; it does
not label every unexplained decision as skill.

Absolute PRP numbers are comparable only when candidate depth, model, prompt, tie handling and
aggregation match.

## 5. Decision criteria

- A baseline axiom result is interpretable only with non-trivial coverage and query-level
  uncertainty.
- A relaxed variant enters Phase 2 only if it is non-degenerate; entry is not a claim of
  literature-equivalent semantics or individual significance.
- WordNet enters only as an ablation if it improves held-out prediction; otherwise the Phase 2
  headline remains lexical.
- DL19 and DL20 may be pooled for a headline diagnostic only with query-set-specific robustness
  results.

These are analysis choices, not a gate on whether RQ4 remains the main contribution.

## 6. Completed results

### 6.1 Effectiveness reference

| query set | ranker system | BM25 | LLM reranked | Δ nDCG@10 [95% CI] | W/T/L |
|---|---|---:|---:|---|---|
| DL19 | Qwen | 0.480 | 0.548 | +0.069 [+0.048,+0.090] | 32/5/6 |
| DL20 | Qwen | 0.494 | 0.556 | +0.062 [+0.039,+0.085] | 41/4/9 |
| DL19 | FLAN-T5-large | 0.480 | 0.529 | +0.050 [+0.027,+0.073] | 30/5/8 |
| DL20 | FLAN-T5-large | 0.494 | 0.531 | +0.038 [+0.014,+0.061] | 37/4/13 |

MAP moved in the same direction. The result establishes depth-10 aggregate effectiveness for the
tested systems. Later depth analysis shows why published depth-100 PRP values are not direct
anchors.

### 6.2 Corrected lexical profile

Across top-10 cells, AND (approximately 0.77–0.80 agreement) and LB1 (0.71–0.80) are strongest;
the proximity family is mostly above chance; TFC1 is approximately 0.47–0.52 despite broad
coverage; DIV is at or below chance with over 90% coverage. Profile correlations are 0.79/0.86
across query sets and 0.93 across the two systems on DL20. These quantify within-study stability,
not population-wide generality.

M-TDC's pilot agreement did not survive its tiny evaluable sample. Relaxation increased coverage
but reduced agreement toward chance. TFC3 remained effectively dead, and TFC1 length-margin
variants were identical to strict TFC1.

### 6.3 Joint lexical prediction

The strict core adds roughly 0–2 accuracy points over majority baselines. The full lexical battery
reaches approximately 0.57–0.64 accuracy (AUC 0.63–0.67), a 4–9 point gain depending on cell.
This is the classical fitted baseline, not an “explained fraction.”

### 6.4 WordNet semantic result

Every semantic axiom's DL20 interval spans chance, and lexical+WordNet lowers top-10 joint
accuracy by approximately 0.7–1.6 points across the four cells. FastText was not run. The result
is therefore a WordNet-specific null.

### 6.5 Rank-gap result

Model decisiveness increases with BM25 rank gap, but per-axiom agreement mostly does not; DIV
moves below chance at wider gaps. The hypothesised easy-pair agreement gradient is unsupported.
This narrows interpretation but does not invalidate the replicated top-10 baseline.

## 7. Handoff

Phase 2 fits the corrected existing battery on pooled stable/decisive top-10 pairs, reports its
incremental predictive limit compactly, and separates confirmed measurements from exploratory
candidate leads. RQ4 then owns all new-axiom fitting and reranking claims.
