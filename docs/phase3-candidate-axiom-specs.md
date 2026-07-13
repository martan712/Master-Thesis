# Phase 3 Candidate Axioms — Preconditions and Implementation Plan

> **Status:** development specification and D0-v2 record, 2026-07-13. These candidates were derived from
> [`phase3-qualitative-casebook.md`](phase3-qualitative-casebook.md) and are therefore
> retrospective hypotheses. They must be evaluated on development data and frozen before the
> locked confirmation set is opened.

## 1. Design principle

The qualitative cases do not justify a universal “semantic quality” score. They motivate a family
of narrow conditional constraints. Every candidate has the same pairwise form:

\[
A(q,d_1,d_2)=
\begin{cases}
\operatorname{sign}(s_A(q,d_1)-s_A(q,d_2)), & P_A(q,d_1,d_2) \\ 
0, & \text{otherwise,}
\end{cases}
\]

where `P_A` is an explicit precondition and `s_A` is a deterministic evidence score whenever a
rule-based implementation is possible. A score difference below the candidate's fixed margin is
neutral. The downstream fitted axiom model learns whether the direction deserves positive,
negative or zero weight; the axiom definition is not flipped after seeing results.

### 1.1 Shared topical-comparability precondition

For most answer-quality candidates, first extract query **anchor terms** by removing
interrogatives, auxiliaries, stopwords and the relation cue itself. Let
`anchor_cov(q,d)` be distinct anchor-term coverage.

The strict v0 precondition is:

- the query parser identifies exactly one supported intent;
- each document covers at least 50% of the anchor terms;
- the absolute anchor-coverage difference is at most 0.25; and
- the candidate-specific evidence difference reaches its stated margin.

This “other things approximately equal” condition prevents a new axiom from merely reproducing
QCOV or BM25. The values 0.50 and 0.25 are engineering starting points, not literature constants.
They may receive at most two logged development revisions and are then frozen.

### 1.2 Implemented subset

The first programmatic increment implements only the following narrow D0 operationalisations.
The aliases are versioned because they are not the complete D0+D1 concepts described below.

| alias | implemented query/evidence scope | deliberately absent |
|---|---|---|
| `DEFANS_d0v2` | declared definition forms; colon, dash, copula and explicit definition relations | sense resolution, acronym-initial validation, D1 apposition/dependencies |
| `NUMANS_d0v2` | money, count, duration and age patterns; counts bind to the requested noun within a local token window | factual verification and temporal calculation |
| `COMPARE_d0v2` | `difference between X and Y`; lexical two-sentence contrast window | `versus`/`compare` forms and implicit semantic comparison |
| `CBP_d0v2` | URLs, templates, repeated fragments, true numbered lists (list punctuation required) and finite-verb cues | D1 parsing and a positive semantic answer-quality score |

`CBP_d0v2` abstains on explicit list requests and bare `who is NAME` queries. The latter composes
the AMBIG safety condition with CBP; it does not attempt entity disambiguation.

## 2. Cost levels

| level | implementation | dependency/cost | role |
|---|---|---|---|
| D0 | regexes, token sets, cue lexicons | existing Python/tokenizer; cheapest | preferred first implementation |
| D1 | POS, NER and dependency patterns | existing `en_core_web_sm`; deterministic CPU | use when roles/types matter |
| D2 | frozen sentence encoder or NLI model | new pinned model; cached CPU inference | only if D0/D1 coverage is inadequate |
| D3 | LLM criterion judge or external verification | expensive and potentially circular | diagnostic ceiling only, never the cheap final axiom |

`en_core_web_sm` has no useful dense vectors. It can provide tokens, lemmas, entities and
dependencies at D1, but genuinely paraphrastic relation recognition requires D2 or D3.

## 3. Candidate list

### 3.1 DEFANS — Definition and acronym-expansion answer

**Motivation:** BMT definition versus acronym-site citation; answer-bearing definitions versus
pages that only mention the term.

**Query precondition**

- query matches `define X`, `definition/meaning of X`, `what does X mean`, or `what does X stand
  for`;
- exclude `who`, `where`, quantity, purpose and comparison queries;
- both passages satisfy topical comparability for `X`.

**Evidence and preference**

- score a sentence when it binds `X` to an expansion or description through `X is/are ...`,
  `X means/refers to/stands for ...`, `X: ...`, an apposition, or a copular dependency;
- give one additional point when the definiens contains at least one content noun not copied from
  the query;
- prefer the document whose best sentence score exceeds the other by at least one.

**Neutral cases:** unsupported query form, missing subject anchor, both/neither define the term, or
equal evidence.

**Implementation:** D0 regex patterns plus D1 copula/apposition dependencies. Cheap and
deterministic. Acronym expansion can additionally check whether expansion initials match the
query acronym.

**Risk:** domain-ambiguous acronyms. A definition can be fluent but select the wrong sense; query
domain qualifiers must be included in the precondition.

### 3.2 NUMANS — Compatible numeric answer

**Motivation:** motivational-speaker income and duration questions.

**Query precondition**

- query begins with `how much`, `how many`, `how long`, or `how old`;
- infer exactly one expected unit family: money, count, duration or age;
- both passages satisfy topical comparability.

**Evidence and preference**

- detect a number and a compatible unit in the same sentence as at least one anchor term;
- money requires currency/salary context; duration requires time units; age requires explicit age
  wording; count accepts a cardinal tied to the requested noun;
- score compatible spans, capped at two so long numeric lists do not win automatically;
- prefer on a one-point evidence margin.

**Neutral cases:** bare numbers, incompatible units, both documents provide compatible answers,
or an indirect birth date requiring calculation.

**Implementation:** D0 number/unit regexes, optionally D1 `MONEY`, `DATE`, `TIME`, `QUANTITY` and
`CARDINAL` entities. No new model required.

**Risk:** answer-shaped numbers can be outdated or false. This axiom measures answer compatibility,
not factual correctness. Temporal arithmetic and verification require D2/D3 and should not be
smuggled into the cheap rule.

### 3.3 PURPOSE — Purpose, use and treatment-indication relation

**Motivation:** Famvir treatment indication versus pregnancy caution.

**Query precondition**

- query contains `used for`, `prescribed for`, `indicated for`, `treat`, `prevent`, or `what does
  X do`;
- extract the subject entity `X`; both passages mention it and satisfy topical comparability.

**Evidence and preference**

- score a local proposition connecting `X` to a purpose/condition through `used to/for`,
  `prescribed/indicated for`, `treats`, `prevents`, `manages` or a passive equivalent;
- the relation and its object must occur in one sentence or dependency clause;
- prefer the higher proposition count, capped at two, with a one-point margin.

**Neutral cases:** entity mention without a purpose relation, safety/dosage-only passages, both
documents asserting a purpose, or an unsupported paraphrase.

**Implementation:** D0 cue lexicon plus D1 dependency direction and negation checks. Still cheap,
but high precision will require careful passive-voice tests.

**Risk:** medical factuality cannot be established from syntax. D2 NLI may improve paraphrase
coverage; D3 verification is diagnostic only.

### 3.4 LOCREL — Requested location relation

**Motivation:** CDG relative to Paris and *Shameless* filming location rather than setting or title
disambiguation.

**Query precondition**

- query is a `where` question or contains a supported role such as `located`, `filmed`, `set`,
  `born`, or `based`;
- extract the subject entity and requested location role;
- both passages mention the subject.

**Evidence and preference**

- require a `GPE`, `LOC` or `FAC` answer connected to the subject through the requested predicate
  in one clause;
- do not treat `set in` as evidence for `filmed in`, or airport transport to Paris as evidence
  that an airport is inside Paris;
- prefer the document with a supported role-location proposition on a one-point margin.

**Neutral cases:** under-specified entity, location mention without the requested role, both
documents supplying locations, or NER/parser uncertainty.

**Implementation:** D1 NER and dependency/cue patterns; deterministic and already supported by
the installed spaCy model. D0 alone is unsafe because the predicate role is essential.

**Risk:** paraphrases such as `shot on location` require a larger cue inventory or D2 relation
entailment.

### 3.5 COMPOSE — Composition, constituent and precursor relation

**Motivation:** named hip-replacement metals and carnitine precursors versus generic/incidental
mentions.

**Query precondition**

- query contains a supported relation such as `made of`, `composed of`, `contains`, `produces`,
  `derived from`, or `synthesized from`;
- extract subject and relation direction; both passages mention the subject.

**Evidence and preference**

- score propositions connecting the subject to noun-phrase objects through the requested relation
  or a declared equivalent;
- preserve direction: `X produces Y` is not the same as `X is produced from Y`;
- for plural/material queries, count distinct compatible objects, capped at four;
- prefer on a one-object margin.

**Neutral cases:** generic `made of metal` versus a named list can be compared only when the query
asks which material; incidental object mentions and reversed relations are neutral.

**Implementation:** D0 relation-phrase lexicon plus D1 dependency direction and coordinated noun
phrases. A small material/biochemical lexicon can raise precision but narrows domain coverage.

**Risk:** synonymy and scientific relation direction are hard. D2 NLI/embedding relation matching
may be required for acceptable coverage.

### 3.6 COMPARE — Explicit comparison completeness

**Motivation:** company strategy versus business model and hotel-versus-motel cases.

**Query precondition**

- query matches `difference between X and Y`, `X versus Y`, or `compare X and Y`;
- both comparison entities can be extracted and occur in both passages.

**Evidence and preference**

- require both entities in one sentence or adjacent sentence pair;
- score contrast predicates/cues such as `differs`, `whereas`, `while`, `unlike`, `but`,
  `compared with`, and paired copular clauses;
- add one point when the passage supplies attributes for both sides, capped at two;
- prefer with a one-point margin.

**Neutral cases:** headings or meta-descriptions promising a comparison, only one side described,
or equal comparison evidence.

**Implementation:** D0 entity/cue windows plus D1 dependency and sentence-boundary checks. Cheap
and deterministic.

**Risk:** implicit comparisons without cue words need D2 semantic relation modelling.

### 3.7 CAUSE — Causal/explanatory adequacy

**Motivation:** reasons for US entry into WWI versus timeline discussion.

**Query precondition**

- query begins with `why` or explicitly asks for a cause/reason;
- extract event anchors; both passages satisfy topical comparability.

**Evidence and preference**

- score a proposition linking event anchors to a cause through `because`, `due to`, `caused by`,
  `reason`, `led to`, `resulted from`, or a causal dependency verb;
- count distinct causal propositions, capped at three;
- prefer on a one-proposition margin.

**Neutral cases:** chronology without causation, merely denying a cause, generic causal markers not
linked to the event, or equal evidence.

**Implementation:** D0 cue patterns plus D1 dependency/negation checks. This provides a
high-precision, low-coverage first version.

**Risk:** causal adequacy and truth are genuinely semantic. D2 NLI or semantic-role extraction is
likely needed if the deterministic version is too sparse; D3 can measure a diagnostic ceiling.

### 3.8 QCS — Query-constraint satisfaction

**Motivation:** Oklahoma jurisdiction, Della's role in the conflict, `medical` BMT, and the
filmed-versus-set distinction.

**Query precondition**

- the query contains at least one extractable qualifier: named entity, location, time, domain
  modifier, requested role or relation predicate;
- the base subject is present in both passages.

**Evidence and preference**

- score the number of query qualifiers bound to the subject/relation in one sentence or dependency
  neighbourhood, not their document-wide occurrence;
- exact lexical qualifiers count only when their grammatical/relational role is preserved;
- prefer on a one-qualifier margin.

**Neutral cases:** no qualifier, qualifiers scattered across unrelated sentences, under-specified
entity queries, or equal binding.

**Implementation:** D1 NER, POS and dependency graphs with a fixed local-distance threshold. No
new model, but more engineering and more parser-error exposure than earlier candidates.

**Risk:** semantic roles cannot always be recovered from surface dependencies. D2 sentence-pair
entailment is the honest fallback.

### 3.9 CBP — Content-bearing passage over boilerplate

**Motivation:** answer prose versus search-term lists, citation metadata, title-disambiguation,
navigation and advertising.

**Pair precondition**

- both passages satisfy shared topical comparability;
- exclude list-seeking queries unless explicit boilerplate cues occur;
- at least one passage exceeds the fixed boilerplate threshold.

**Evidence and preference**

- deterministic penalties for URLs/citation templates, `incoming search terms`, navigation/link
  fragments, calls to click/contact, repeated fragments, high numbered-list density and very low
  finite-verb sentence density;
- deterministic credit for complete declarative sentences containing anchors;
- prefer the higher content-bearing score only when the gap is at least two rule points.

**Neutral cases:** two prose passages, two boilerplate passages, legitimate requested lists, or a
small score gap.

**Implementation:** D0 regex/duplication/list features plus D1 finite-verb and sentence-completeness
features. This is the cheapest broadly applicable candidate.

**Risk:** snippets can mix boilerplate with the exact answer, and terse definitions may look like
fragments. Candidate-specific answer evidence should override a generic style penalty only through
the downstream fitted model, not through hidden exceptions.

### 3.10 TSC — Typed specificity and completeness

**Motivation:** explicit money/unit answers, named materials, tissue class and acronym expansion.

**Query precondition**

- one expected answer type is identified: quantity/unit, person, location, material, biological
  class, definition or plural list;
- both passages are topical and at least one contains a compatible answer candidate.

**Evidence and preference**

- count distinct candidates compatible with the requested type in answer-bearing sentences;
- cap counts by type and require a one-candidate margin;
- generic entities/numerals of the wrong type do not score.

**Neutral cases:** ambiguous expected type, unsupported domain type, candidates only in navigation
or citations, or equal typed evidence.

**Implementation:** D0 unit/material cue lexicons and D1 NER/noun chunks. General material and
biomedical class typing may require a pinned ontology or D2 encoder; the cheap version must report
its deliberately narrow type inventory.

**Risk:** specificity is not factuality, and longer lists can contain more errors. Treat TSC as an
answer-shape axiom, not an evidence-quality axiom.

### 3.11 ALOC — Answer localisation

**Precondition:** both passages have evidence from the **same** typed detector above and their
evidence-strength difference is below that detector's margin.

**Preference:** prefer the passage whose first best-evidence sentence occurs earlier after
normalising position by document length; require a position gap of at least 0.20.

**Implementation:** D0 once typed detectors exist. It should not run on raw query-term matches,
because an early irrelevant mention is not an answer.

### 3.12 AMBIG — Ambiguity abstention flag, not a relevance axiom

**Motivation:** `who is Robert Gray` provides no sense-disambiguating context, and Qwen/FLAN chose
different people.

**Precondition:** entity-seeking query contains only a name plus generic `who/what is` language and
no profession, location, date, relation or other qualifier.

**Action:** force entity-selection and typed-relation axioms to remain neutral; optionally expose a
separate uncertainty/cascade feature. It must not choose the most famous or collection-preferred
sense.

**Implementation:** D0 query-only rule. Cheap, deterministic and important for preventing false
confidence.

## 4. What cannot honestly be done cheaply

- **Factual correctness or faithfulness:** syntax, answer type and directness cannot establish that
  a claim is true. This needs external evidence or a diagnostic judge.
- **Open-domain relation paraphrase:** a finite cue lexicon misses many valid formulations. D2 NLI
  or relation embeddings may be necessary after measuring D0/D1 coverage.
- **Unspecified entity intent:** no axiom can infer a missing sense from the query without relying
  on priors external to the stated information need. Abstention is the correct behaviour.
- **Temporal calculation:** deriving current age from a birth date needs a reference date and
  temporal reasoning. The deterministic NUMANS v0 remains neutral.

## 5. Implementation architecture

1. `src/axiomrank/axioms/answering.py` provides the shared immutable `QueryFrame`, query-side
   precondition inspection and the independent D0 axioms. A cached `DocumentEvidence`
   representation is deferred until D1 features would otherwise repeat parsing.
2. Implement one deterministic query parser producing a single intent, anchors, expected answer
   type, relation direction and qualifiers. Ambiguous parses return `UNKNOWN`.
3. Before D1, add a shared cached spaCy representation; do not run NER/dependency parsing once per
   axiom and pair.
4. Implement each candidate as an independent `ir_axioms.Axiom` returning `{-1,0,+1}`. QARA is a
   family label, not a monolithic feature.
5. Put margins and cue-set versions in config aliases and result manifests.
6. Add synthetic tests for positive direction, reversed document order, failed precondition,
   neutrality, negation, adversarial boilerplate and determinism.
7. Run add-one and leave-family-out development ablations. Report coverage before agreement,
   query-grouped OOF fidelity, fitted nDCG@10, harmful reversals and redundancy with classical
   axioms.

## 6. Implementation order

1. **D0 first (implemented as v1):** DEFANS, NUMANS, COMPARE, CBP and AMBIG query abstention.
2. **D1 next:** PURPOSE, LOCREL, CAUSE and the fuller DEFANS/COMPARE/CBP variants.
3. **Then:** COMPOSE, QCS, TSC and ALOC, reusing the typed evidence framework.
4. **D2 only after a measured need:** add one pinned frozen sentence/NLI model if deterministic
   coverage is too low or misses declared paraphrase strata.
5. **D3 diagnostic only:** measure factuality/semantic ceilings without entering the cheap final
   battery.

No candidate or threshold may be evaluated on the locked NFCorpus test collection during this
process.

## 7. Development evaluation and revision ledger

`experiments/rq4_candidates/run.py` recomputes candidate-only preferences from the cached
DL19/DL20 pools and pairs, while loading LLM preferences with `allow_new=False`. It never mutates
the Phase 1/2 axiom caches. Classical, four add-one and all-D0 logistic models share one
query-disjoint fold map, predict every top-ten pair, and are Copeland-aggregated before paired
qrel evaluation. The experiment is configured by `configs/rq4_candidates_d0.yaml`; generated
artifacts are under `results/rq4_candidates/`.

### 7.1 Revisions

| version | reason | change | disposition |
|---|---|---|---|
| D0-v0 | first executable specification | four deterministic rules and synthetic sign/neutrality tests | retained locally as the initial run |
| D0-v1 | v0 count evidence was not bound to the requested noun; CBP could act on the already-identified ambiguous `who is NAME` case | bind count numerals to the requested unit noun; compose list and unqualified-person abstention into CBP | superseded; artifacts retained under `results/rq4_candidates/d0v1/` |
| D0-v2 | independent review found two result-changing implementation flaws: CBP counted ordinary prose numbers (years, quantities) as numbered-list boilerplate because list punctuation was optional; NUMANS bound a count to the requested noun only at sentence granularity, so an unrelated number in the same sentence could score | require real list punctuation for the CBP list feature; bind counts within a local number–noun token window with deterministic singular/plural matching; add honest bootstrap-cluster suppression and an exact bidirectional candidate merge in the evaluator; add a frozen-coefficient feature-zero diagnostic | second and final logged revision |

The v1 correction is conceptual, not threshold tuning: a net-worth number is not evidence for
“how many sons,” and a context-free person name cannot safely select an entity sense. The v2
corrections are also conceptual, not threshold tuning: a prose year is not a list item, and a
number sharing a sentence with the requested noun is not the same as a number bound to it. These
are the two logged development revisions permitted before the thresholds freeze; the shared
precondition constants (0.50, 0.25) were not touched.

### 7.2 Coverage

Coverage is model-independent. “Eligible” is the query-side precondition; conditional coverage
is the active-pair fraction within eligible queries.

| collection | candidate | eligible/active queries | evaluable queries | all-pair coverage | eligible-pair coverage |
|---|---|---:|---:|---:|---:|
| DL19 | DEFANS | 2/1 | 1 | 0.005 | 0.100 |
| DL19 | NUMANS | 1/1 | 1 | 0.008 | 0.356 |
| DL19 | COMPARE | 2/2 | 2 | 0.011 | 0.233 |
| DL19 | CBP | 40/8 | 7 | 0.051 | 0.055 |
| DL20 | DEFANS | 6/6 | 6 | 0.040 | 0.363 |
| DL20 | NUMANS | 5/4 | 4 | 0.019 | 0.200 |
| DL20 | COMPARE | 2/1 | 1 | 0.011 | 0.289 |
| DL20 | CBP | 53/21 | 21 | 0.075 | 0.076 |

The single largest v2 effect is on CBP: requiring genuine list punctuation removed the spurious
prose-number activations that had inflated D0-v1 CBP coverage (DL19 29→8 active queries,
0.176→0.051 all-pair; DL20 40→21, 0.167→0.075). The corrected CBP is a much narrower, more honest
detector. DEFANS, NUMANS and COMPARE remain too sparse for standalone population claims; agreement
CIs are now suppressed to `NA` for any candidate below five evaluable queries, so high conditional
agreement on one or two queries is no longer reported as a 95% interval.

### 7.3 Fitted results

All-D0 minus classical OOF pair-accuracy lifts were −0.0028 for Qwen (CI [−0.0109, +0.0059]),
+0.0044 for FLAN-large (CI [−0.0035, +0.0137]) and −0.0026 for FLAN-XL (CI [−0.0105, +0.0046]);
every query-bootstrap interval included zero. Corresponding log-loss lifts were +0.0005, +0.0067
and +0.0008; only the FLAN-large log-loss interval ([+0.0013, +0.0132]) excluded zero, and it is a
marginal, unadjusted, retrospective development interval. Correcting the D0-v1 flaws therefore
removed the apparent all-D0 fidelity signal rather than strengthening it.

Fold coefficient directions after v2:

| candidate | Qwen | FLAN-large | FLAN-XL |
|---|---|---|---|
| NUMANS | +++++ (β≈+1.52) | +++++ (+1.21) | +++++ (+1.54) |
| COMPARE | +++++ (+0.79) | +++++ (+0.82) | +++++ (+0.85) |
| DEFANS | −−+−− (−0.24) | +++++ (+1.42) | +++++ (+0.42) |
| CBP | +++++ (+0.21) | −+−−− (−0.16) | −−−−− (−0.29) |

NUMANS and COMPARE keep a positive coefficient in every fold and for every target; NUMANS is the
strongest and most consistent, but both are far too sparse (1–4 evaluable queries) for a population
claim. DEFANS is target-dependent — negative and fold-unstable for Qwen, positive for FLAN. CBP is
target-inconsistent — positive for Qwen but negative and fold-unstable for both FLAN targets.

All-D0 minus classical nDCG@10 (paired; * = 95% interval excludes zero):

| target | DL19 | DL20 |
|---|---:|---:|
| Qwen | −0.0057 [−0.0138, +0.0002] | +0.0024 [−0.0027, +0.0083] |
| FLAN-large | +0.0002 [−0.0005, +0.0008] | +0.0044 [−0.0011, +0.0106] |
| FLAN-XL | −0.0058 [−0.0121, −0.0007] * | +0.0031 [−0.0028, +0.0094] |

Only the negative FLAN-XL DL19 delta now excludes zero; unlike D0-v1, none of the small positive
DL20 deltas are individually significant. The DL19-negative / DL20-positive pattern persists but is
weaker after removing the spurious CBP and NUMANS activations.

Absolute nDCG@10 keeps the effectiveness gap visible:

| target | collection | BM25 | LLM | classical | all D0 |
|---|---|---:|---:|---:|---:|
| Qwen | DL19 | 0.4795 | 0.5483 | 0.5028 | 0.4971 |
| Qwen | DL20 | 0.4936 | 0.5555 | 0.4845 | 0.4869 |
| FLAN-large | DL19 | 0.4795 | 0.5294 | 0.5045 | 0.5046 |
| FLAN-large | DL20 | 0.4936 | 0.5315 | 0.4900 | 0.4944 |
| FLAN-XL | DL19 | 0.4795 | 0.5262 | 0.5084 | 0.5026 |
| FLAN-XL | DL20 | 0.4936 | 0.5325 | 0.4908 | 0.4939 |

The all-D0 minus LLM nDCG@10 delta is large and negative for every target/collection cell (from
−0.024 to −0.069, every interval excluding zero): the fitted deterministic axioms remain far below
the LLM reranker they are meant to model.

**Feature-zero direct effect (frozen coefficients).** Because an add-one model refits the classical
weights, its query-level changes cannot be attributed to the candidate alone. Zeroing only the
candidate feature inside the fitted all-D0 union model (coefficients held fixed) isolates the direct
effect. Each candidate does directly flip pairs — e.g. for Qwen: DEFANS 27 pairs / 7 queries,
NUMANS 11/3, COMPARE 6/2, CBP 40/15 — so the near-zero net fidelity lift reflects roughly balanced
helpful and harmful direct reversals, not an inactive feature.

**CBP versus qrel grades (DL19 vs DL20).** On judged CBP-active pairs, the boilerplate direction
agrees with the higher qrel grade on DL19 (37 agree, 14 disagree, 35 tie) but is close to chance on
DL20 (49 agree, 37 disagree, 96 tie). This mirrors CBP's LLM agreement (DL19 0.74, DL20 0.56 at
chance) and confirms the DL19/DL20 split is a genuine property of the style signal, not a target
artefact.

**Disposition (candidate-by-candidate).** The four-rule D0 union is **not ready to freeze**.

- **NUMANS** — retain for D1 development: the only candidate with a consistent positive coefficient
  across all targets and every fold, but far too sparse (1–4 evaluable queries) to claim population
  effect. Needs broader typed-number coverage before any freeze.
- **COMPARE** — defer to D1: directionally consistent (positive every fold/target) but even sparser
  (1–2 queries). A probe, not evidence.
- **DEFANS** — reject in its current D0 form: negative and fold-unstable for Qwen, DL20 agreement at
  chance with more harmful than helpful activations. Requires sense-aware abstention (D1) before
  reconsideration.
- **CBP** — reject as a universal axiom: target-inconsistent (helps Qwen, hurts both FLAN targets)
  and collection-inconsistent (DL19 vs DL20 at chance) even after the precision correction. A style
  penalty is not a relevance signal.

No single positive collection or target rescues the union; it is not frozen.
