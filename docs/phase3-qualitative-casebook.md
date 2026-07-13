# Phase 3 Qualitative Casebook — Relevance-Improving Pair Reversals

> **Status:** development-only hypothesis generation, recorded 2026-07-13. These cases come
> from DL19/DL20, which already informed the project, and cannot confirm a new axiom. The
> machine-readable annotations are in
> [`resources/phase3-qualitative-case-annotations.csv`](resources/phase3-qualitative-case-annotations.csv).

## 1. Purpose

This case study asks what textual distinctions appear in pairwise preferences that contribute
to better Qwen rankings but are often missed by the fitted classical axiom model. Its purpose is
to propose testable Phase 3 candidates, identify counterexamples early, and preserve qualitative
evidence rather than relying only on aggregate residual statistics.

The analysis does **not** infer why the LLM made a choice internally. It describes observable
differences between two passages and records candidate retrieval constraints that could reproduce
the useful preference.

## 2. Reproducible selection

`experiments/rq4_qualitative/run.py` joins the cached Phase 3 pair predictions, complete runs,
BM25 pools, axiom preferences and qrels. A pair enters the candidate pool when:

1. the query-level Qwen run improves nDCG@10 over BM25;
2. Qwen's order-collapsed pair label is decisive;
3. Qwen prefers a passage with a higher qrel grade;
4. BM25 originally ranked that passage below the lower-grade passage; and
5. the final Qwen Copeland ranking places it above the lower-grade passage.

This produced **583 candidate reversals**. A pair satisfying these conditions is called a
*contributory reversal*. It is not a causal attribution: Copeland positions depend on the entire
pair graph, and removing one edge need not undo the query-level nDCG improvement.

The pool contains 261 DL19 and 322 DL20 edges across 71 distinct queries. The fitted classical
model supports Qwen's relevance-improving direction on 309 edges and opposes it on 274; all three
tested LLM systems support the same direction on 373 edges. These are descriptions of the
outcome-conditioned pool, not population rates or inferential tests.

The thirteen cases below are a purposive maximum-variation sample, not a prevalence estimate.
They cover both query sets, multiple information-need types, cross-model consensus and
disagreement, and cases where the fitted classical model succeeds or fails. The primary analyst
read the query and both cached passages without changing qrels or model outputs.

Generated inspection resources (gitignored result artifacts):

- `results/rq4_axioms/pooled_top10/qualitative/candidate_reversals.parquet` — all candidates,
  including full passage text;
- `candidate_reversals.csv` — text-free quantitative index;
- `inspection_packets.md` — automatically generated reading packets.

## 3. Cases

Ranks are zero-based. `W` is the higher-grade passage preferred by Qwen; `L` is the lower-grade
passage it displaced. Model support is `Q/F/XL`; `+` means that system preferred W, `0` a tie and
`−` a preference for L. “Classical” is the OOF fitted classical-axiom preference.

| ID | query | grades W>L | BM25 W/L → Qwen W/L | models | classical | observed distinction |
|---|---|---:|---|---|---:|---|
| C01 | do google docs auto save | 3>0 | 2/0 → 0/6 | +/+/+ | − | explicit answer to the autosave proposition vs instructions for manually saving another file |
| C02 | what is famvir prescribed for | 3>0 | 8/0 → 1/9 | +/+/+ | − | treatment indication vs pregnancy/breastfeeding caution that merely mentions the drug |
| C03 | is cdg airport in main paris | 2>0 | 8/0 → 1/8 | +/+/+ | − | geographic relation “northeast of Paris” vs airport-shuttle advertising |
| C04 | why did the us voluntarily enter ww1 | 2>0 | 8/0 → 0/2 | +/+/+ | − | explicit causal factors vs a timeline correction and unsupported counterfactual claim |
| C05 | who sings monk theme song | 3>0 | 7/0 → 1/3 | +/+/+ | + | named song and singer vs search-term boilerplate about a different programme |
| C06 | how much money do motivational speakers make | 3>0 | 8/0 → 0/1 | +/+/+ | − | numerical answer with unit, period and qualifications vs a generic discussion of income sources |
| C07 | define bmt medical | 3>0 | 6/0 → 0/5 | +/+/+ | + | acronym expansion plus definition vs a citation to an acronym website |
| C08 | what type of tissue are bronchioles | 3>0 | 9/0 → 4/8 | +/+/0 | + | explicit tissue class and anatomical relation vs a disease that affects bronchioles |
| C09 | what metal are hip replacements made of | 3>1 | 7/0 → 3/4 | +/0/+ | + | enumerated materials and device components vs the generic statement “made of metal” |
| C10 | what amino produces carnitine | 3>0 | 8/4 → 0/9 | +/+/+ | − | lysine/methionine biosynthetic relation vs an incidental statement about carnitine and body odour |
| C11 | where is the show shameless filmed | 3>0 | 6/1 → 0/5 | +/+/+ | − | filming location while distinguishing setting from location vs a title-disambiguation list |
| C12 | what conflict does Della face in The Gift of the Magi | 2>0 | 5/1 → 1/6 | +/+/+ | − | character-specific action and conflict discussion vs a broad comparison with another story |
| C13 | workplace discrimination in Oklahoma City | 2>1 | 8/0 → 0/3 | +/+/+ | − | jurisdiction-specific action and complaint route vs a navigation list of general legal topics |

Short source fragments supporting these readings include “automatically save your updates,”
“Famvir” among drugs used to treat genital herpes, CDG “northeast of Paris,” “two main reasons,”
“by Randy Newman,” “around $88,000 per year,” “BMT: Bone marrow transplantation,” “simple
cuboidal epithelium,” and a list of stainless steel, titanium, chromium and cobalt. These fragments
show answer-bearing relations; term overlap alone does not distinguish them from the demoted
passages.

## 4. Cross-case findings

### 4.1 Query–answer relation alignment is stronger than topical similarity

The clearest repeated distinction is not whether both passages discuss the query topic. It is
whether a sentence instantiates the **relation requested by the query**:

- feature → behaviour (`Google Docs` → autosaves);
- drug → indication (`Famvir` → treats genital herpes);
- entity → location (`CDG` → northeast of Paris; *Shameless* → filmed in Los Angeles);
- process → cause (US entry → submarine warfare and the Zimmermann note);
- acronym → expansion (`BMT` → bone marrow transplantation);
- substance → constituents (hip implant → named metals; carnitine → precursor amino acids).

This explains why generic semantic similarity may remain too blunt. Both passages can be highly
similar to the query while only one satisfies its predicate.

**Candidate QARA — Query–Answer Relation Alignment.** Classify the query's requested relation or
answer type, then compare each passage's strongest sentence-level evidence for that relation. A
Tier-A version could combine a frozen query-intent classifier, typed entities/numerals and a
frozen sentence-pair model. It returns a preference only when the evidence difference exceeds a
fixed margin. This extends Semantic Intent Coverage: it asks whether the requested relation is
asserted, not merely whether query meaning appears.

### 4.2 Answer-bearing content must be separated from boilerplate and incidental mention

Several lower-grade passages are search-term lists, navigation pages, citations, advertising, or
generic descriptions. They contain strong lexical matches without delivering an answer. This is
visible in C03, C05, C07, C11 and C13.

**Candidate CBP — Content-Bearing Passage preference.** Prefer passages containing a complete
declarative answer sentence over passages dominated by navigation fragments, repeated search
terms, citation metadata or calls to action. A cheap proxy can use sentence completeness,
punctuation, link/list density, duplicate n-grams and the proportion of sentence-like content.
This should be tested separately from answer localisation: a complete answer may occur late.

### 4.3 Typed specificity helps only when conditioned on the question

C06 and C09 reward numbers, units and enumerated materials; C07 rewards an acronym expansion;
C08 rewards a tissue class. Raw numeral/entity density would also reward irrelevant passages.

**Candidate TSC — Typed Specificity and Completeness.** First infer the expected answer type
(quantity, person, location, material, biological class, definition); then prefer passages that
contain compatible values and, for plural/list questions, broader compatible enumeration. This
is a question-conditioned refinement of the existing Specificity and Aspect Completeness menu.

### 4.4 Qualifiers and roles should be represented explicitly

The useful passages often satisfy constraints that are easy to lose in bag-of-terms matching:
*filmed* rather than *set*, *medical* BMT, Oklahoma jurisdiction, Della's conflict rather than the
story generally, and tissue *lining* bronchioles rather than a disease *of* bronchioles.

**Candidate QCS — Query-Constraint Satisfaction.** Extract named-entity, geographic, temporal,
role and relation qualifiers from the query and require evidence that binds them in one local
span or proposition. This differs from QCOV: all query terms can occur without the requested
roles being connected correctly.

### 4.5 Causal adequacy is promising but narrow

C04 supports the Phase 3 causal/explanatory candidate: the preferred passage names causes and
connects them to the event, whereas the alternative mostly disputes one proposed cause. C12 also
shows that explanation of a narrative conflict requires character/action relations rather than
topic overlap. These cases motivate a query-type-specific axiom, not a universal causal-marker
bonus.

## 5. Counterexamples and risks

Two additional inspected cases constrain the candidate definitions:

- **“Who is Robert Gray?”** Qwen preferred the qrel-relevant explorer over a Mississippi
  politician, while both FLAN systems preferred the politician. The query itself provides no
  disambiguating context. This is not safe evidence for an entity-selection axiom; it may reflect
  collection intent or model priors unavailable from the query.
- **Wisdom-tooth removal duration.** Qwen promoted a passage with explicit durations, but both
  FLAN systems failed to support it and the passage is informal and internally variable. An
  answer-shaped number is not necessarily reliable evidence.

Further validity limits:

- selection conditions on positive qrel outcomes and therefore cannot estimate how often each
  pattern helps or harms;
- MS MARCO qrels are incomplete, and grade zero includes unjudged as well as judged-irrelevant
  passages in this analysis;
- one analyst produced the labels without a blinded second coder;
- passages are truncated by the ranker protocol, and their source quality is not independently
  verified;
- the same development data generated and illustrated the hypotheses;
- high cross-model agreement is robustness evidence, not an isolated architecture effect.

## 6. Phase 3 action

The complete preconditions, neutral cases, preference rules, cost levels and implementation order
are specified in
[`phase3-candidate-axiom-specs.md`](phase3-candidate-axiom-specs.md), with a machine-readable
registry in [`resources/phase3-candidate-registry.yaml`](resources/phase3-candidate-registry.yaml).

The first programmatic increment implemented the narrow D0 DEFANS, NUMANS, COMPARE and CBP
operationalisations, corrected across two logged revisions (v1 count binding and person abstention;
v2 local count binding and a real-list boilerplate fix). NUMANS survived as a coherent but very
sparse development lead. COMPARE was directionally coherent but extremely sparse; CBP is target- and
collection-inconsistent (it helps Qwen but hurts both FLAN targets, and its DL20 qrel/LLM agreement
is at chance); Qwen DEFANS is unstable. Thus the casebook patterns can be expressed as preconditioned
axioms, but the current implementations do not yet constitute a validated battery: NUMANS is retained
and COMPARE deferred to D1 as sparse probes, while DEFANS and CBP are rejected in their current D0
form. Exact results and the v0→v1→v2 ledger are in `phase3-candidate-axiom-specs.md` §7.

Continue with harmful-reversal analysis and the D1 relation variants, followed by TSC/QCS as
explicit ablations. For each candidate:

1. write the mathematical definition, precondition, neutral case and fixed margin before running;
2. add synthetic positive, negative and adversarial tests based on the distinctions above without
   copying the evaluated passages into training rules;
3. record every revision in the candidate ledger;
4. evaluate pairwise fidelity and fitted reranking effectiveness separately;
5. inspect both helpful and harmful reversals, not only positive-qrel cases; and
6. freeze retained definitions before external confirmation.

The present casebook supports hypothesis generation. It does not upgrade any candidate to a
validated axiom.
