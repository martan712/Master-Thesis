# Phase 3 — RQ4: an axiomatic framework for neural reranking

> The main contribution. Phase 2 decomposed a competent LLM pairwise ranker into a thin
> axiom-explained part (pseudo-R² 0.057–0.074) and a large, systematic, content-shaped residual,
> and pre-registered RQ4 as the primary act. **Increment 1** formalised the two lexical-adjacent
> residual seeds Phase 2 named — a **verbosity/length** cluster (`d_len`) and a **query-coverage**
> cluster (`d_qcov`), `phase2-writeup.md` §3.4–3.5 — as new retrieval axioms (VERB, QCOV) and ran
> the decomposition-capture test. That probe is *done* and its result is recorded in
> `phase3-implementation.md` §6–7 (from the runners' `results/` JSON): **QCOV validated, VERB a
> reported null; untuned axioms ≈ BM25, a fitted lexical surrogate still short of the LLM.** This
> document keeps
> Increment 1 as the first probe and lays out the **main RQ4 line it motivated** — a pivot from
> lexical-adjacent axioms to **LLM-aligned semantic preference axioms**, i.e. an axiomatic
> framework for *neural* reranking.

## 1. Why pivot — the residual is semantic, and the classical semantic tier was blunt

Three facts from Phases 1–2, stated precisely, force the direction of the main RQ4 line.

- **The classical LEXICAL axioms explain a thin-but-real slice.** Pooled CV accuracy is
  0.629–0.666 against a 0.559–0.587 base (`phase2-writeup.md` §3.1), pseudo-R² 0.057–0.074, and
  Increment 1 added a genuinely new lexical-adjacent axiom to that battery: **QCOV** earns a
  coefficient (+1.19 to +1.29) rivalling AND, sharpens the decomposition out-of-fold with a
  log-loss CI above zero on two of three models, and lifts DL20 reranking (writeup §3.1–3.3). The
  lexical account is real; it is also small.

- **The classical SEMANTIC tier sat at chance.** The RQ2 WordNet axioms — STMC1, STMC2, REG,
  ANTI-REG on ir_axioms' WordNet synonym-set backend — added nothing in Phase 1 (STMC1 0.52–0.55,
  the rest hovering at 0.5 with CIs spanning it, `phase1-writeup.md` §3.7) and *lowered* pooled CV
  accuracy under cross-validation in Phase 2 (Qwen 0.629 → 0.623, flan 0.666 → 0.652,
  `phase2-writeup.md` §3.2). The reported reading was explicit that this is a null **on WordNet
  similarity specifically**, "genuinely blunt" — synonym-set overlap is a poor model of how an
  LLM judges relevance, not evidence that semantics is irrelevant.

- **The large residual is semantic/relevance-shaped.** The reducible residual is ~0.21 (Qwen) of
  the behaviour and >90 % of the model's label entropy is unexplained (§3.1, §3.3). Its
  characterised part is *content*: length and query-coverage clusters, replicated across two
  architectures (§3.4–3.5). Increment 1's VERB→coverage finding sharpens this — the length seed
  did **not** survive discretisation (VERB null), while the *coverage* seed did (QCOV), which
  says the surviving structure is about **what query need the document covers**, a semantic /
  relevance dimension, not raw verbosity. Most of the reducible residual is still uncaptured even
  by QCOV.

A reranking check makes the ceiling concrete. On BM25's top-10, the untuned axiom vote
(classical + VERB + QCOV) reranks to **BM25 parity** (DL19 +0.008, DL20 −0.013 nDCG@10, both CIs
spanning 0), and even a *fitted* weighted axiom surrogate — an L2 logistic trained out-of-fold to
mimic each LLM, the RQ5 preview in `experiments/rq5_surrogate/` — recovers only **~37–60 % of the
LLM's over-BM25 gain on DL19 and nothing on the saturated DL20**, while the LLMs themselves beat
BM25 by +0.04–0.07 (≈ 70 % of the perfect-top-10 oracle's +0.09). So the lexical axioms, *even
optimally weighted*, top out well short of the LLM: the ranking skill the thesis is after lives in
the part they cannot reach — which is the semantic residual the main RQ4 line goes after.

**The move.** Replace the *blunt WordNet operationalisation of semantics* with better ones. The
residual is relevance-shaped; WordNet was the wrong instrument for relevance; QCOV shows a
coverage-style semantic criterion pays. So the main RQ4 line is a systematic investigation of
**LLM-aligned semantic preference axioms** — constraints that model the dimensions along which an
LLM judge actually decides relevance — evaluated by the same capture protocol Increment 1
established. This is *not* a claim that semantics dominates; it is the disciplined next probe into
the part of the residual Phase 2 left uncharacterised.

## 2. The two-tier design — and how it resolves the circularity objection

The obvious objection to explaining a neural ranker with neural signals is **circularity**: an
LLM-judge that scores "relevance" the way the ranker scores relevance can trivially predict the
ranker without teaching us anything transferable or cheap. The design answers this structurally by
splitting every semantic criterion into two tiers with different jobs.

- **Tier A — cheap frozen-neural proxy axioms (deliverable-grade).** Computed *once* from a
  **frozen** sentence-embedder plus spaCy — no fine-tuning, no per-query model reasoning,
  interpretable, and orders of magnitude cheaper than the 35B pairwise ranker. A Tier-A axiom is a
  deterministic function of embeddings/POS/NER over the query and the two documents, in the same
  pairwise `{+1, 0, −1}` shape as PROX1 and QCOV. These are the **shippable** axioms. **The RQ5
  linear-time surrogate is built ONLY from the cheap tier** — Tier B never enters the surrogate,
  so the efficiency deliverable (RQ5, and RQ6 if reached) is preserved: a surrogate made of frozen
  embed-once + spaCy features is still an N-not-N² scorer, still far cheaper than the ranker.

- **Tier B — LLM-judge "oracle" axioms (DIAGNOSTIC, not the final axiom).** An LLM judge scores
  the criterion directly per pair (e.g. "which document more completely addresses the query's
  sub-questions?"). Its role is **not** to be shipped. It is to *measure what semantic criterion
  predicts the ranker's residual* — an **upper bound** on how much that criterion, operationalised
  as well as an LLM can, captures the ranker. It buys that measurement at the price of being
  circular and expensive. It is scaffolding.

**The logic to spell out.** The Tier-B oracle establishes **what matters** — which semantic
criteria actually predict the residual, at their best-case operationalisation. We then invest in a
cheap deterministic **how** — a Tier-A proxy (or a future distilled approximation) — **only for
criteria that pay off** under the oracle. This inverts the usual risk: rather than hand-build a
dozen cheap proxies and hope one hits, we let the (expensive, throwaway) oracle prune the menu,
then spend engineering only where the ceiling is high.

**The A↔B capture gap as a built-in roadmap.** Where a Tier-A proxy and a Tier-B oracle target the
**same** criterion (Semantic Intent Coverage ↔ Aspect Completeness; the evidence proxy ↔ the
faithfulness judge), the difference between the oracle's capture and the proxy's capture measures
**how much the cheap approximation leaves on the table**. A small gap says the frozen-neural proxy
is already near the criterion's ceiling — ship it and move on. A large gap says the criterion
matters but our cheap `how` is inadequate — a flagged target for future approximation work. Either
way the gap is a *result*, not a failure, and it converts "explain a neural ranker with neural
judges" from a circularity trap into a two-sided measurement: Tier B says how far one could go,
Tier A says how far one can go cheaply, and the gap is the research agenda.

## 3. The candidate axioms — an investigation menu (not a shipping commitment)

A deliberately **broad menu**, presented as an investigation into *what works*. The capture
protocol (§4) prunes it; RQ4 does not commit to shipping all of these. Each candidate is tagged
**A** (cheap frozen-neural proxy) or **B** (LLM-judge oracle); several categories carry a matched
A/B pair so the §2 gap can be read.

| # | Category | Candidate | Tier | Operationalisation (sketch) |
|---|---|---|---|---|
| 1a | Coverage / Completeness | **Semantic Intent Coverage** | **A** | Per query-aspect (or query-term), max sentence-embedding similarity over the document's sentences; aggregate. The neural upgrade of the validated QCOV — graded coverage of *meaning*, not surface term overlap. |
| 1b | Coverage / Completeness | **Aspect Completeness** | **B** | LLM decomposes the query into sub-questions / aspects; judge counts how many each document addresses. Oracle for 1a. |
| 2a | Focus / Precision | **Semantic Focus / distraction** | **A** | Mean vs. spread of sentence-to-query similarity across the document; penalise off-topic sentences (high spread / low mean tail). |
| 2b | Focus / Precision | **Redundancy penalty** | **A** | Intra-document mean pairwise sentence similarity; penalise documents that repeat themselves rather than adding coverage. |
| 3a | Specificity / Depth | **Specificity** | **A (cheap)** | IDF-weighted content-word density + named-entity density + numeral density via spaCy — no embedder needed. Prefer the more specific document. |
| 3b | Specificity / Depth | **Explanation / Causal quality** | **B** | For why/how queries: judge scores explanatory adequacy. Possible cheap approx = discourse-/causal-marker density ("because", "therefore", …) — noted as a candidate Tier-A `how`, not built yet. |
| 4 | Directness | **Answer localisation** | **A** | Normalised position of the peak query-similarity sentence; earlier = better (a direct answer up front beats a buried one). |
| 5a | Grounding | **Evidence Support** | **A (cheap)** | Citation / number / quote-marker density as a proxy for grounded claims. |
| 5b | Grounding | **Factual faithfulness** | **B** | Judge scores whether claims are supported. Oracle partner for 5a. |
| 5c | Grounding | **Factual Consistency** | **B** | Internal-contradiction judge. **No obvious cheap approximation — flagged as hard**; likely oracle-only, a ceiling measurement with no Tier-A partner yet. |
| 6 | Entity Alignment | **Entity Relevance** | **A** | spaCy NER; overlap of the query's entities / entity-types with the document's entities. Prefer the better entity-aligned document. |

Matched A↔B pairs for the gap analysis: **1a↔1b** (semantic coverage), **5a↔5b** (grounding).
Category 3 offers a *prospective* pair (3a specificity is standalone; 3b's discourse-marker approx
would be its cheap partner if built). Category 5c is the honest hard case — an oracle with no cheap
`how`, reported as an upper bound only.

Note the spaCy pin is `en_core_web_sm`, which has no useful word vectors; the Tier-A embedding
candidates therefore need a *separate* frozen sentence-embedder (§6), while the pure-spaCy
candidates (3a, 5a, 6) need only the existing dependency.

## 4. Evaluation protocol (reuse Increment 1's — same runner, same test)

Every candidate, Tier A or Tier B, is reduced to a **per-pair signed preference** `{+1, 0, −1}`
(+1 favours doc 1), added as a **battery column**, and put through the **same
decomposition-capture test** Increment 1 used for QCOV in the `rq4_axioms` runner
(`experiments/rq4_axioms/run.py`):

1. **Capture (per candidate).** Classical battery vs. classical + candidate, query-grouped CV,
   per ranker (Qwen primary; flan-t5-large and flan-t5-xl replication). Report OOF **accuracy**
   lift *and* per-pair **log-loss** lift, each with a 2000-draw paired query-bootstrap CI, plus
   the **fitted coefficient** (direction honesty — an oppositely-signed coefficient is reported,
   not flipped, exactly as VERB was). Rank candidates by capture.
2. **Tier-B oracles report a per-criterion upper bound**, at the cost of being circular; **Tier-A
   proxies are evaluated both standalone and against their Tier-B oracle** — the A↔B capture gap
   of §2.
3. **Reranking validation as in Increment 1.** Add the surviving cheap-tier axioms to two
   aggregates — the untuned **majority vote** and the **fitted surrogate** (the RQ5-preview
   baseline, `experiments/rq5_surrogate/`) — and report **lift over the classical battery** on
   each, paired query-bootstrap CI, **not absolute nDCG** (the depth-10 oracle ceiling of ≈ 0.58,
   `phase2-writeup.md` §3.8, caps the metric). The fitted surrogate is the fair test: the untuned
   vote sits at BM25 parity (it re-expresses BM25), while the fitted lexical surrogate already
   recovers ~37–60 % of the LLM's gain on DL19 (§1) — so a semantic axiom *earns its place* only
   if it moves the **fitted** surrogate measurably toward the LLM, not merely the weak vote.

Only cheap-tier (Tier-A) axioms are eligible for the reranking deliverable and the RQ5 surrogate;
Tier-B oracles stop at the capture-upper-bound role.

## 5. Pre-registered decision criteria (fixed before the numbers)

Keeping the Phase 1/2 pre-registration discipline, written before any semantic-axiom result:

1. **Capture.** A candidate *counts* only if its capture CI is above zero on the **primary model
   (Qwen)** *and* the effect **replicates** on at least one FLAN rung. A clean null is a
   reportable boundary result (as VERB was in Increment 1).
2. **Content-carried.** The lift must be carried by the semantic content criterion, not by
   re-absorbing a confidence / BM25 proxy — enforced by comparing against the classical *axiom*
   battery, never the ranker's own signals.
3. **Direction honesty.** The fitted coefficient's sign is reported as found; an axiom whose
   coefficient contradicts its designed direction is a finding, not a bug to flip.
4. **Tier discipline (framework-level).** A criterion is only "captured cheaply" if a **Tier-A**
   proxy meets 1–3; a Tier-B-only pass is an *upper bound*, reported as such, and names a target
   for future cheap approximation (the A↔B gap).

## 6. Infra prerequisites (OPEN decisions — not resolved here)

Both are genuinely unresolved and are recorded as open, not chosen:

- **A frozen sentence-embedder** for the Tier-A embedding candidates (1a, 2a, 2b, 4). `en_core_web_sm`
  has no vectors, so this is a new dependency. fastText is a gated ~7.24 GB dep (the same one that
  fired only nominally in Phase 1); `sentence-transformers` would need `uv add`. **Undecided** —
  weigh dependency weight, licence, and CPU cost on the bronze box.
- **An LLM-judge backend** for the Tier-B oracles. Options: bronze's Qwen via the existing
  `openai` ranker backend (`src/axiomrank/rankers/openai_api.py`, already wired for a base-URL +
  model), or an external API. **Undecided.** The Tier-B judge carries **real API/compute cost and
  is a diagnostic budget** — spent to measure ceilings, not to ship. If judge verdicts are cached,
  they follow the **same append-only preference-store contract** as `data/preferences/` (never
  overwrite; verdict storage is append-only, per CLAUDE.md).

## 7. Constraints and deliverables

- **uv only** (`uv run --no-sync` per the flaky-fetch note); Python 3.12. Resolve all paths through
  `axiomrank.paths`; write results only under `results/`. No `Co-Authored-By` trailer; do not
  commit unless asked.
- **Append-only caches** throughout: new axiom columns only; existing axiom and LLM-verdict columns
  are never overwritten (Increment 1's additive-column discipline extends unchanged). Tier-B judge
  verdicts, if cached, obey the same contract.
- **Cost boundary.** Tier A is (near-)free CPU over cached text plus a one-time frozen-embed pass.
  Tier B spends LLM/API budget deliberately and only on criteria the design chooses to ceiling.
- **Deliverables.** The semantic-axiom implementations + tests; the extended configs; the
  per-candidate capture ranking and the A↔B gap tables under `results/rq4_axioms/`; the cheap-tier
  reranking validation; and the write-up in the Phase 1/2 discipline (criteria §5 stated before
  results, nulls reported as boundary results). Increment 1 (QCOV validated, VERB null) stands as
  the recorded first probe; the semantic investigation is the main RQ4 line.
</content>
</invoke>
