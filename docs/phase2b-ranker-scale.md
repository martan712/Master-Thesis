# Phase 2b — Ranker-scale validation (the competence-and-scaling check)

> A focused validation study bridging Phase 2 (RQ3 decomposition) and Phase 3 (RQ4, new
> axioms). It answers one question that the whole thesis rests on but that Phase 1's
> effectiveness gate only partly settled — *are the rankers we study actually good, and does
> the axiom-null survive as they get better?* — before RQ4 commits to the residual. It
> introduces no new research question; it re-runs the existing effectiveness gate
> (`phase1-design.md` §, `phase1-writeup.md` §3.1) and the Phase 2 decomposition across a
> controlled model-scale ladder. Companion scientific/engineering docs are the Phase 2 pair;
> this is deliberately one document because the study is small.

## 1. The objection being answered

Phase 1's effectiveness gate passed: aggregated by Copeland (= PRP-allpair), both rankers
beat BM25 on both collections with bootstrap CIs above zero, so the residual the thesis
studies is skill rather than model error. But the gate came with a soft spot the write-up
flagged (`phase1-writeup.md` §3.1): reranked nDCG@10 ≈ 0.55 sits **below the ≈ 0.70
PRP-allpair anchor** for strong FLAN-T5 rerankers in the literature (Qin et al. 2024, NAACL
Findings). Neither ranker is random, but neither is a *strong* reranker either.

This leaves a real risk that Phase 1's gate does not close: **the axiom-null could be an
artefact of ranker mediocrity, or could reverse for a strong ranker.** If a genuinely strong
reranker were *more* axiom-aligned, the thin explained part of Phase 2 (§7.1) would be a
property of weak models, not of LLM pairwise ranking; if it were *less* aligned, the residual
would be even larger and RQ4 even better motivated. Either way the current two-model evidence
cannot say, and RQ4 should not be built on an unaddressed confound.

There is a within-Phase-2 hint that already leans the reassuring way: flan-t5-large — the
*weaker* ranker on the gate — has the *higher* pseudo-R² (0.074 vs Qwen's 0.057, §7.1), i.e.
the weaker ranker is *more* axiom-explained, the stronger one less. That is the opposite of
what the "strong rankers are more axiomatic" objection needs. But it is two points confounded
by architecture (0.8B seq2seq vs 35B MoE chat), so it is suggestive, not decisive. A
controlled scale ladder within one family resolves it.

## 2. Why a within-family ladder (and not rank_llm, and not a downloaded dataset)

**The pipeline already is a PRP reranker.** `rankers/hf.py` loads any seq2seq model by name
through `AutoModelForSeq2SeqLM` and scores the two passages by label log-likelihood — "the
scoring mode of PRP (Qin et al.)". The configs merely name `model: google/flan-t5-large`. So
the entire model change is `flan-t5-large` → `flan-t5-xl` (3B) → `flan-t5-xxl` (11B): the
same T5 family, the same prompt, the same order-swap, tie-collapse, store and axiom
decomposition, with only the scale varying. That makes large→xl→xxl a **clean scale contrast
with everything else held fixed** — cleaner than Qwen-vs-flan, which confounds scale with
architecture and training.

**rank_llm (Castorini) is the wrong tool here.** It is listwise-focused (RankZephyr, RankGPT,
sliding-window) and emits final run files, not the per-pair verdicts the store and the
fidelity analysis need; adopting it means re-plumbing its output for no gain, since we already
implement exactly the pairwise-scoring method the PRP FLAN-T5 numbers come from. Its only
possible use is an *external* listwise nDCG reference — a footnote, not infrastructure.

**No drop-in dataset exists.** The IR reranking community treats intermediate pairwise
comparisons as disposable compute and publishes only run files; the PRP paper *planned* to
release its pairwise inference JSON but no live release is findable. Adjacent pairwise
datasets (LMSYS Arena, LLM-Blender MixInstruct, PairRM/OpenHermesPreferences) judge
answer-quality, not query-document relevance, so the axiom battery has nothing meaningful to
compute on them — a task mismatch, not a format gap. Reproduction is the only route, and it is
cheap because the SOTA PRP models are open weights.

## 3. What the ladder validates — two questions

**Q-A — Pipeline fidelity to the literature.** Does our PRP-scoring pipeline reproduce the
paper's PRP-allpair nDCG@10? The anchor points (Qin et al., PRP-Allpair):

| model | DL19 | DL20 |
|---|---|---|
| FLAN-T5-XL (3B)  | 0.698 | 0.681 |
| FLAN-T5-XXL (11B) | 0.699 | 0.699 |
| FLAN-UL2 (20B)   | 0.724 | 0.707 |

Note flan-t5-**large** (0.8B) is *below* the paper's range (their smallest is XL), so it is
the unanchored low rung of the ladder; **XL is the first anchorable point.** Two protocol
differences mean we should expect the *neighbourhood* of these numbers, not the exact values:
our aggregation **collapses position-inconsistent pairs to ties** (theirs sums raw wins), and
the prompt wording differs. So the rigorous comparison is the **internal ladder**
(our-large / our-xl / our-xxl under one protocol); the PRP table is an external sanity anchor.
If even XXL lands far below ~0.70 under our protocol, that flags a protocol issue (the
tie-collapse or the prompt) to fix *before* RQ4, and explains the Qwen-35B-at-0.55 anomaly as
setup rather than model.

**Q-B — Scaling of axiom-explainability.** Re-run the Phase 2 decomposition on each rung and
track, as a function of scale: pooled CV accuracy and gain over base, pseudo-R² (the honest
explained fraction), the reliability ceiling / reducible-residual estimate, and the content
residual-model lift and its clusters (verbosity `d_len`, coverage `d_qcov`, §7.2). The
question is whether the axiom-null and the verbosity/coverage residual **persist, strengthen,
or dissolve** as the ranker gets stronger.

## 4. Method (engineering)

The stages (`build_pool` / `build_pairs` / `build_axiom_prefs`) are model-independent and
already cached, so a new ranker reuses them and computes **zero new axiom columns**; only the
model-call collection step is new, and the decomposition is then free from cache.

1. **GPU/dtype support (prerequisite code change).** `make_ranker` currently passes only
   `model_name`, `prompt_version`, `max_chars` to `HFPairwiseRanker`; `RankerConfig` has no
   `device` or `dtype` field, so the backend always runs **CPU / fp32**. XL in fp32 is ~12 GB
   (tight) and XXL ~44 GB (won't fit a consumer GPU). Add `device` and `dtype` fields to
   `RankerConfig`, thread them through `make_ranker`, and use them in `HFPairwiseRanker._load()`
   (`torch_dtype=torch.float16`, optional `load_in_8bit` for XXL). Small, localised change;
   guard with a test that the fields parse and default to today's `cpu`/`fp32` (so all
   existing configs stay byte-for-byte reproducible).
2. **Add the rungs.** Add `google/flan-t5-xl` (then `google/flan-t5-xxl`) as `backend: hf`
   ranker entries to the two top-10 configs (`rq2_dl19_top10.yaml`, `rq2_dl20_top10.yaml`),
   `device: cuda`, `dtype: float16`. No other config changes — the effectiveness and rq3
   runners already iterate over `rankers`.
3. **Collect verdicts** for the new rungs on the two top-10 cells, order-swap mandatory, into
   the append-only store (new `model_name` keys — never overwrite existing verdicts, per the
   project rule). This is the only compute cost.
4. **Effectiveness gate** (`ranking_effectiveness` runner): Copeland/PRP-allpair vs BM25,
   nDCG@10 primary + MAP, paired query-bootstrap CIs, per new rung — extending the Phase 1
   gate table.
5. **Decomposition**: re-run `experiments/rq3_decomposition/run.py` with the new rankers in
   scope; per-rung `decomposition.json` / `residual_model.json` / clusters land under
   `results/rq3_decomposition/pooled_top10/metrics/<model>/` exactly as for the existing two.

Sequence to control cost: **XL first** (first anchorable rung, ~4× flan-large per pass), read
Q-A before spending XXL (~14×). flan-large is already the bottom rung (its verdicts and
decomposition exist from Phase 2); Qwen stays in as the cross-architecture check.

## 5. Cost

- **Model calls: open-weight, no API cost.** Only the collection step is new; axiom stages and
  the decomposition are cached/free.
- **Local GPU.** flan-t5-large took ~4.6 h CPU for its Phase 1 cells; here only the two top-10
  cells are needed (DL19 top-10 1,900 pairs + DL20 top-10 2,430 pairs, each in both orders,
  two label-logliks per comparison). XL ≈ 4× and XXL ≈ 14× the per-pass cost — feasible on CPU
  for XL but slow, so a GPU is strongly preferred; XXL effectively requires one (fp16 ≈ 22 GB,
  fits a 24 GB card, or 8-bit ≈ 11 GB).

## 6. Outputs

- **Effectiveness (extends the Phase 1 gate table):** BM25 vs reranked nDCG@10 / MAP with CIs
  for XL and XXL, plus a column giving the paper's published PRP-allpair nDCG at each anchored
  rung and the gap to it.
- **Scaling table:** one row per rung (flan-large 0.8B, xl 3B, xxl 11B, and Qwen 35B as
  cross-arch) with n, base rate, CV accuracy, gain, pseudo-R², reliability ceiling,
  reducible-residual upper bound, and content residual-model lift [CI]. This is the figure
  that answers the objection.
- **A validation note** appended to `phase2-writeup.md` (or a short §8) recording the outcome
  and the decision below.

## 7. Pre-registered decision criteria

Fixed before the numbers are seen, in the Phase 1/2 discipline:

1. **Competence confirmed.** XL and/or XXL reproduce PRP-allpair nDCG@10 within ~0.05 of the
   paper under our protocol → pairwise LLM ranking reaches ~0.70 in our pipeline, the gate's
   soft spot is closed, and the Qwen-0.55 gap is attributable to architecture/setup, not to a
   ceiling on the method. If even XXL lands far short, the finding is a **protocol fix**
   (tie-collapse / prompt) to make before RQ4, not a green light.
2. **The finding is quality-invariant (or its scaling is characterised).** Across the ladder
   the explained part stays thin (pseudo-R² small, no rung becomes highly axiom-explained) and
   the content residual (verbosity/coverage) persists → "rankers too weak" is refuted and RQ4
   stands as the main act. If axiom-explainability instead changes monotonically with scale,
   that trend is itself a scaling result folded into RQ4's framing (develop new axioms where
   the residual is largest, i.e. at the strong end).
3. **Recorded outcome.** The scaling table, the reproduced-nDCG check, and which of the above
   fired are written into §8 of the Phase 2 write-up and gate entry to Phase 3.

## 8. Risks

- **Compute / GPU.** XXL needs the dtype change and a ≥24 GB card; XL is the low-risk first
  rung. Mitigation: XL first, XXL only if Q-A/Q-B warrant it.
- **Protocol mismatch with the paper.** We will not hit 0.70 exactly; the internal ladder is
  the rigorous comparison and the PRP table only an anchor — stated as such, not treated as a
  target to tune toward (which would be nDCG-hacking).
- **Store integrity.** New rungs append under new `model_name` keys; the append-only store is
  never overwritten (the project's most expensive artefact rule).
- **Scope.** The ladder is within one family (flan-t5); cross-*architecture* generality still
  rests on Qwen. The scaling claim is therefore "within the FLAN-T5 family, with Qwen as a
  cross-architecture check", and is stated with that limit.
- **Reproducibility of existing cells.** The `device`/`dtype` fields must default to the
  current `cpu`/`fp32` so every Phase 0/1/2 config and result stays bit-for-bit reproducible;
  enforced by the parse test.
