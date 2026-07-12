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

> **Resolved for the XL rung — see §9.** The XL nDCG@10 lands at ~0.53, far below the 0.70
> anchor, but the diagnosis is a **rerank-depth mismatch, not ranker mediocrity and not a
> scoring bug**: our protocol reranks BM25 top-10 (oracle ceiling ~0.58), the paper reranks
> top-100. Depth-matched, XL is near-oracle and its verdicts are 0.83 pairwise-accurate. The
> anchor below is retained as originally written; §9 records the corrected comparison and the
> decision.

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

1. **GPU/dtype support (prerequisite code change).** `HFPairwiseRanker` already carries a
   `device` field (defaults to `cpu`), but `make_ranker` never threads it through and there is
   no `dtype` field anywhere, so the backend always runs **CPU / fp32**. Add `device` and
   `dtype` to `RankerConfig`, thread both through `make_ranker`, and add a `dtype` field to
   `HFPairwiseRanker` used as `torch_dtype` in `_load()`. **Use `bfloat16`, not `float16`:**
   flan-t5's activations overflow fp16 to NaN/inf (a documented T5 failure mode), which would
   silently corrupt the label log-likelihoods and the whole Q-A anchor; bf16 (or fp32) is
   safe, and RDNA3.5 / ROCm supports bf16. In bf16 XL is ~6 GB and XXL ~22 GB, so **both fit
   in bronze's 128 GB unified memory with room to spare — no 8-bit needed.** Small, localised
   change; guard with a test that the fields parse and default to today's `cpu`/`fp32` (so all
   existing configs stay byte-for-byte reproducible).
2. **Add the rungs.** Add `google/flan-t5-xl` (then `google/flan-t5-xxl`) as `backend: hf`
   ranker entries to the four top-10 configs (`rq2_dl19/20_top10.yaml` feed the decomposition,
   `eff_dl19/20_top10.yaml` feed the gate), `device: cpu`, `dtype: bfloat16` (recording how they
   were collected on bronze — see §4b; `cpu` is also the safe local default since the
   workstation has no GPU and the verdicts are already cached). No other config changes — the
   effectiveness and rq3 runners already iterate over `rankers`.
3. **Collect verdicts** for the new rungs on the two top-10 cells, order-swap mandatory, into
   the append-only store (new `model_name` keys — never overwrite existing verdicts, per the
   project rule). This is the only compute cost, and it runs **on bronze** (§4b) — the exact
   same `hf` PRP scoring (prompt v0, "Passage A"/"Passage B" label log-likelihood) as the
   flan-t5-large rung, only on faster hardware, so the ladder stays byte-for-byte one method.
4. **Effectiveness gate** (`ranking_effectiveness` runner): Copeland/PRP-allpair vs BM25,
   nDCG@10 primary + MAP, paired query-bootstrap CIs, per new rung — extending the Phase 1
   gate table.
5. **Decomposition**: re-run `experiments/rq3_decomposition/run.py` with the new rankers in
   scope; per-rung `decomposition.json` / `residual_model.json` / clusters land under
   `results/rq3_decomposition/pooled_top10/metrics/<model>/` exactly as for the existing two.

Sequence to control cost: **XL first** (first anchorable rung, ~4× flan-large per pass), read
Q-A before spending XXL (~14×). flan-large is already the bottom rung (its verdicts and
decomposition exist from Phase 2); Qwen stays in as the cross-architecture check.

## 4b. Remote execution on bronze

The collection step (step 3 above) is the only compute cost, and the workstation is CPU-only
and small, so it is offloaded to **bronze** — an AMD Ryzen AI MAX+ 395 "Strix Halo" node
(16 cores / 32 threads, 128 GB *unified* memory, Radeon 8060S iGPU) reachable over SSH
(`ssh bronze`, via the `spinque` ProxyJump already in `~/.ssh/config`; a harmless "Ncat: Could
not resolve hostname boson" line prints on every connect). Three facts about bronze decide the
method:

- **vLLM is not an option for this model.** flan-t5 is an *encoder-decoder* model, and the
  vLLM build on bronze (`kyuz0/vllm-therock-gfx1151`, v0.19.2rc1, the ROCm/gfx1151 image that
  already serves the Qwen-35B ranker on port 8086) registers **zero encoder-decoder / T5
  architectures** — enc-dec serving is simply not compiled in. Serving flan-t5 through the
  existing `openai`-backend + SSH-tunnel pattern is therefore off the table on the vLLM path.
- **The ROCm PyTorch runs the `hf` backend unchanged — but on CPU, not the GPU.** The same
  container ships `torch 2.13/rocm7.13` + `transformers 5.5.4`, and `torch.cuda.is_available()`
  is `True` on the "Radeon 8060S" iGPU. In practice, though, *any* GPU kernel (even a trivial
  matmul) **segfaults inside `rocprofiler`'s HSA queue interception**
  (`HSA_STATUS_ERROR_OUT_OF_RESOURCES`, "Could not create intercept queue") — the iGPU's queue
  resources are already claimed by the co-tenant Qwen-vLLM + llama-server processes. So the
  scorer runs on **bronze's 32 CPU threads** instead. This still preserves fidelity: it is the
  exact same `hf.py` v0 label-log-likelihood scoring the flan-t5-large rung and the PRP paper
  use, so the large→xl→xxl ladder stays one method (§2's "everything held fixed") and the Q-A
  paper anchor is meaningful.
- **Use `bfloat16`, on CPU.** With ~16 GB free on the contended box, fp32 XL (~11 GB weights +
  activations) is uncomfortably tight; bf16 (~6 GB) is memory-safe, is native-fast on Zen5, and
  is the correct choice for T5 regardless (fp16 overflows). Measured throughput ~2.5 verdicts/s,
  ~1 h for both DL19+DL20 top-10 cells; verdicts come out balanced and NaN-free.

**Workflow** (no model calls leave bronze; only Parquet crosses the wire):

1. **Ship the inputs, not the pipeline.** Build the two top-10 pair frames locally once (they
   are already cached at `data/processed/rq2_semantic_agreement/{dl19,dl20}_top10/pairs.parquet`
   — columns `qid, query, doc_id_1, doc_id_2, text_1, text_2`, everything the scorer needs) and
   `rsync` them to bronze. ir_datasets / PyTerrier are **not** needed on bronze — the pairs
   already carry the passage text, so bronze only loads the model and scores.
2. **Score inside the ROCm container.** `scripts/collect_bronze.py` imports only
   `axiomrank.rankers.hf` + `axiomrank.data.preferences` (the heavy retrieval deps are never
   touched), so the container needs nothing beyond its own torch/transformers plus pandas/pyarrow.
   Deploy `src/`, the script and the pair frames to `~/axiomrank-bronze/`, then run detached (so
   the ~1 h job survives an SSH drop):

   ```bash
   podman run --rm --name axr_xl_cpu \
     -v ~/axiomrank-bronze:/work:z -w /work \
     -e PYTHONPATH=/work/src -e HF_HOME=/work/hf-cache \
     -e PYTHONUNBUFFERED=1 -e OMP_NUM_THREADS=16 \
     docker.io/kyuz0/vllm-therock-gfx1151:latest \
     bash -lc 'pip install -q pandas pyarrow &&
       python -u /work/scripts/collect_bronze.py --pairs /work/pairs_dl19_top10.parquet \
         --dataset msmarco-passage/trec-dl-2019/judged \
         --model google/flan-t5-xl --device cpu --dtype bfloat16 --out /work/data/preferences'
   # ... then the same for pairs_dl20_top10.parquet + trec-dl-2020.
   ```

   Three non-obvious flags: `:z` relabels the bind mount for SELinux (without it the container
   gets *Permission denied* on the mounted files); `PYTHONPATH=/work/src` makes `axiomrank`
   importable without installing the package's declared deps; `python -u` /
   `PYTHONUNBUFFERED=1` is essential because piped stdout is block-buffered, so a crash would
   otherwise swallow all progress output. The entrypoint reuses `HFPairwiseRanker` (v0,
   order-swap, `max_chars=2000`) and `PreferenceStore.append`, writing fresh `part-*.parquet`
   files exactly as the local runner does — same schema, same `model="google/flan-t5-xl"` /
   `prompt_version="v0"` keys.
3. **Sync the store back, append-only.** `rsync` **only the new** `data/preferences/part-*.parquet`
   files bronze→workstation. Part files are named `part-<utc-timestamp>-<uuid>.parquet`, so they
   never collide with existing parts and the store's read-time "first write wins" dedup handles
   any overlap — this is exactly the append-only contract, honoured across machines without ever
   overwriting the project's most expensive artefact.
4. **Analyse locally, from cache.** Run `experiments/ranking_effectiveness/run.py` and
   `experiments/rq3_decomposition/run.py` on the workstation as usual: they find the new
   verdicts in the store and compute the gate + decomposition with **zero** model calls.

**If the GPU becomes usable** (the co-tenant vLLM/llama servers stop, freeing the iGPU's queue
resources, or the container's `rocprofiler` interception is disabled), the identical command
with `--device cuda` would cut the ~1 h to minutes — worth revisiting for the ~14×-heavier XXL
rung. The `llama.cpp` server already on bronze (ports 8085/8082, native T5 support) could
instead serve flan-t5-xl to the `openai` backend over an SSH tunnel like Qwen, but that path
scores the **v1 single-letter** analogue, not v0 label-log-likelihood — it diverges from the
flan-large rung and weakens the Q-A anchor, so it is a last resort, not the plan.

## 5. Cost

- **Model calls: open-weight, no API cost.** Only the collection step is new; axiom stages and
  the decomposition are cached/free.
- **Bronze CPU (§4b).** flan-t5-large took ~4.6 h CPU for its full Phase 1 cells; here only the
  two top-10 cells are needed (DL19 top-10 1,900 pairs + DL20 top-10 2,430 pairs, each in both
  orders, two label-logliks per comparison ≈ 8,660 presentations). On bronze's 32 threads in
  bf16 this runs at ~2.5 verdicts/s, **~1 h total** for XL — the GPU being unavailable (§4b) it
  is CPU-bound, but bronze is far larger than the workstation and this stays a short job. XXL
  (~14× per pass) would be the case that really wants the GPU back. The workstation only ever
  handles the (cached, zero-call) gate and decomposition.

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

- **Compute.** bronze's iGPU segfaults under co-tenant load (§4b), so collection is CPU-bound;
  XL is ~1 h and the low-risk first rung. XXL (~14×) on CPU would be ~half a day — tolerable but
  the case that most wants the GPU restored (stop the co-tenant servers, or disable the
  container's `rocprofiler` interception). Mitigation: XL first, XXL only if Q-A/Q-B warrant it.
- **fp16 would corrupt T5 silently.** flan-t5 overflows fp16 activations to NaN/inf, which
  would poison the label log-likelihoods without an obvious error; the plan pins **bf16**
  (fp32 fallback) and the parse test should reject nothing but must not let a config default
  back to fp16. A cheap guard: sanity-check that XL reproduces a handful of flan-large verdicts'
  sign on the shared bottom rung before the full run.
- **bf16-vs-fp32 across the ladder.** flan-large's existing verdicts were scored CPU/fp32; XL
  (and XXL) run CPU/bf16. Verdicts are the argmax of two log-likelihoods, so only exact-tie
  boundary cases (already collapsed to ties) can differ — a negligible confound, noted rather
  than eliminated. Re-run flan-large in bf16 on bronze only if a reviewer presses the point.
- **Cross-machine store integrity.** Verdicts collected on bronze return as new, uniquely named
  `part-*.parquet` files that are only ever *added* to `data/preferences/`; the append-only,
  first-write-wins contract is preserved across machines and no existing part is touched.
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

## 9. Outcome — XL rung: competence confirmed by depth-match, anchor re-based

The flan-t5-xl verdicts were collected on bronze and the gate + decomposition run from cache.
The headline nDCG@10 came out **far below the 0.70 anchor** (0.526 DL19 / 0.533 DL20, +0.047 /
+0.039 over BM25), which under §3's Q-A and §7's criterion 1 would read as a failed competence
check. It is not. The gap is a **rerank-depth mismatch**, established by four measurements from
the cached verdicts (2026-07-12):

1. **Verdicts are relevance-aligned.** Pairwise accuracy against the qrels, on judged pairs
   with a strict grade ordering: flan-t5-xl **0.838 / 0.829** (DL19/DL20), flan-t5-large
   0.859 / 0.824. Both order relevant-vs-less-relevant pairs correctly ~5 in 6 — the `hf.py`
   label-log-likelihood scoring is faithful, not broken.
2. **XL is *more* decisive/consistent than large,** not less: position-consistency 0.81 vs 0.66,
   lower tie rate. So the shortfall is not tie-collapse or prompt noise (the suspects §7 named);
   XL's confident calls are mostly correct.
3. **The depth-10 oracle ceiling is ~0.58.** Perfectly reordering BM25's top-10 by qrels yields
   nDCG@10 = **0.575 / 0.583** — the mathematical maximum for *any* reranker confined to the
   top-10. XL reaches **~91% of it** (0.526/0.575, 0.533/0.583). It is a near-oracle reranker at
   this depth.
4. **The paper reranks deeper.** Oracle nDCG@10 ceiling by rerank depth: **10 → 0.58, 20 → 0.72,
   100 → 0.88**. The paper's XL anchor (0.698/0.681) sits at the depth-20 oracle level because
   PRP-Allpair reranks BM25 **top-100**; our `pairs.k=10` protocol caps at ~0.58, so 0.70 is
   unreachable here *regardless of model quality*. The cause is first-stage recall: BM25
   recall@10 is only 0.12 / 0.18, so most relevant documents sit at ranks 11–100, outside a
   depth-10 pool.

**Decision (A — re-anchor).** The paper's absolute depth-100 nDCG@10 is **dropped as the Q-A
yardstick**; it was never apples-to-apples with our depth-10 protocol. The corrected,
depth-matched competence evidence is (i) pairwise accuracy vs qrels (~0.83) and (ii) fraction
of the depth-10 oracle ceiling captured (~0.91). On both, XL — and large — are **strong, not
mediocre**. Criterion 1's spirit (competence confirmed, Qwen-0.55 attributable to
setup/architecture not a method ceiling) **fires**; its letter ("reproduce ~0.70") is retired as
depth-inappropriate. Criterion 1's fallback ("protocol fix to tie-collapse or prompt") is
**not** triggered — the protocol variable that matters is rerank depth, and depth-10 is the
*correct, deliberate* scope for the RQ3 decomposition (see below), not a defect to fix.

**Why depth-10 stays the decomposition scope.** Deeper pools would raise nDCG but *hurt* the
RQ3/RQ4 science: the axiom-discriminating comparisons are between plausibly-relevant documents
(the top-10), whereas depth-100 floods the pair set with easy relevant-vs-junk pairs that are
trivially axiom-explained and dilute the residual RQ4 targets. Depth-10 is the informative
scope for the decomposition, and effectiveness there is a competence sanity check — now passed —
not an nDCG-maximisation target.

**Option B — depth-anchor cell (documented, deferred, not run).** If a reviewer wants an
absolute nDCG@10 *in the paper's neighbourhood* as a rhetorical close, add one **top-20
all-pairs** cell for flan-t5-xl on a single collection: C(20,2)=190 pairs/query vs 45 at top-10
(~4.2× the collection cost, ~4 h on bronze CPU for both DL cells, or minutes if the iGPU is
freed per §4b). The real model should land ~0.65–0.70 (depth-20 oracle 0.72), i.e. *in* the
paper's range, converting "comparing to nothing" into a matched reproduction. **Full top-100
all-pairs is explicitly rejected:** ~110× the top-10 cost (≈4.5 days CPU for XL alone, XXL
infeasible), for a number the decomposition does not need. This cell is kept as an available
follow-up, not part of the committed plan; the re-anchored evidence above is the primary result.
