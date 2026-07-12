# Phase 3 Untouched Confirmation Protocol

## Status

The external confirmation set is **locked as of 2026-07-13**. The machine-readable contract is
[`phase3-confirmation-lock.yaml`](phase3-confirmation-lock.yaml). No query, document, qrel, BM25
pool, LLM preference or result from the locked dataset has been loaded in the project.

The locked collection is `beir/nfcorpus/test`, using all official test queries. The choice was
made from public metadata only. `ir_datasets` describes NFCorpus as a BEIR test collection with
323 queries and roughly 3.6K documents, and a prebuilt Terrier index is available. This makes a
complete BM25 top-10 all-pairs confirmation operationally feasible without selecting queries by
their content or qrels. See the official [`ir_datasets` BEIR catalogue](https://ir-datasets.com/beir.html)
and the [PyTerrier index catalogue](https://huggingface.co/pyterrier).

## Why this is separate from development

DL19 and DL20 are fully contaminated for confirmation: aggregate results, residual profiles,
individual passages and successful reversals have all been inspected. Query-grouped CV estimates
internal development generalisation but cannot undo hypothesis-development reuse.

NFCorpus is a cross-domain test. A positive result supports external transfer of the frozen axiom
battery. A null result is not, by itself, proof that the axiom is invalid within MS MARCO because
domain transfer is part of the challenge. An optional later TREC DL21 study can test a closer
task family, but it is not the primary confirmation set and cannot replace this locked result.

## No-touch boundary

Until an explicit unlock commit, the following are prohibited:

- importing or displaying holdout queries or documents;
- importing, counting or summarising holdout qrels;
- building or loading its BM25 run or index;
- collecting any LLM or axiom preferences;
- computing descriptive dataset statistics;
- testing candidate definitions, thresholds, prompts or code paths on the collection.

Public catalogue metadata and identifier/schema validation performed before this lock are not
outcome access. No local dataset API was called during selection.

## Freeze before unlock

Before confirmation, one versioned development manifest must contain:

1. retained candidate definitions, directions, preconditions, neutral cases and margins;
2. feature extractor and dependency versions;
3. development-only hyperparameter selection;
4. serialized classical and extended coefficients fitted on DL19/DL20;
5. family membership and Holm-adjusted secondary ablations;
6. exact ranker, prompts, order handling, aggregation and tie rules;
7. primary fidelity and effectiveness contrasts; and
8. a hash of the complete frozen manifest.

The confirmation runner must first pass a full dry run on non-holdout data. Unlocking then requires
a dedicated commit that creates `configs/confirmation/UNLOCKED.yaml` and records the frozen hash.

## One-shot analysis

The frozen development models are applied to the holdout **without refitting on holdout LLM
labels or qrels**. The two primary tests are:

- extended minus classical pairwise log-loss improvement on stable, decisive LLM preferences;
- extended minus classical nDCG@10 improvement.

Both use paired query-bootstrap 95% intervals and succeed only when the interval is above zero.
Fidelity and effectiveness remain separate claims. Results are written and reported regardless of
direction. After the first result read, the collection is no longer untouched; any modification
motivated by it creates a new exploratory analysis, not a second confirmation attempt.
