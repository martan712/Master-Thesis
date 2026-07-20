# data/ — not in git

Created on demand by `axiomrank.paths`. Layout:

- `cache/` — library-managed caches (`ir_datasets/`, `pyterrier/`, `huggingface/`).
  Fully regenerable; safe to delete.
- `raw/` — manually downloaded inputs that no library manages. Record the source URL in
  a note next to each file.
- `preferences/` — **the cached LLM pairwise-verdict dataset** (Parquet). Expensive to
  recompute (thousands of LLM calls): back this up off-machine. Append-only; never edit
  rows in place. Key: (dataset, query_id, doc_id_a, doc_id_b, model, prompt_version,
  presentation_order).
- `processed/` — derived intermediates (sampled pairs, axiom feature matrices).
  Regenerable from cache + preferences given the config and seed.
- `backups/` — verified local snapshots made by `scripts/snapshot_artifacts.py`; `LATEST.json`
  records the most recent checksum and restoration test.

The append-only preference store and current results can be snapshotted and verified with:

```bash
uv run --no-sync python scripts/snapshot_artifacts.py create
uv run --no-sync python scripts/snapshot_artifacts.py verify data/backups/<snapshot>
uv run --no-sync python scripts/snapshot_artifacts.py restore-test data/backups/<snapshot>
```

Snapshots under `data/backups/` protect against accidental local corruption, but a durable thesis
archive should also live on a different device or managed backup service.
