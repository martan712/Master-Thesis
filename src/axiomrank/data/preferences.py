"""Append-only Parquet store for LLM pairwise verdicts (data/preferences/).

One row per *presentation*: doc_id_a was shown first, doc_id_b second. Rows are never
edited or deleted; each flush writes a new part file and duplicates are resolved on read
by keeping the earliest write. This store is the reusable core artefact of the thesis —
treat it as expensive.
"""

import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from axiomrank import paths

KEY_COLUMNS = ["dataset", "query_id", "doc_id_a", "doc_id_b", "model", "prompt_version"]
VALUE_COLUMNS = ["verdict", "prob_a", "score_a", "score_b", "latency_ms", "created_at"]
ALL_COLUMNS = KEY_COLUMNS + VALUE_COLUMNS

VERDICTS = ("a", "b", "tie")


class PreferenceStore:
    def __init__(self, root: Path | None = None):
        self.root = Path(root) if root is not None else paths.PREFERENCES_DIR
        self.root.mkdir(parents=True, exist_ok=True)

    def _part_files(self) -> list[Path]:
        return sorted(self.root.glob("part-*.parquet"))

    def load(
        self,
        dataset: str | None = None,
        model: str | None = None,
        prompt_version: str | None = None,
    ) -> pd.DataFrame:
        """All stored verdicts, deduplicated (first write wins), optionally filtered."""
        parts = self._part_files()
        if not parts:
            return pd.DataFrame(columns=ALL_COLUMNS)
        df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
        df = df.sort_values("created_at", kind="stable")
        df = df.drop_duplicates(subset=KEY_COLUMNS, keep="first", ignore_index=True)
        for column, value in (
            ("dataset", dataset),
            ("model", model),
            ("prompt_version", prompt_version),
        ):
            if value is not None:
                df = df[df[column] == value]
        return df.reset_index(drop=True)

    def append(self, rows: pd.DataFrame) -> Path:
        """Write new verdicts as a fresh part file; never touches existing files."""
        missing = set(ALL_COLUMNS) - set(rows.columns)
        if missing:
            raise ValueError(f"Preference rows missing columns: {sorted(missing)}")
        bad = set(rows["verdict"].unique()) - set(VERDICTS)
        if bad:
            raise ValueError(f"Invalid verdict values: {sorted(bad)}")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        path = self.root / f"part-{stamp}-{uuid.uuid4().hex[:8]}.parquet"
        rows[ALL_COLUMNS].to_parquet(path, index=False)
        return path


def new_row(
    *,
    dataset: str,
    query_id: str,
    doc_id_a: str,
    doc_id_b: str,
    model: str,
    prompt_version: str,
    verdict: str,
    prob_a: float,
    score_a: float,
    score_b: float,
    latency_ms: float,
) -> dict:
    return {
        "dataset": dataset,
        "query_id": query_id,
        "doc_id_a": doc_id_a,
        "doc_id_b": doc_id_b,
        "model": model,
        "prompt_version": prompt_version,
        "verdict": verdict,
        "prob_a": prob_a,
        "score_a": score_a,
        "score_b": score_b,
        "latency_ms": latency_ms,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
