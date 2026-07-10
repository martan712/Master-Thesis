import pandas as pd
import pytest

from axiomrank.preferences import ALL_COLUMNS, PreferenceStore, new_row


def row(query_id="q1", doc_a="d1", doc_b="d2", verdict="a", model="m"):
    return new_row(
        dataset="ds",
        query_id=query_id,
        doc_id_a=doc_a,
        doc_id_b=doc_b,
        model=model,
        prompt_version="v0",
        verdict=verdict,
        prob_a=0.9,
        score_a=-1.0,
        score_b=-2.0,
        latency_ms=3.0,
    )


def test_append_and_load_roundtrip(tmp_path):
    store = PreferenceStore(tmp_path)
    store.append(pd.DataFrame([row(), row(doc_a="d2", doc_b="d1", verdict="b")]))
    df = store.load()
    assert len(df) == 2
    assert list(df.columns) == ALL_COLUMNS


def test_dedup_keeps_first_write(tmp_path):
    store = PreferenceStore(tmp_path)
    first = row(verdict="a")
    store.append(pd.DataFrame([first]))
    later = row(verdict="b")
    later["created_at"] = "9999-01-01T00:00:00+00:00"
    store.append(pd.DataFrame([later]))
    df = store.load()
    assert len(df) == 1
    assert df["verdict"].iloc[0] == "a"


def test_filters(tmp_path):
    store = PreferenceStore(tmp_path)
    store.append(pd.DataFrame([row(model="m1"), row(query_id="q2", model="m2")]))
    assert len(store.load(model="m1")) == 1
    assert len(store.load(dataset="other")) == 0


def test_rejects_bad_verdict_and_missing_columns(tmp_path):
    store = PreferenceStore(tmp_path)
    bad = row(verdict="unsure")
    with pytest.raises(ValueError, match="verdict"):
        store.append(pd.DataFrame([bad]))
    with pytest.raises(ValueError, match="missing"):
        store.append(pd.DataFrame([{"dataset": "ds"}]))
