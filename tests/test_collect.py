"""Cache-only collection behavior used by analysis phases."""

import pandas as pd
import pytest

from axiomrank.config import RankerConfig
from axiomrank.data.preferences import PreferenceStore
from axiomrank.pipeline.collect import collect_verdicts


def test_cache_only_collection_rejects_missing_presentations(tmp_path):
    pairs = pd.DataFrame(
        {
            "qid": ["q1"],
            "query": ["query"],
            "doc_id_1": ["d1"],
            "doc_id_2": ["d2"],
            "text_1": ["one"],
            "text_2": ["two"],
        }
    )
    cfg = RankerConfig(backend="mock", order_swap=True)
    with pytest.raises(RuntimeError, match="2/2 presentations missing"):
        collect_verdicts("ds", cfg, pairs, PreferenceStore(tmp_path), allow_new=False)
