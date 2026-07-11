"""OpenAI-compatible API backend (vLLM et al.): PRP-style choice from token logprobs.

One chat completion per presentation, temperature 0, max_tokens 1: the verdict and
`prob_a` are derived from the top logprobs at the first generated position, comparing the
"A" and "B" continuations — the API-server analogue of the label-likelihood scoring in
`rankers/hf.py`. Reasoning/thinking modes must be disabled server-side or via
`extra_body` (e.g. vLLM's `chat_template_kwargs: {"enable_thinking": false}`), otherwise
the first token is a thinking token and the scoring silently degenerates to ties.
"""

import math
import os
from dataclasses import dataclass, field

from axiomrank.rankers.base import PairVerdict, PairwiseRanker

PROMPTS = {
    "v1": (
        'Given a query "{query}", which of the following two passages is more relevant '
        "to the query?\n\n"
        "Passage A: {passage_a}\n\n"
        "Passage B: {passage_b}\n\n"
        "Answer with a single letter, A or B:"
    ),
}
MISSING = float("-inf")  # label token absent from the top logprobs


def verdict_from_top_logprobs(entries: list[tuple[str, float]]) -> PairVerdict:
    """Score the A/B choice from (token, logprob) pairs at one generation position.

    Tokenisers vary in whitespace/casing of the single-letter continuation, so tokens
    are stripped and upper-cased; the best-scoring variant of each label wins. If
    neither label is present the presentation is an unparseable refusal -> tie.
    """
    scores = {"A": MISSING, "B": MISSING}
    for token, logprob in entries:
        label = token.strip().upper()
        if label in scores:
            scores[label] = max(scores[label], logprob)
    score_a, score_b = scores["A"], scores["B"]
    if score_a == MISSING and score_b == MISSING:
        return PairVerdict("tie", 0.5, MISSING, MISSING)
    if score_a == score_b:
        return PairVerdict("tie", 0.5, score_a, score_b)
    m = max(score_a, score_b)
    exp_a, exp_b = math.exp(score_a - m), math.exp(score_b - m)
    prob_a = exp_a / (exp_a + exp_b)
    return PairVerdict("a" if score_a > score_b else "b", prob_a, score_a, score_b)


@dataclass
class OpenAIPairwiseRanker(PairwiseRanker):
    model_name: str
    base_url: str
    prompt_version: str = "v1"
    max_chars: int = 2000
    top_logprobs: int = 20
    extra_body: dict | None = None  # e.g. {"chat_template_kwargs": {"enable_thinking": false}}
    _client: object = field(init=False, repr=False, default=None)

    @property
    def name(self) -> str:
        return self.model_name

    def _load(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                base_url=self.base_url,
                api_key=os.environ.get("OPENAI_API_KEY", "unused"),
                max_retries=3,
            )
        return self._client

    def compare(self, query: str, passage_a: str, passage_b: str) -> PairVerdict:
        prompt = PROMPTS[self.prompt_version].format(
            query=query,
            passage_a=passage_a[: self.max_chars],
            passage_b=passage_b[: self.max_chars],
        )
        response = self._load().chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=1,
            logprobs=True,
            top_logprobs=self.top_logprobs,
            extra_body=self.extra_body or None,
        )
        content = response.choices[0].logprobs.content
        if not content:
            return PairVerdict("tie", 0.5, MISSING, MISSING)
        entries = [(t.token, t.logprob) for t in content[0].top_logprobs]
        return verdict_from_top_logprobs(entries)
