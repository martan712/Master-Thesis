"""Hugging Face backend: PRP-style binary choice, scored by label likelihood.

Instead of free generation we compare the model's log-likelihood of the two label
continuations ("Passage A" / "Passage B"), following the scoring mode of PRP (Qin et al.
2024). This is deterministic, cannot produce unparseable output, and yields a probability
(needed as a confidence signal for RQ6). Works for seq2seq (Flan-T5) and causal models.

Requires the `llm` extra: `uv sync --extra llm`.
"""

from dataclasses import dataclass, field

from axiomrank import paths
from axiomrank.rankers.base import PairVerdict, PairwiseRanker

PROMPTS = {
    "v0": (
        'Given a query "{query}", which of the following two passages is more relevant '
        "to the query?\n\n"
        "Passage A: {passage_a}\n\n"
        "Passage B: {passage_b}\n\n"
        "Output Passage A or Passage B:"
    ),
}
LABELS = ("Passage A", "Passage B")


@dataclass
class HFPairwiseRanker(PairwiseRanker):
    model_name: str
    prompt_version: str = "v0"
    max_chars: int = 2000
    device: str = "cpu"
    _bundle: tuple = field(init=False, repr=False, default=None)

    @property
    def name(self) -> str:
        return self.model_name

    def _load(self):
        if self._bundle is None:
            paths.configure_caches()
            try:
                import torch
                from transformers import (
                    AutoConfig,
                    AutoModelForCausalLM,
                    AutoModelForSeq2SeqLM,
                    AutoTokenizer,
                )
            except ImportError as e:
                raise ImportError(
                    "The hf ranker backend needs torch/transformers: uv sync --extra llm"
                ) from e
            config = AutoConfig.from_pretrained(self.model_name)
            cls = AutoModelForSeq2SeqLM if config.is_encoder_decoder else AutoModelForCausalLM
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = cls.from_pretrained(self.model_name).to(self.device).eval()
            self._bundle = (torch, tokenizer, model, config.is_encoder_decoder)
        return self._bundle

    def _label_loglik(self, prompt: str, label: str) -> float:
        torch, tokenizer, model, is_seq2seq = self._load()
        with torch.no_grad():
            if is_seq2seq:
                enc = tokenizer(prompt, return_tensors="pt", truncation=True).to(self.device)
                target = tokenizer(text_target=label, return_tensors="pt").to(self.device)
                logits = model(**enc, labels=target.input_ids).logits
                logprobs = logits.log_softmax(-1)
                token_ids = target.input_ids[0]
            else:
                prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids
                full_ids = tokenizer(prompt + " " + label, return_tensors="pt").input_ids
                full_ids = full_ids.to(self.device)
                logits = model(full_ids).logits
                start = prompt_ids.shape[1]
                logprobs = logits[:, start - 1 : -1].log_softmax(-1)
                token_ids = full_ids[0, start:]
            return float(logprobs[0, range(len(token_ids)), token_ids].sum())

    def compare(self, query: str, passage_a: str, passage_b: str) -> PairVerdict:
        torch, *_ = self._load()
        prompt = PROMPTS[self.prompt_version].format(
            query=query,
            passage_a=passage_a[: self.max_chars],
            passage_b=passage_b[: self.max_chars],
        )
        score_a = self._label_loglik(prompt, LABELS[0])
        score_b = self._label_loglik(prompt, LABELS[1])
        prob_a = float(torch.tensor([score_a, score_b]).softmax(0)[0])
        if score_a == score_b:
            return PairVerdict("tie", 0.5, score_a, score_b)
        return PairVerdict("a" if score_a > score_b else "b", prob_a, score_a, score_b)
