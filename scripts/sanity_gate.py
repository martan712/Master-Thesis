"""4-way order-swap sanity gate for pairwise ranker candidates (phase0-plan §8.2).

Two obvious relevant/irrelevant pairs, each shown in both presentation orders. A model
passes only if it picks the relevant passage in all four presentations — anything else
(most commonly always answering "A") is position bias, which disqualifies the model from
full collection. flan-t5-small failed exactly this way.

Usage:
    uv run python scripts/sanity_gate.py --config configs/pilot.yaml [--model NAME]
"""

import argparse

from axiomrank.config import load_config
from axiomrank.rankers import make_ranker

CASES = [
    (
        "what causes rainbows",
        "Rainbows appear when sunlight is refracted, dispersed and reflected inside "
        "raindrops, splitting white light into its component colours.",
        "The 2008 financial crisis began with the collapse of the subprime mortgage "
        "market in the United States.",
    ),
    (
        "how to bake sourdough bread",
        "Mix flour, water, salt and an active sourdough starter, let the dough ferment "
        "and rise, then bake it in a hot Dutch oven until the crust is deep brown.",
        "Jupiter is the largest planet in the solar system, with a mass more than twice "
        "that of all other planets combined.",
    ),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="YAML whose ranker section to gate")
    parser.add_argument("--model", help="override ranker.model from the config")
    args = parser.parse_args()

    cfg = load_config(args.config).ranker
    if args.model:
        cfg.model = args.model
    ranker = make_ranker(cfg)
    print(f"gating {ranker.name} (backend={cfg.backend}, prompt={cfg.prompt_version})\n")

    correct = 0
    for query, relevant, irrelevant in CASES:
        for first, second, want in ((relevant, irrelevant, "a"), (irrelevant, relevant, "b")):
            v = ranker.compare(query, first, second)
            ok = v.verdict == want
            correct += ok
            print(
                f"  [{'ok' if ok else 'XX'}] {query!r}: relevant shown as "
                f"{'A' if want == 'a' else 'B'} -> verdict={v.verdict} prob_a={v.prob_a:.3f}"
            )

    passed = correct == 2 * len(CASES)
    print(f"\n{'PASS' if passed else 'FAIL'}: {correct}/{2 * len(CASES)} presentations correct")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
