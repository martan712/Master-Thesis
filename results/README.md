# results/ — not in git

One subdirectory per experiment, created via `axiomrank.paths.results_dir(<experiment>)`,
named after the experiment directory that produced it (e.g. `rq1_lexical_agreement/`).

Inside an experiment directory the convention is:

- `runs/` — TREC run files
- `metrics/` — evaluation output (CSV/JSON)
- `figures/` — plots
- `tables/` — LaTeX/markdown tables destined for the thesis
- `config.yaml` — a copy of the exact config the run used (written by the script)

Final numbers that appear in the thesis get copied into `thesis/` when the manuscript is
written; everything here is regenerable from `data/` + the config.
