"""Figure drafts for the analysis outputs (matplotlib, Agg backend)."""

import pandas as pd


def gap_figure(gradient: pd.DataFrame, path, title: str) -> None:
    """Draft of the RQ1 signature figure: agreement vs. rank gap."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(8, 5))
    context = gradient.drop_duplicates("gap_bin").sort_values("gap_bin")
    for axiom, group in gradient.groupby("axiom"):
        group = group.sort_values("gap_bin")
        axis.plot(group["gap_bin"], group["agreement"], color="0.75", lw=1, zorder=1)
    if "joint_cv_accuracy" in gradient.columns:
        axis.plot(
            context["gap_bin"], context["joint_cv_accuracy"],
            color="C0", lw=2.5, label="joint logistic (CV acc.)", zorder=3,
        )
    axis.plot(
        context["gap_bin"], context["position_consistency"],
        color="C3", lw=1.5, ls="--", label="position consistency", zorder=2,
    )
    axis.axhline(0.5, color="0.4", lw=0.8, ls=":")
    axis.set_xlabel("BM25 rank gap (bin)")
    axis.set_ylabel("agreement / accuracy")
    axis.set_ylim(0, 1)
    axis.set_title(title)
    axis.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
