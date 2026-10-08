"""Optional reject-performance analysis plot.

Lifted from ``annotate_func.py:338-382``. matplotlib is imported LAZILY inside
:func:`plot_reject_performance` so that importing this module (or the pipeline)
never pulls in matplotlib. The orchestrator never calls this function.
"""
from __future__ import annotations
import numpy as np


def plot_reject_performance(all_rejection_scores: dict,
                            all_cycle_checking_results: dict,
                            consis_summary: dict,
                            out_path: str) -> None:
    """Plot consistent/inconsistent rejection curves over 20 thresholds.

    Ranks samples by mean rejection score, then, for each cycle-checking
    outcome, finds the rank of each sample and accumulates rejection counts
    across the threshold sweep. The result is normalized by the consistent /
    inconsistent sample counts in ``consis_summary`` and saved to ``out_path``.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Rank all samples according to their rejection scores.
    samples_to_sort = [
        [f"{traj_name}_{step}", np.mean(score).item()]
        for traj_name, traj_steps_scores in all_rejection_scores.items()
        for step, score in traj_steps_scores.items()
    ]
    samples_to_sort.sort(key=lambda x: x[1])

    # Plot rejection performance curves.
    thresholds = np.linspace(0, 1, 20)
    num_consis_rejected = np.zeros((len(thresholds),))
    num_inconsis_rejected = np.zeros((len(thresholds),))

    for traj_name, cycle_checking_results in all_cycle_checking_results.items():
        for step_idx in cycle_checking_results["result"]["consistent"]:
            rank = 0
            while samples_to_sort[rank][0] != f"{traj_name}_{step_idx}":
                rank += 1
            proper_idx = np.argmax(rank / len(samples_to_sort) < thresholds)
            num_consis_rejected[proper_idx:] += 1

        for step_idx in cycle_checking_results["result"]["inconsistent"]:
            rank = 0
            while samples_to_sort[rank][0] != f"{traj_name}_{step_idx}":
                rank += 1
            proper_idx = np.argmax(rank / len(samples_to_sort) < thresholds)
            num_inconsis_rejected[proper_idx:] += 1

    num_consis_rejected_normalzied = num_consis_rejected / consis_summary["num_consis"]
    num_inconsis_rejected_normalzied = num_inconsis_rejected / consis_summary["num_inconsis"]

    plt.figure()
    plt.plot(thresholds, num_consis_rejected_normalzied, label='consis')
    plt.plot(thresholds, num_inconsis_rejected_normalzied, label='inconsis')
    plt.plot([0.4, 0.4], [0.0, 1.0], linestyle='--', color='k', label='Threshold')
    plt.title(f"{len(samples_to_sort)} samples in total")
    plt.xlabel('threshold')
    plt.ylabel('ratio')
    plt.legend()
    plt.savefig(out_path)
