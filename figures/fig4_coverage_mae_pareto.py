"""Fig. 4. Stage 2 coverage-MAE Pareto curves for all four Stage 1 candidates."""
import json

import matplotlib.pyplot as plt
import numpy as np

from config import RESULTS_DIR

CANDIDATES = [("RF", "Random Forest"), ("LR", "Linear Regression"), ("NN", "Neural Network"), ("BJ", "Barkjohn")]
PARETO_THRESHOLDS = np.linspace(0.0, 1.0, 51)

fig, axes = plt.subplots(1, len(CANDIDATES), figsize=(22, 5), dpi=150)

for ax, (key, title) in zip(axes, CANDIDATES):
    with open(f"{RESULTS_DIR}/{key}_results.json") as f:
        pareto = json.load(f)["pareto"]
    coverage = np.array(pareto["coverage"], dtype=float)
    mae_mean = np.array(pareto["mae_mean"], dtype=float)
    mae_std = np.array(pareto["mae_std"], dtype=float)

    ax.plot(coverage, mae_mean, linewidth=2)
    ax.fill_between(coverage, mae_mean - mae_std, mae_mean + mae_std, alpha=0.2)
    for thr in [0.5, 0.7, 0.8, 0.9]:
        idx = np.argmin(np.abs(PARETO_THRESHOLDS - thr))
        ax.scatter(coverage[idx], mae_mean[idx], s=40)
        ax.text(coverage[idx], mae_mean[idx], f"  t={thr:.1f}", va="center")
    ax.set_title(title)
    ax.set_xlabel("Coverage (fraction of readings kept)")
    ax.set_ylabel("MAE on kept samples")
    ax.grid(alpha=0.3)

fig.suptitle("Coverage-MAE Pareto Curve (average over 10 unseen-sensor splits)")
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/fig4_coverage_mae_pareto.png")
