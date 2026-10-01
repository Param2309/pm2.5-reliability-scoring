"""Fig. 3. Stage 2 reliability diagrams by AQI group for all four Stage 1 candidates."""
import json

import matplotlib.pyplot as plt
import numpy as np

from config import RESULTS_DIR

CANDIDATES = [("RF", "Random Forest"), ("LR", "Linear Regression"), ("NN", "Neural Network"), ("BJ", "Barkjohn")]
AQI_LABELS = ["Non-smoky", "Smoky"]
REL_BIN_CENTERS = np.linspace(0.05, 0.95, 10)

fig, axes = plt.subplots(len(CANDIDATES), len(AQI_LABELS), figsize=(14, 16), dpi=150, sharey=True)

for row, (key, title) in enumerate(CANDIDATES):
    with open(f"{RESULTS_DIR}/{key}_results.json") as f:
        reldiag = json.load(f)["reldiag"]
    for col, group in enumerate(AQI_LABELS):
        ax = axes[row, col]
        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="gray", label="Perfect")
        for kind, marker, label in [("raw", "o", "RAW"), ("cal", "s", "CAL")]:
            d = reldiag[kind][group]
            x = np.array(d["x_mean"], dtype=float)
            y = np.array(d["y_mean"], dtype=float)
            s = np.array(d["y_std"], dtype=float)
            ax.plot(x, y, marker=marker, label=label)
            ax.fill_between(x, y - s, y + s, alpha=0.2)
        ax.set_title(group if row == 0 else "")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        if col == 0:
            ax.set_ylabel(f"{title}\nObserved reliable rate")
        if row == len(CANDIDATES) - 1:
            ax.set_xlabel("Mean predicted trust probability")

fig.suptitle("Reliability Diagram by AQI Group (average over 10 unseen-sensor splits)")
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/fig3_reliability_diagrams.png")
