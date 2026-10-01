"""Fig. 1. Random Forest Stage 1 calibration on held-out test rows."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import DATA_DIR, RESULTS_DIR

pred = pd.read_csv(f"{DATA_DIR}/rf_pred_all.csv")
_, test_idx = train_test_split(np.arange(len(pred)), test_size=0.2, random_state=42)
test = pred.iloc[test_idx]

err_raw = (test["ref"] - test["PA"]).abs()
err_cal = (test["ref"] - test["Predicted_ref"]).abs()

fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150, gridspec_kw={"width_ratios": [2, 1]})

ax = axes[0]
ax.scatter(test["ref"], test["Predicted_ref"], s=2, alpha=0.4)
ax.plot([0, 100], [0, 100], color="tab:blue")
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.set_xlabel("EPA ref (µg/m³)")
ax.set_ylabel("Calibrated pm_cal (µg/m³)")
ax.set_title("Calibrated pm_cal vs EPA (held-out test rows)")
ax.grid(alpha=0.3)

ax = axes[1]
ax.hist(err_raw, bins=50, range=(0, 250), alpha=0.6, label="raw PA - ref")
ax.hist(err_cal, bins=50, range=(0, 250), alpha=0.6, label="pm_cal - ref")
ax.set_xlabel("Absolute error (µg/m³)")
ax.set_ylabel("Count")
ax.set_title("Calibration reduces absolute error")
ax.legend()

plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/fig1_rf_calibration.png")
