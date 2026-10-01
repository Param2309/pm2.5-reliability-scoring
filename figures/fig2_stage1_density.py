"""Fig. 2. Predicted vs. actual reference PM2.5 density for the Linear Regression and neural network Stage 1 candidates."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

from config import RESULTS_DIR

PANELS = [
    ("lr_pred_all.csv", "Density Plot: Predicted vs Actual ref (Linear Regression)"),
    ("nn_pred_all.csv", "Density Plot: Predicted vs Actual ref (Neural Network)"),
]

fig, axes = plt.subplots(1, 2, figsize=(20, 6), dpi=100)

for ax, (filename, title) in zip(axes, PANELS):
    pred = pd.read_csv(f"{RESULTS_DIR}/{filename}")
    _, test_idx = train_test_split(np.arange(len(pred)), test_size=0.2, random_state=42)
    test = pred.iloc[test_idx]
    sns.kdeplot(test["ref"], label="Actual Values (ref)", fill=True, alpha=0.5, color="blue", ax=ax)
    sns.kdeplot(test["Predicted_ref"], label="Predicted Values (ref)", fill=True, alpha=0.5, color="red", ax=ax)
    ax.set_xlabel("PM 2.5 Concentration (µg/m³)", fontsize=14)
    ax.set_ylabel("Density", fontsize=14)
    ax.set_title(title)
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/fig2_stage1_density.png")
