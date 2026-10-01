"""Table I. AQI-category distribution of the post-lag dataset."""
import numpy as np
import pandas as pd

from config import DATA_DIR, RESULTS_DIR
from stage2.pipeline import build_features

AQI_EDGES = [-np.inf, 35.4, 55.4, np.inf]
AQI_LABELS = ["Good + Moderate", "USG", "Unhealthy+"]

df = build_features(pd.read_csv(f"{DATA_DIR}/rf_pred_all.csv"))
groups = pd.cut(df["Predicted_ref"], bins=AQI_EDGES, labels=AQI_LABELS, right=True)
counts = groups.value_counts().reindex(AQI_LABELS)

table = pd.DataFrame({"Readings": counts, "% of Total": (counts / counts.sum() * 100).round(2)})
print(f"Rows: {len(df)}")
print(table.to_string())
table.to_csv(f"{RESULTS_DIR}/table_01_aqi_distribution.csv")
