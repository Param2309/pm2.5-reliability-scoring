"""Tables IX-XIII. Internal Stage 2 metrics, threshold sweep, bin-wise reliability and delta values."""
import json

import pandas as pd

from config import RESULTS_DIR

CANDIDATES = [("RF", "Random Forest"), ("LR", "Linear Regression"), ("NN", "Neural Network"), ("BJ", "Barkjohn")]


def load(key):
    with open(f"{RESULTS_DIR}/{key}_results.json") as f:
        return json.load(f)


results = {key: load(key) for key, _ in CANDIDATES}

metrics = []
for key, name in CANDIDATES:
    r = results[key]
    for kind in ("raw", "cal"):
        row = {"Candidate": name, "Score": kind.upper()}
        for m in ("pr", "roc", "brier", "ece"):
            mean, std = r[kind][m]
            row[m] = f"{mean:.3f} ± {std:.3f}"
        metrics.append(row)
    row = {"Candidate": name, "Score": "Dummy"}
    for m in ("pr", "roc", "brier"):
        mean, std = r["dummy"][m]
        row[m] = f"{mean:.3f} ± {std:.3f}"
    metrics.append(row)
table_ix = pd.DataFrame(metrics)

sweep = []
for key, name in CANDIDATES:
    for thr, v in results[key]["threshold_sweep"].items():
        sweep.append({"Candidate": name, "Threshold": thr, "Coverage": round(v["coverage"], 2), "MAE": round(v["mae"], 3)})
table_x = pd.DataFrame(sweep)

binwise = pd.DataFrame(results["RF"]["binwise_last_repeat"]).round(3)
table_xi = binwise[binwise["AQI_group"] == "Non-smoky"]
table_xii = binwise[binwise["AQI_group"] == "Smoky"]

table_xiii = pd.DataFrame([
    {"Candidate": name, "delta(Non-smoky)": round(results[key]["AQI_DELTA_MAP"]["Non-smoky"], 3),
     "delta(Smoky)": round(results[key]["AQI_DELTA_MAP"]["Smoky"], 3)}
    for key, name in CANDIDATES
])

for title, table, fname in [
    ("Table IX", table_ix, "table_09_stage2_metrics.csv"),
    ("Table X", table_x, "table_10_threshold_sweep.csv"),
    ("Table XI", table_xi, "table_11_binwise_nonsmoky.csv"),
    ("Table XII", table_xii, "table_12_binwise_smoky.csv"),
    ("Table XIII", table_xiii, "table_13_delta_values.csv"),
]:
    print(f"\n{title}")
    print(table.to_string(index=False))
    table.to_csv(f"{RESULTS_DIR}/{fname}", index=False)
