"""Stage 2 internal tables: aggregate metrics, threshold sweep, bin-wise reliability (Non-smoky, Smoky) and delta values."""
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
    ("Aggregate metrics", table_ix, "stage2_aggregate_metrics.csv"),
    ("Threshold sweep", table_x, "stage2_threshold_sweep.csv"),
    ("Bin-wise, Non-smoky", table_xi, "stage2_binwise_nonsmoky.csv"),
    ("Bin-wise, Smoky", table_xii, "stage2_binwise_smoky.csv"),
    ("Delta values", table_xiii, "stage2_delta_values.csv"),
]:
    print(f"\n{title}")
    print(table.to_string(index=False))
    table.to_csv(f"{RESULTS_DIR}/{fname}", index=False)
