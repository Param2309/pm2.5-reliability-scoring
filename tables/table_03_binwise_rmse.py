"""Table III. Stage 1 bin-wise RMSE on the held-out test rows."""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from config import DATA_DIR, RESULTS_DIR

BINS = [0, 10, 14, 28, 42, 66, 120, 180, 300, 400, 600, np.inf]
LABELS = ["0-10", "10-14", "14-28", "28-42", "42-66", "66-120", "120-180", "180-300", "300-400", "400-600", "600+"]

data = pd.read_csv(f"{DATA_DIR}/WildFireData.csv")
pred = pd.read_csv(f"{DATA_DIR}/rf_pred_all.csv")
_, test_idx = train_test_split(np.arange(len(data)), test_size=0.2, random_state=42)

test = pd.DataFrame({
    "ref": data["ref"].values[test_idx],
    "PA": data["PA"].values[test_idx],
    "Predicted_ref": pred["Predicted_ref"].values[test_idx],
})
test["PA_range"] = pd.cut(test["PA"], bins=BINS, labels=LABELS)

rows = []
for label in LABELS:
    sub = test[test["PA_range"] == label]
    if sub.empty:
        continue
    rmse = float(np.sqrt(mean_squared_error(sub["ref"], sub["Predicted_ref"])))
    rows.append({"Bin (µg/m³)": label, "RMSE": round(rmse), "n": len(sub)})

table = pd.DataFrame(rows)
print(table.to_string(index=False))
table.to_csv(f"{RESULTS_DIR}/table_03_binwise_rmse.csv", index=False)
