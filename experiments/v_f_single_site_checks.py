"""Sections V-F and V-I. Single-sensor checks: Hoopa (in-distribution consistency check) and Los Angeles January 2025 (boundary condition)."""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from config import DATA_DIR, RESULTS_DIR

AQI_EDGES = [-np.inf, 35.4, np.inf]
AQI_LABELS = ["Non-smoky", "Smoky"]
AQI_DELTA_MAP = {"Non-smoky": 0.981723249508968, "Smoky": 9.697103519851735}
NUM_FEATS = ["RH", "PA", "Predicted_ref", "lag1_PA", "dp_PA", "lag1_pred", "dp_pred", "corr",
             "abs_dp_PA", "abs_dp_pred", "PA_roll_mean_3", "PA_roll_std_3",
             "pred_roll_mean_3", "pred_roll_std_3", "PA_roll_mean_6", "PA_roll_std_6",
             "pred_roll_mean_6", "pred_roll_std_6"]
CAT_FEATS = ["AQI_group"]
THRESHOLDS = [0.5, 0.7, 0.8, 0.9]


SITES = {
    "Hoopa": "rf_pred_Hoopa.csv",
    "LA_Wildfire_Jan2025": "rf_pred_LA.csv",
}


def run_single_site_evaluation(site):
    trust_pipe = joblib.load(f"{DATA_DIR}/trust_pipe.pkl")
    platt = joblib.load(f"{DATA_DIR}/platt.pkl")

    la = pd.read_csv(f"{DATA_DIR}/{SITES[site]}")
    la = la[["hr", "id", "RH", "PA", "ref", "Predicted_ref"]].copy()
    la["hr"] = pd.to_datetime(la["hr"], utc=True, errors="coerce")
    la["RH"] = la["RH"].astype(float)
    la["PA"] = la["PA"].astype(float)
    la["ref"] = la["ref"].astype(float)
    la["id"] = la["id"].astype(str)
    la = la.dropna(subset=["hr", "RH", "PA", "ref", "Predicted_ref"]).copy()
    la = la.sort_values("hr").reset_index(drop=True)

    mae_calib = np.mean(np.abs(la["ref"] - la["Predicted_ref"]))
    rmse_calib = np.sqrt(np.mean((la["ref"] - la["Predicted_ref"]) ** 2))
    r2_calib = 1 - np.sum((la["ref"] - la["Predicted_ref"]) ** 2) / np.sum((la["ref"] - la["ref"].mean()) ** 2)
    print(f"[{site}] Stage 1: MAE={mae_calib:.4f} RMSE={rmse_calib:.4f} R2={r2_calib:.4f}")

    la = la.sort_values(["id", "hr"]).reset_index(drop=True)
    la["lag1_PA"] = la.groupby("id")["PA"].shift(1)
    la["dp_PA"] = la["PA"] - la["lag1_PA"]
    la["lag1_pred"] = la.groupby("id")["Predicted_ref"].shift(1)
    la["dp_pred"] = la["Predicted_ref"] - la["lag1_pred"]
    la["corr"] = la["Predicted_ref"] - la["PA"]
    la["abs_dp_PA"] = la["dp_PA"].abs()
    la["abs_dp_pred"] = la["dp_pred"].abs()

    for w in [3, 6]:
        la[f"PA_roll_mean_{w}"] = la.groupby("id")["PA"].transform(lambda s: s.rolling(window=w, min_periods=1).mean())
        la[f"PA_roll_std_{w}"] = la.groupby("id")["PA"].transform(lambda s: s.rolling(window=w, min_periods=2).std())
        la[f"pred_roll_mean_{w}"] = la.groupby("id")["Predicted_ref"].transform(lambda s: s.rolling(window=w, min_periods=1).mean())
        la[f"pred_roll_std_{w}"] = la.groupby("id")["Predicted_ref"].transform(lambda s: s.rolling(window=w, min_periods=2).std())

    roll_std_cols = [c for c in la.columns if "roll_std" in c]
    la[roll_std_cols] = la[roll_std_cols].fillna(0.0)
    la["AQI_group"] = pd.cut(la["Predicted_ref"], bins=AQI_EDGES, labels=AQI_LABELS).astype(str).str.strip()
    la = la.dropna(subset=["lag1_PA", "dp_PA", "lag1_pred", "dp_pred"]).copy()
    la = la.reset_index(drop=True)

    p_raw = trust_pipe.predict_proba(la[CAT_FEATS + NUM_FEATS])[:, 1]
    eps = 1e-6
    logit_raw = np.log(np.clip(p_raw, eps, 1 - eps) / np.clip(1 - p_raw, eps, 1 - eps)).reshape(-1, 1)
    p_cal = platt.predict_proba(logit_raw)[:, 1]
    la["trust_prob_raw"] = p_raw
    la["trust_prob_cal"] = p_cal

    la["delta_used"] = la["AQI_group"].map(AQI_DELTA_MAP).astype(float)
    la["abs_error"] = np.abs(la["ref"] - la["Predicted_ref"])
    la["reliable"] = (la["abs_error"] <= la["delta_used"]).astype(int)

    y_true = la["reliable"].values
    result = {
        "n_rows": int(len(la)),
        "stage1": {"mae": float(mae_calib), "rmse": float(rmse_calib), "r2": float(r2_calib)},
        "reliable_rate": float(y_true.mean()),
        "baseline_mae": float(la["abs_error"].mean()),
        "pr_auc": float(average_precision_score(y_true, p_cal)),
        "roc_auc": float(roc_auc_score(y_true, p_cal)),
        "brier": float(brier_score_loss(y_true, p_cal)),
        "reliable_rate_by_group": la.groupby("AQI_group")["reliable"].mean().to_dict(),
        "threshold_sweep": {},
    }
    for thr in THRESHOLDS:
        mask = la["trust_prob_cal"] >= thr
        mae_kept = float(np.mean(np.abs(la.loc[mask, "ref"] - la.loc[mask, "Predicted_ref"]))) if mask.sum() > 0 else None
        result["threshold_sweep"][str(thr)] = {"coverage": float(mask.mean()), "n_kept": int(mask.sum()), "mae": mae_kept}

    print(json.dumps(result, indent=2))
    json.dump(result, open(f"{RESULTS_DIR}/{site}_results.json", "w"), indent=2)


if __name__ == "__main__":
    for site in SITES:
        run_single_site_evaluation(site)
