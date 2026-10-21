"""Shared helpers for the safety tests. Read-only on the repo; everything is written to safety/."""
import numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

import os
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
DATA = os.path.join(REPO, "data")
SAFE = HERE

AQI_EDGES = [-np.inf, 35.4, np.inf]
AQI_LABELS = ["Non-smoky", "Smoky"]
NUM_FEATS = ["RH", "PA", "Predicted_ref", "lag1_PA", "dp_PA", "lag1_pred", "dp_pred", "corr",
             "abs_dp_PA", "abs_dp_pred", "PA_roll_mean_3", "PA_roll_std_3",
             "pred_roll_mean_3", "pred_roll_std_3", "PA_roll_mean_6", "PA_roll_std_6",
             "pred_roll_mean_6", "pred_roll_std_6"]
CAT_FEATS = ["AQI_group"]
COLS = CAT_FEATS + NUM_FEATS
RS = 42
CANON_DELTA = {"Non-smoky": 0.981723249508968, "Smoky": 9.697103519851735}
XGB_RF = {'n_estimators': 1350, 'learning_rate': 0.029454474784661306, 'max_depth': 7,
          'min_child_weight': 5.482646385370849, 'subsample': 0.7646717241827257,
          'colsample_bytree': 0.9926123594067943, 'reg_lambda': 19.383944005992362,
          'reg_alpha': 0.00010444341231664277, 'gamma': 2.202410950657256,
          'scale_pos_weight': 3.898438563311057}
# PA-only temporal features for the isolation forest (no Stage 1 output used)
ISO_FEATS = ["PA", "RH", "lag1_PA", "dp_PA", "abs_dp_PA", "PA_roll_mean_3", "PA_roll_std_3",
             "PA_roll_mean_6", "PA_roll_std_6"]
# EPA 2018 AQI technical assistance document PM2.5 breakpoints
AQI6_EDGES = [-np.inf, 12.0, 35.4, 55.4, 150.4, 250.4, np.inf]


def add_features(d, pred_col=None):
    """Same feature construction as stage2/pipeline.py (sorted by id, hr)."""
    d = d.copy()
    if pred_col is not None:
        d["Predicted_ref"] = d[pred_col]
    d = d.sort_values(["id", "hr"]).reset_index(drop=True)
    d["lag1_PA"] = d.groupby("id")["PA"].shift(1)
    d["dp_PA"] = d["PA"] - d["lag1_PA"]
    d["lag1_pred"] = d.groupby("id")["Predicted_ref"].shift(1)
    d["dp_pred"] = d["Predicted_ref"] - d["lag1_pred"]
    d["corr"] = d["Predicted_ref"] - d["PA"]
    d["abs_dp_PA"] = d["dp_PA"].abs()
    d["abs_dp_pred"] = d["dp_pred"].abs()
    for w in [3, 6]:
        d[f"PA_roll_mean_{w}"] = d.groupby("id")["PA"].transform(lambda s: s.rolling(w, min_periods=1).mean())
        d[f"PA_roll_std_{w}"] = d.groupby("id")["PA"].transform(lambda s: s.rolling(w, min_periods=2).std())
        d[f"pred_roll_mean_{w}"] = d.groupby("id")["Predicted_ref"].transform(lambda s: s.rolling(w, min_periods=1).mean())
        d[f"pred_roll_std_{w}"] = d.groupby("id")["Predicted_ref"].transform(lambda s: s.rolling(w, min_periods=2).std())
    rstd = [c for c in d.columns if "roll_std" in c]
    d[rstd] = d[rstd].fillna(0.0)
    d["AQI_group"] = pd.cut(d["Predicted_ref"], bins=AQI_EDGES, labels=AQI_LABELS).astype(str).str.strip()
    d["abs_error"] = (d["ref"] - d["Predicted_ref"]).abs()
    return d


def make_labels(d, kind, delta_map=None):
    """Reliability label under different definitions. Returns int array."""
    if kind == "delta":
        dm = d["AQI_group"].map(delta_map).astype(float)
        return (d["abs_error"] <= dm).astype(int).values
    if kind == "rel50":  # within +-50 percent of the reference (ref floored at 1 ug/m3 to avoid /0)
        return (d["abs_error"] <= 0.5 * np.maximum(d["ref"], 1.0)).astype(int).values
    if kind == "aqi6":  # same EPA AQI category for prediction and reference
        a = pd.cut(d["Predicted_ref"], bins=AQI6_EDGES, labels=False)
        b = pd.cut(d["ref"], bins=AQI6_EDGES, labels=False)
        return (a == b).astype(int).values
    raise ValueError(kind)


def trust_model(params, seed):
    pre = ColumnTransformer([("aqi_ohe", OneHotEncoder(handle_unknown="ignore"), CAT_FEATS)], remainder="passthrough")
    p = dict(params)
    clf = XGBClassifier(**p, objective="binary:logistic", eval_metric="logloss", tree_method="hist",
                        random_state=seed, n_jobs=2)
    return Pipeline([("pre", pre), ("clf", clf)])


def platt_fit_apply(p_raw_cal, y_cal, p_raw_list):
    eps = 1e-6
    lg = lambda p: np.log(np.clip(p, eps, 1 - eps) / np.clip(1 - p, eps, 1 - eps)).reshape(-1, 1)
    if len(np.unique(y_cal)) < 2:
        return [p.copy() for p in p_raw_list]
    pl = LogisticRegression(max_iter=2000).fit(lg(p_raw_cal), y_cal)
    return [pl.predict_proba(lg(p))[:, 1] for p in p_raw_list]


def fit_score(df_train, df_cal, df_test, ytr, ycal, params, seed):
    """XGBoost + Platt exactly as the paper's Stage 2. Returns (p_cal_on_cal, p_cal_on_test)."""
    pipe = trust_model(params, seed)
    pipe.fit(df_train[COLS], ytr)
    pr_cal = pipe.predict_proba(df_cal[COLS])[:, 1]
    pr_test = pipe.predict_proba(df_test[COLS])[:, 1]
    pc_cal, pc_test = platt_fit_apply(pr_cal, ycal, [pr_cal, pr_test])
    return pc_cal, pc_test
