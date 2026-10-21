"""Nested (leakage-free) tuning of the Stage 2 XGBoost for the PurpleAir reliability paper.
Only the tuning step changes versus the paper. Per repeat and regime:
  Optuna TPE (seed 42), N_TRIALS trials, searched on the 15 TRAINING sensors only;
  objective = mean PR-AUC of 3-fold GroupKFold (by sensor id) inside those 15 sensors.
  Calibration and test sensors are never read during tuning.
  Then refit on all 15 training sensors with the best params, Platt on the 6 calibration sensors, score the 6 test sensors.
Arms per cell: old = paper hyperparameters, new = nested-tuned. Same splits (GroupShuffleSplit seeds 42+rep, 42+100+rep),
features, labels and delta as before. The original study's Optuna sampler was unseeded; here TPE seed is fixed to 42.

Regimes: R1 (frozen Stage 1 predictions from rf_pred_all.csv, canonical delta), R3 (cold start, Stage 1 refit on the
training sensors without sensor id, delta re-derived per repeat), R2 (known sensors, Stage 1 on first 80% rows per sensor,
Stage 2 pool = future 20% rows; tuning on the training sensors' future rows = the pool Stage 2 is fit on).
Usage:  python nested_tune.py --regime R1|R3|R2 [--abl] [--reps 0-9] [--trials 25]
Resumable: a repeat whose json already exists in OUT_DIR is skipped."""
import os, json, time, argparse
import numpy as np, pandas as pd, optuna

# ======================= PATHS (auto-detected, nothing to edit) =======================
# Looks for the two data files in the script folder, the current folder and their subfolders (and the parent folder).
# Results go to a folder "nested_out" next to this script. To override, set env vars WILDFIRE_CSV, RFPRED_CSV, NESTED_OUT.
def _find(name):
    here = os.path.dirname(os.path.abspath(__file__))
    for root in (here, os.getcwd(), os.path.dirname(here), os.path.expanduser("~")):
        for dp, dn, fn in os.walk(root):
            if name in fn: return os.path.join(dp, name)
            if dp.count(os.sep) - root.count(os.sep) >= 4: dn[:] = []
    raise SystemExit(f"Cannot find {name}. Put it in this folder (or a subfolder) or set the env var.")
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_RAW = os.environ.get("WILDFIRE_CSV") or _find("WildFireData.csv")   # raw sensor data (hr, RH, PA, id, ref)
DATA_PRED = os.environ.get("RFPRED_CSV") or _find("rf_pred_all.csv")     # frozen Stage 1 RF predictions (Predicted_ref)
OUT_DIR = os.environ.get("NESTED_OUT") or os.path.join(HERE, "nested_out")
# There are no pkl files: the frozen Stage 1 is rf_pred_all.csv; R2/R3 refit their own Stage 1 inside this script.
# ======================================================================================
N_TRIALS = 25      # Optuna trials per (repeat, regime); matches the original search
N_FOLDS = 3        # grouped CV folds inside the 15 training sensors
TPE_SEED = 42
N_JOBS = -1        # all CPU cores for XGBoost and the random forest

from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
from xgboost import XGBClassifier
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---- shared helpers (copied from the project's common code) ----
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


def platt_fit_apply(p_raw_cal, y_cal, p_raw_list):
    eps = 1e-6
    lg = lambda p: np.log(np.clip(p, eps, 1 - eps) / np.clip(1 - p, eps, 1 - eps)).reshape(-1, 1)
    if len(np.unique(y_cal)) < 2:
        return [p.copy() for p in p_raw_list]
    pl = LogisticRegression(max_iter=2000).fit(lg(p_raw_cal), y_cal)
    return [pl.predict_proba(lg(p))[:, 1] for p in p_raw_list]



# ---- tuning and evaluation ----
ABL_COLS = ["PA", "RH", "lag1_PA", "dp_PA", "abs_dp_PA", "PA_roll_mean_3", "PA_roll_std_3",
            "PA_roll_mean_6", "PA_roll_std_6"]          # no Predicted_ref-derived features, no AQI_group
NEED = NUM_FEATS + CAT_FEATS + ["abs_error"]

def suggest(tr):   # same search space as the original run
    return dict(n_estimators=tr.suggest_int("n_estimators", 500, 1600, step=50),
        learning_rate=tr.suggest_float("learning_rate", 0.005, 0.08, log=True),
        max_depth=tr.suggest_int("max_depth", 3, 8),
        min_child_weight=tr.suggest_float("min_child_weight", 3.0, 30.0, log=True),
        subsample=tr.suggest_float("subsample", 0.6, 1.0), colsample_bytree=tr.suggest_float("colsample_bytree", 0.6, 1.0),
        reg_lambda=tr.suggest_float("reg_lambda", 0.1, 20.0, log=True), reg_alpha=tr.suggest_float("reg_alpha", 1e-4, 3.0, log=True),
        gamma=tr.suggest_float("gamma", 0.0, 5.0), scale_pos_weight=tr.suggest_float("scale_pos_weight", 0.5, 5.0, log=True))

def build(cols, params, seed):
    clf = XGBClassifier(**params, objective="binary:logistic", eval_metric="logloss", tree_method="hist",
                        random_state=seed, n_jobs=N_JOBS)
    if "AQI_group" in cols:
        pre = ColumnTransformer([("aqi_ohe", OneHotEncoder(handle_unknown="ignore"), ["AQI_group"])], remainder="passthrough")
        return Pipeline([("pre", pre), ("clf", clf)])
    return Pipeline([("clf", clf)])

def ece_score(y, p, n_bins=15):   # same as stage2/pipeline.py
    bins = np.linspace(0, 1, n_bins + 1); ece, n = 0.0, len(p)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        if m.sum() == 0: continue
        ece += (m.sum() / n) * abs(y[m].mean() - p[m].mean())
    return float(ece)

def tune_nested(t, cols, seed, n_trials):
    """t = tuning rows (training sensors only). 3-fold GroupKFold by sensor id, mean PR-AUC."""
    folds = list(GroupKFold(n_splits=N_FOLDS).split(t, groups=t["id"])); y = t["reliable"].values
    hist = []
    def obj(tr):
        P = suggest(tr); sc = []
        for tri, vai in folds:
            if len(np.unique(y[tri])) < 2 or len(np.unique(y[vai])) < 2: continue
            m = build(cols, P, seed); m.fit(t.iloc[tri][cols], y[tri])
            sc.append(average_precision_score(y[vai], m.predict_proba(t.iloc[vai][cols])[:, 1]))
        v = float(np.mean(sc)) if sc else 0.0; hist.append(v); return v
    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=TPE_SEED))
    st.optimize(obj, n_trials=n_trials)
    return st.best_params, float(st.best_value), hist

def fit_eval(a, b, c, cols, params, seed):
    ya, yb, yc = a["reliable"].values, b["reliable"].values, c["reliable"].values
    m = build(cols, params, seed); m.fit(a[cols], ya)
    pc, pt = platt_fit_apply(m.predict_proba(b[cols])[:, 1], yb, [m.predict_proba(b[cols])[:, 1], m.predict_proba(c[cols])[:, 1]])
    return dict(pr=float(average_precision_score(yc, pt)), base=float(yc.mean()), roc=float(roc_auc_score(yc, pt)),
                brier=float(brier_score_loss(yc, pt)), ece=ece_score(yc, pt), n_test=int(len(yc)))

def run_arms(tune_rows, a, b, c, rep, n_trials, mode):
    # leak guard: tuning rows must come from training sensors only
    ids_a = set(a["id"]); assert set(tune_rows["id"]) <= ids_a
    assert not (ids_a & set(b["id"])) and not (ids_a & set(c["id"])) and not (set(b["id"]) & set(c["id"]))
    seed = RS + rep; out = {"rep": rep, "n_train_sensors": len(ids_a), "n_cal_sensors": b["id"].nunique(), "n_test_sensors": c["id"].nunique()}
    arms = [("full", COLS)] if mode == "full" else [("abl", ABL_COLS)]
    for nm, cols in arms:
        t1 = time.time()
        best, cv, hist = tune_nested(tune_rows, cols, seed, n_trials)
        out[nm] = {"old": fit_eval(a, b, c, cols, XGB_RF, seed), "new": fit_eval(a, b, c, cols, best, seed),
                   "best": best, "cv_pr": cv, "trial_cv": hist, "tune_rows": int(len(tune_rows)), "tune_sec": time.time() - t1}
    return out

# ---------- frames ----------
def read_skeleton():
    p = pd.read_csv(DATA_PRED); p["hr"] = pd.to_datetime(p["hr"], errors="coerce"); p["id"] = p["id"].astype(str)
    return add_features(p.dropna(subset=["hr", "RH", "PA", "id", "ref", "Predicted_ref"])).dropna(subset=NEED).reset_index(drop=True)

def read_raw():
    raw = pd.read_csv(DATA_RAW); raw["hr"] = pd.to_datetime(raw["hr"], errors="coerce")
    raw = raw.dropna(subset=["hr", "RH", "PA", "id", "ref"]).copy(); raw["id"] = raw["id"].astype(str); return raw

def split3(df, rep):   # identical to the paper's split
    itc, ite = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RS + rep).split(df, groups=df["id"]))
    tc = df.iloc[itc]; te = df.iloc[ite]
    itr, ica = next(GroupShuffleSplit(n_splits=1, test_size=0.2 / 0.8, random_state=RS + 100 + rep).split(tc, groups=tc["id"]))
    return tc.iloc[itr], tc.iloc[ica], te

def rf_fit(rows):
    sc = StandardScaler(); return sc, RandomForestRegressor(n_estimators=100, max_depth=None, random_state=RS, n_jobs=N_JOBS).fit(sc.fit_transform(rows[["RH", "PA"]].values), rows["ref"].values)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--regime", required=True); ap.add_argument("--reps", default="0-9")
    ap.add_argument("--trials", type=int, default=N_TRIALS); ap.add_argument("--out", default=OUT_DIR); ap.add_argument("--abl", action="store_true", help="ablated model (9 PA/RH features)")
    A = ap.parse_args(); lo, hi = [int(x) for x in A.reps.split("-")]; os.makedirs(A.out, exist_ok=True); mode = "abl" if A.abl else "full"
    T0 = time.time(); reg = A.regime
    if reg == "R1":
        df0 = read_skeleton(); df0["reliable"] = make_labels(df0, "delta", CANON_DELTA)
    elif reg == "R3":
        df0 = read_skeleton(); raw = read_raw()
    elif reg == "R2":
        raw = read_raw().sort_values(["id", "hr"]).reset_index(drop=True)
        raw["row_in_sensor"] = raw.groupby("id").cumcount(); cnt = raw.groupby("id")["id"].transform("count")
        raw["is_future"] = raw["row_in_sensor"] >= np.ceil(0.8 * cnt).astype(int)
        hist = raw[~raw["is_future"]]
        ct = ColumnTransformer([("encoder", OneHotEncoder(handle_unknown="ignore"), ["id"])], remainder="passthrough")
        sc = StandardScaler(with_mean=False); Xh = sc.fit_transform(ct.fit_transform(hist[["RH", "PA", "id"]]))
        rf = RandomForestRegressor(n_estimators=100, max_depth=None, random_state=RS, n_jobs=N_JOBS).fit(Xh, hist["ref"].values)
        raw["Predicted_ref"] = rf.predict(sc.transform(ct.transform(raw[["RH", "PA", "id"]]))).astype(float)
        dfc = add_features(raw[["hr", "RH", "PA", "id", "ref", "Predicted_ref", "is_future"]]).dropna(subset=NEED).reset_index(drop=True)
        dfc["reliable"] = make_labels(dfc, "delta", CANON_DELTA)
        df0 = dfc[dfc["is_future"]].reset_index(drop=True)   # Stage 2 pool: future rows, as in the paper's R2
    else: raise SystemExit("regime must be R1, R3 or R2")
    print(reg, "rows", len(df0), "sensors", df0["id"].nunique(), flush=True)
    for rep in range(lo, hi + 1):
        fn = f"{A.out}/{reg}{'abl' if A.abl else ''}_rep{rep}.json"
        if os.path.exists(fn): print("skip", fn, flush=True); continue
        a, b, c = split3(df0, rep)
        if reg == "R1":
            res = run_arms(a, a, b, c, rep, A.trials, mode)
        elif reg == "R2":
            res = run_arms(a, a, b, c, rep, A.trials, mode)
        else:   # R3: Stage 1 refit on the 15 training sensors, no sensor id; delta re-derived on training sensors
            tr, ca, ts = set(a["id"]), set(b["id"]), set(c["id"])
            s1, rf1 = rf_fit(raw[raw["id"].isin(tr)]); base = raw[["hr", "RH", "PA", "id", "ref"]].copy()
            base["Predicted_ref"] = rf1.predict(s1.transform(raw[["RH", "PA"]].values)).astype(float)
            d = add_features(base).dropna(subset=NEED).reset_index(drop=True)
            dd = d[d["id"].isin(tr)].groupby("AQI_group")["abs_error"].mean().to_dict()
            for l in AQI_LABELS:
                if l not in dd or pd.isna(dd[l]): dd[l] = CANON_DELTA[l]
            d["reliable"] = make_labels(d, "delta", dd)
            a3, b3, c3 = d[d["id"].isin(tr)], d[d["id"].isin(ca)], d[d["id"].isin(ts)]
            res = run_arms(a3, a3, b3, c3, rep, A.trials, mode); res["delta"] = {k: float(v) for k, v in dd.items()}
        res["regime"] = reg; res["trials"] = A.trials; res["mode"] = mode
        json.dump(res, open(fn, "w"))
        f = res[mode]
        print(f"{reg}{mode if A.abl else ''} rep {rep} t={time.time()-T0:.0f}s tune={f['tune_sec']:.0f}s cv={f['cv_pr']:.4f} PR old={f['old']['pr']:.4f} new={f['new']['pr']:.4f}"
             , flush=True)
    print("DONE", reg, flush=True)

if __name__ == "__main__":
    main()
