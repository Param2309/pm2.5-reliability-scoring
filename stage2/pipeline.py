"""Sections IV-E to IV-I. Stage 2 reliability pipeline shared by all four Stage 1 candidates."""

from config import DATA_DIR, RESULTS_DIR

import numpy as np, pandas as pd, json, sys, time
from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
                              mean_absolute_error)
from sklearn.dummy import DummyClassifier
from xgboost import XGBClassifier
from scipy.stats import spearmanr

AQI_EDGES = [-np.inf, 35.4, np.inf]
AQI_LABELS = ["Non-smoky", "Smoky"]
NUM_FEATS = ["RH","PA","Predicted_ref","lag1_PA","dp_PA","lag1_pred","dp_pred","corr",
             "abs_dp_PA","abs_dp_pred","PA_roll_mean_3","PA_roll_std_3",
             "pred_roll_mean_3","pred_roll_std_3","PA_roll_mean_6","PA_roll_std_6",
             "pred_roll_mean_6","pred_roll_std_6"]
CAT_FEATS = ["AQI_group"]
THRESHOLDS = [0.5,0.7,0.8,0.9]
ECE_BINS = 15
RANDOM_STATE = 42


def ece_score(y_true, p, n_bins=15):
    bins = np.linspace(0,1,n_bins+1)
    ece, n = 0.0, len(p)
    for i in range(n_bins):
        lo,hi = bins[i],bins[i+1]
        mask = (p>=lo)&(p<hi) if i<n_bins-1 else (p>=lo)&(p<=hi)
        if mask.sum()==0: continue
        ece += (mask.sum()/n)*abs(y_true[mask].mean()-p[mask].mean())
    return float(ece)


def safe_auc(y,p): return float(roc_auc_score(y,p)) if len(np.unique(y))>1 else np.nan


def safe_prauc(y,p): return float(average_precision_score(y,p)) if len(np.unique(y))>1 else np.nan


def build_features(pred_df):
    df = pred_df.copy()
    df["hr"] = pd.to_datetime(df["hr"], errors="coerce")
    df = df.dropna(subset=["hr","RH","PA","id","ref","Predicted_ref"]).copy()
    df = df.sort_values(["id","hr"]).reset_index(drop=True)

    df["lag1_PA"] = df.groupby("id")["PA"].shift(1)
    df["dp_PA"] = df["PA"] - df["lag1_PA"]
    df["lag1_pred"] = df.groupby("id")["Predicted_ref"].shift(1)
    df["dp_pred"] = df["Predicted_ref"] - df["lag1_pred"]
    df["corr"] = df["Predicted_ref"] - df["PA"]
    df["abs_dp_PA"] = df["dp_PA"].abs()
    df["abs_dp_pred"] = df["dp_pred"].abs()

    for w in [3,6]:
        df[f"PA_roll_mean_{w}"] = df.groupby("id")["PA"].transform(lambda s: s.rolling(w,min_periods=1).mean())
        df[f"PA_roll_std_{w}"] = df.groupby("id")["PA"].transform(lambda s: s.rolling(w,min_periods=2).std())
        df[f"pred_roll_mean_{w}"] = df.groupby("id")["Predicted_ref"].transform(lambda s: s.rolling(w,min_periods=1).mean())
        df[f"pred_roll_std_{w}"] = df.groupby("id")["Predicted_ref"].transform(lambda s: s.rolling(w,min_periods=2).std())

    roll_std_cols = [c for c in df.columns if "roll_std" in c]
    df[roll_std_cols] = df[roll_std_cols].fillna(0.0)

    df["AQI_group"] = pd.cut(df["Predicted_ref"], bins=AQI_EDGES, labels=AQI_LABELS).astype(str).str.strip()
    df["abs_error"] = (df["ref"] - df["Predicted_ref"]).abs()

    need = NUM_FEATS + CAT_FEATS + ["abs_error"]
    df0 = df.dropna(subset=need).copy().reset_index(drop=True)
    return df0


def build_trust_model(xgb_params, random_state=42):
    pre = ColumnTransformer([("aqi_ohe", OneHotEncoder(handle_unknown="ignore"), ["AQI_group"])],
                             remainder="passthrough")
    clf = XGBClassifier(
        n_estimators=xgb_params.get("n_estimators",800),
        learning_rate=xgb_params.get("learning_rate",0.02),
        max_depth=xgb_params.get("max_depth",8),
        min_child_weight=xgb_params.get("min_child_weight",5),
        subsample=xgb_params.get("subsample",0.8),
        colsample_bytree=xgb_params.get("colsample_bytree",0.8),
        reg_lambda=xgb_params.get("reg_lambda",1.0),
        reg_alpha=xgb_params.get("reg_alpha",0.0),
        gamma=xgb_params.get("gamma",0.0),
        scale_pos_weight=xgb_params.get("scale_pos_weight",1.0),
        objective="binary:logistic", eval_metric="logloss", tree_method="hist",
        random_state=random_state, n_jobs=2
    )
    return Pipeline([("pre",pre),("clf",clf)])


def run_pipeline(model_name, pred_df, xgb_params, n_repeats=10, out_dir=RESULTS_DIR, smoky_delta_unhealthy_only=False):
    t0=time.time()
    df0 = build_features(pred_df)
    print(f"[{model_name}] rows after feature build: {len(df0)}, sensors: {df0['id'].nunique()}")

    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    idx_tc, idx_test0 = next(gss1.split(df0, groups=df0["id"]))
    traincal0 = df0.iloc[idx_tc]
    regime_stats = traincal0.groupby("AQI_group")["abs_error"].mean()
    AQI_DELTA_MAP = regime_stats.to_dict()
    if smoky_delta_unhealthy_only:
        AQI_DELTA_MAP["Smoky"] = float(traincal0.loc[traincal0["Predicted_ref"] > 55.4, "abs_error"].mean())
    print(f"[{model_name}] AQI_DELTA_MAP (mean abs error, train+cal, rep0): {AQI_DELTA_MAP}")

    df0["delta_used"] = df0["AQI_group"].map(AQI_DELTA_MAP).astype(float)
    df0["reliable"] = (df0["abs_error"] <= df0["delta_used"]).astype(int)

    raw_metrics = {"pr":[],"roc":[],"brier":[],"ece":[]}
    cal_metrics = {"pr":[],"roc":[],"brier":[],"ece":[]}
    dummy_metrics = {"pr":[],"roc":[],"brier":[]}
    threshold_results = {t: {"coverage":[],"mae_kept":[]} for t in THRESHOLDS}
    pareto_curve_runs = []
    binwise_runs = []
    spearman_runs = {"Non-smoky":[], "Smoky":[], "Overall":[]}

    REL_BIN_EDGES = np.linspace(0.0, 1.0, 11)
    reldiag_runs = {"raw": {g: [] for g in AQI_LABELS}, "cal": {g: [] for g in AQI_LABELS}}

    def bin_curve(score, y, group_mask):
        xs = np.full(10, np.nan); ys = np.full(10, np.nan)
        s = score[group_mask]; yy = y[group_mask]
        if len(s) == 0:
            return xs, ys
        idx = np.digitize(s, REL_BIN_EDGES[1:-1], right=False)
        for b in range(10):
            m = idx == b
            if m.sum() >= 3:
                xs[b] = s[m].mean()
                ys[b] = yy[m].mean()
        return xs, ys

    PARETO_THRESHOLDS = np.linspace(0.0,1.0,51)

    for rep in range(n_repeats):
        gss1r = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE+rep)
        idx_traincal, idx_test = next(gss1r.split(df0, groups=df0["id"]))
        df_traincal = df0.iloc[idx_traincal].copy()
        df_test = df0.iloc[idx_test].copy()

        gss2r = GroupShuffleSplit(n_splits=1, test_size=0.2/(1-0.2), random_state=RANDOM_STATE+100+rep)
        idx_train, idx_cal = next(gss2r.split(df_traincal, groups=df_traincal["id"]))
        df_train = df_traincal.iloc[idx_train].copy()
        df_cal = df_traincal.iloc[idx_cal].copy()

        X_train = df_train[CAT_FEATS+NUM_FEATS]; y_train = df_train["reliable"].values
        X_cal = df_cal[CAT_FEATS+NUM_FEATS]; y_cal = df_cal["reliable"].values
        X_test = df_test[CAT_FEATS+NUM_FEATS]; y_test = df_test["reliable"].values

        pipe = build_trust_model(xgb_params, random_state=RANDOM_STATE+rep)
        pipe.fit(X_train, y_train)

        p_raw_test = pipe.predict_proba(X_test)[:,1]
        p_raw_cal = pipe.predict_proba(X_cal)[:,1]

        dummy = DummyClassifier(strategy="prior")
        dummy.fit(np.zeros((len(y_train),1)), y_train)
        p_dummy = dummy.predict_proba(np.zeros((len(y_test),1)))[:,1]
        dummy_metrics["pr"].append(safe_prauc(y_test,p_dummy))
        dummy_metrics["roc"].append(safe_auc(y_test,p_dummy))
        dummy_metrics["brier"].append(brier_score_loss(y_test,p_dummy))

        eps=1e-6
        logit_cal = np.log(np.clip(p_raw_cal,eps,1-eps)/np.clip(1-p_raw_cal,eps,1-eps)).reshape(-1,1)
        platt = LogisticRegression(max_iter=2000)
        if len(np.unique(y_cal))<2:
            p_cal_test = p_raw_test.copy()
        else:
            platt.fit(logit_cal, y_cal)
            logit_test = np.log(np.clip(p_raw_test,eps,1-eps)/np.clip(1-p_raw_test,eps,1-eps)).reshape(-1,1)
            p_cal_test = platt.predict_proba(logit_test)[:,1]

        raw_metrics["pr"].append(safe_prauc(y_test,p_raw_test))
        raw_metrics["roc"].append(safe_auc(y_test,p_raw_test))
        raw_metrics["brier"].append(brier_score_loss(y_test,p_raw_test))
        raw_metrics["ece"].append(ece_score(y_test,p_raw_test,ECE_BINS))

        cal_metrics["pr"].append(safe_prauc(y_test,p_cal_test))
        cal_metrics["roc"].append(safe_auc(y_test,p_cal_test))
        cal_metrics["brier"].append(brier_score_loss(y_test,p_cal_test))
        cal_metrics["ece"].append(ece_score(y_test,p_cal_test,ECE_BINS))

        for t in THRESHOLDS:
            keep = p_cal_test>=t
            cov = float(keep.mean())
            mae = float(np.mean(np.abs(df_test.loc[keep,"ref"]-df_test.loc[keep,"Predicted_ref"]))) if keep.sum()>0 else np.nan
            threshold_results[t]["coverage"].append(cov)
            threshold_results[t]["mae_kept"].append(mae)

        cov_arr, mae_arr = [], []
        for thr in PARETO_THRESHOLDS:
            keep = p_cal_test>=thr
            cov_arr.append(float(keep.mean()))
            mae_arr.append(float(np.mean(np.abs(df_test.loc[keep,"ref"]-df_test.loc[keep,"Predicted_ref"]))) if keep.sum()>0 else np.nan)
        pareto_curve_runs.append((np.array(cov_arr), np.array(mae_arr)))

        out = df_test[["id","AQI_group","ref","Predicted_ref","abs_error"]].copy()
        out["trust_prob_cal"] = p_cal_test
        out["reliable"] = y_test
        out["trust_bin"] = pd.cut(out["trust_prob_cal"], bins=[0,0.5,0.6,0.7,0.8,0.9,1.0],
                                   labels=["0.0-0.5","0.5-0.6","0.6-0.7","0.7-0.8","0.8-0.9","0.9-1.0"],
                                   include_lowest=True)
        bw = out.groupby(["AQI_group","trust_bin"], observed=True).agg(
            mean_trust=("trust_prob_cal","mean"), actual_reliability=("reliable","mean"),
            mean_mae=("abs_error","mean"), samples=("abs_error","count")).reset_index()
        binwise_runs.append(bw)

        aqi_test = df_test["AQI_group"].values
        for grp in AQI_LABELS:
            gm = aqi_test == grp
            xs_raw, ys_raw = bin_curve(p_raw_test, y_test, gm)
            xs_cal, ys_cal = bin_curve(p_cal_test, y_test, gm)
            reldiag_runs["raw"][grp].append((xs_raw, ys_raw))
            reldiag_runs["cal"][grp].append((xs_cal, ys_cal))

        for grp in ["Non-smoky","Smoky"]:
            sub = out[out["AQI_group"]==grp]
            if len(sub)>=3:
                r,_ = spearmanr(sub["trust_prob_cal"], sub["abs_error"])
                spearman_runs[grp].append(r)
        r_all,_ = spearmanr(out["trust_prob_cal"], out["abs_error"])
        spearman_runs["Overall"].append(r_all)

        print(f"[{model_name}] rep {rep+1}/{n_repeats} done ({time.time()-t0:.0f}s elapsed)")

    def m(d): return {k: (float(np.nanmean(v)), float(np.nanstd(v))) for k,v in d.items()}

    results = {
        "model": model_name,
        "n_rows": int(len(df0)),
        "n_sensors": int(df0["id"].nunique()),
        "AQI_DELTA_MAP": {k: float(v) for k,v in AQI_DELTA_MAP.items()},
        "raw": m(raw_metrics), "cal": m(cal_metrics), "dummy": m(dummy_metrics),
        "threshold_sweep": {str(t): {"coverage": float(np.nanmean(v["coverage"])),
                                      "mae": float(np.nanmean(v["mae_kept"]))} for t,v in threshold_results.items()},
        "spearman": {k: (float(np.nanmean(v)) if len(v) else None) for k,v in spearman_runs.items()},
        "elapsed_sec": time.time()-t0
    }

    all_bw = pd.concat(binwise_runs, ignore_index=True)
    agg_bw = all_bw.groupby(["AQI_group","trust_bin"], observed=True).apply(
        lambda g: pd.Series({
            "mean_trust": np.average(g["mean_trust"], weights=g["samples"]) if g["samples"].sum()>0 else np.nan,
            "actual_reliability": np.average(g["actual_reliability"], weights=g["samples"]) if g["samples"].sum()>0 else np.nan,
            "mean_mae": np.average(g["mean_mae"], weights=g["samples"]) if g["samples"].sum()>0 else np.nan,
            "samples": g["samples"].sum()
        }), include_groups=False
    ).reset_index()
    results["binwise"] = agg_bw.to_dict(orient="records")
    results["binwise_last_repeat"] = binwise_runs[-1].astype({"trust_bin": str}).to_dict(orient="records")

    reldiag_out = {}
    for kind in ["raw","cal"]:
        reldiag_out[kind] = {}
        for grp in AQI_LABELS:
            xs_stack = np.stack([c[0] for c in reldiag_runs[kind][grp]])
            ys_stack = np.stack([c[1] for c in reldiag_runs[kind][grp]])
            reldiag_out[kind][grp] = {
                "x_mean": np.nanmean(xs_stack, axis=0).tolist(),
                "y_mean": np.nanmean(ys_stack, axis=0).tolist(),
                "y_std": np.nanstd(ys_stack, axis=0).tolist(),
                "n_valid": np.sum(~np.isnan(ys_stack), axis=0).tolist()
            }
    results["reldiag"] = reldiag_out

    cov_stack = np.stack([c[0] for c in pareto_curve_runs])
    mae_stack = np.stack([m[1] for m in pareto_curve_runs])
    results["pareto"] = {
        "coverage": np.nanmean(cov_stack, axis=0).tolist(),
        "mae_mean": np.nanmean(mae_stack, axis=0).tolist(),
        "mae_std": np.nanstd(mae_stack, axis=0).tolist()
    }

    with open(f"{out_dir}/{model_name}_results.json","w") as f:
        json.dump(results, f, indent=2)
    print(f"[{model_name}] TOTAL TIME: {time.time()-t0:.0f}s")
    return results
