"""Section V-J. Safety metrics under the doubly sensor-disjoint protocol."""

from config import DATA_DIR, RESULTS_DIR


def run_sensor_identity_leak_safety_metrics():
    import pandas as pd, numpy as np, json, time
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import roc_auc_score, average_precision_score
    from xgboost import XGBClassifier

    RAW_PATH = f"{DATA_DIR}/WildFireData.csv"
    AQI_EDGES = [-np.inf, 35.4, np.inf]
    AQI_LABELS = ["Non-smoky", "Smoky"]
    FULL_NUM_FEATS = ["RH","PA","Predicted_ref","lag1_PA","dp_PA","lag1_pred","dp_pred","corr",
                 "abs_dp_PA","abs_dp_pred","PA_roll_mean_3","PA_roll_std_3",
                 "pred_roll_mean_3","pred_roll_std_3","PA_roll_mean_6","PA_roll_std_6",
                 "pred_roll_mean_6","pred_roll_std_6"]
    CAT_FEATS = ["AQI_group"]
    RANDOM_STATE = 42
    CANONICAL_DELTA = {"Non-smoky": 0.981723249508968, "Smoky": 9.697103519851735}
    XGB_PARAMS = {'n_estimators': 1350, 'learning_rate': 0.029454474784661306, 'max_depth': 7,
                  'min_child_weight': 5.482646385370849, 'subsample': 0.7646717241827257,
                  'colsample_bytree': 0.9926123594067943, 'reg_lambda': 19.383944005992362,
                  'reg_alpha': 0.00010444341231664277, 'gamma': 2.202410950657256,
                  'scale_pos_weight': 3.898438563311057}
    THRESHOLDS = [0.5, 0.7, 0.8, 0.9]

    def safe_prauc(y,p): return float(average_precision_score(y,p)) if len(np.unique(y))>1 else np.nan
    def safe_auc(y,p): return float(roc_auc_score(y,p)) if len(np.unique(y))>1 else np.nan

    t0 = time.time()
    raw = pd.read_csv(RAW_PATH)
    raw["hr"] = pd.to_datetime(raw["hr"], errors="coerce")
    raw = raw.dropna(subset=["hr","RH","PA","id","ref"]).copy()
    raw["id"] = raw["id"].astype(str)

    pred_df = pd.read_csv(f"{DATA_DIR}/rf_pred_all.csv")
    pred_df["hr"] = pd.to_datetime(pred_df["hr"], errors="coerce")
    pred_df = pred_df.dropna(subset=["hr","RH","PA","id","ref","Predicted_ref"]).copy()
    pred_df = pred_df.sort_values(["id","hr"]).reset_index(drop=True)
    pred_df["lag1_PA"] = pred_df.groupby("id")["PA"].shift(1)
    pred_df["dp_PA"] = pred_df["PA"] - pred_df["lag1_PA"]
    pred_df["lag1_pred"] = pred_df.groupby("id")["Predicted_ref"].shift(1)
    pred_df["dp_pred"] = pred_df["Predicted_ref"] - pred_df["lag1_pred"]
    pred_df["corr"] = pred_df["Predicted_ref"] - pred_df["PA"]
    pred_df["abs_dp_PA"] = pred_df["dp_PA"].abs()
    pred_df["abs_dp_pred"] = pred_df["dp_pred"].abs()
    for w in [3,6]:
        pred_df[f"PA_roll_mean_{w}"] = pred_df.groupby("id")["PA"].transform(lambda s: s.rolling(w,min_periods=1).mean())
        pred_df[f"PA_roll_std_{w}"] = pred_df.groupby("id")["PA"].transform(lambda s: s.rolling(w,min_periods=2).std())
        pred_df[f"pred_roll_mean_{w}"] = pred_df.groupby("id")["Predicted_ref"].transform(lambda s: s.rolling(w,min_periods=1).mean())
        pred_df[f"pred_roll_std_{w}"] = pred_df.groupby("id")["Predicted_ref"].transform(lambda s: s.rolling(w,min_periods=2).std())
    roll_std_cols = [c for c in pred_df.columns if "roll_std" in c]
    pred_df[roll_std_cols] = pred_df[roll_std_cols].fillna(0.0)
    pred_df["AQI_group"] = pd.cut(pred_df["Predicted_ref"], bins=AQI_EDGES, labels=AQI_LABELS).astype(str).str.strip()
    pred_df["abs_error"] = (pred_df["ref"] - pred_df["Predicted_ref"]).abs()
    need = FULL_NUM_FEATS + CAT_FEATS + ["abs_error"]
    df0 = pred_df.dropna(subset=need).copy().reset_index(drop=True)
    print(f"skeleton rows: {len(df0)}", flush=True)

    def build_trust_model(random_state):
        pre = ColumnTransformer([("aqi_ohe", OneHotEncoder(handle_unknown="ignore"), CAT_FEATS)], remainder="passthrough")
        clf = XGBClassifier(**XGB_PARAMS, objective="binary:logistic", eval_metric="logloss",
                             tree_method="hist", random_state=random_state, n_jobs=2)
        return Pipeline([("pre",pre),("clf",clf)])

    n_repeats = 10
    records = []

    for rep in range(n_repeats):
        gss1r = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE+rep)
        idx_traincal, idx_test = next(gss1r.split(df0, groups=df0["id"]))
        df_traincal_skel = df0.iloc[idx_traincal].copy()
        df_test_skel = df0.iloc[idx_test].copy()
        gss2r = GroupShuffleSplit(n_splits=1, test_size=0.2/(1-0.2), random_state=RANDOM_STATE+100+rep)
        idx_train, idx_cal = next(gss2r.split(df_traincal_skel, groups=df_traincal_skel["id"]))
        df_train_skel = df_traincal_skel.iloc[idx_train].copy()
        df_cal_skel = df_traincal_skel.iloc[idx_cal].copy()

        train_sensors = set(df_train_skel["id"].unique())
        cal_sensors = set(df_cal_skel["id"].unique())
        test_sensors = set(df_test_skel["id"].unique())

        raw_train = raw[raw["id"].isin(train_sensors)].copy()
        X_tr = raw_train[["RH","PA","id"]]; y_tr = raw_train["ref"]
        ct = ColumnTransformer([("encoder", OneHotEncoder(handle_unknown="ignore"), ["id"])], remainder="passthrough")
        X_tr_enc = ct.fit_transform(X_tr)
        sc = StandardScaler(with_mean=False)
        X_tr_scaled = sc.fit_transform(X_tr_enc)
        rf1 = RandomForestRegressor(n_estimators=100, max_depth=None, random_state=RANDOM_STATE, n_jobs=2)
        rf1.fit(X_tr_scaled, y_tr)

        X_all = raw[["RH","PA","id"]]
        X_all_enc = ct.transform(X_all)
        X_all_scaled = sc.transform(X_all_enc)
        pred_all_clean = rf1.predict(X_all_scaled).astype(float)

        dfc = raw[["hr","RH","PA","id","ref"]].copy()
        dfc["Predicted_ref"] = pred_all_clean
        dfc = dfc.sort_values(["id","hr"]).reset_index(drop=True)
        dfc["lag1_PA"] = dfc.groupby("id")["PA"].shift(1)
        dfc["dp_PA"] = dfc["PA"] - dfc["lag1_PA"]
        dfc["lag1_pred"] = dfc.groupby("id")["Predicted_ref"].shift(1)
        dfc["dp_pred"] = dfc["Predicted_ref"] - dfc["lag1_pred"]
        dfc["corr"] = dfc["Predicted_ref"] - dfc["PA"]
        dfc["abs_dp_PA"] = dfc["dp_PA"].abs()
        dfc["abs_dp_pred"] = dfc["dp_pred"].abs()
        for w in [3,6]:
            dfc[f"PA_roll_mean_{w}"] = dfc.groupby("id")["PA"].transform(lambda s: s.rolling(w,min_periods=1).mean())
            dfc[f"PA_roll_std_{w}"] = dfc.groupby("id")["PA"].transform(lambda s: s.rolling(w,min_periods=2).std())
            dfc[f"pred_roll_mean_{w}"] = dfc.groupby("id")["Predicted_ref"].transform(lambda s: s.rolling(w,min_periods=1).mean())
            dfc[f"pred_roll_std_{w}"] = dfc.groupby("id")["Predicted_ref"].transform(lambda s: s.rolling(w,min_periods=2).std())
        rstd_cols = [c for c in dfc.columns if "roll_std" in c]
        dfc[rstd_cols] = dfc[rstd_cols].fillna(0.0)
        dfc["AQI_group"] = pd.cut(dfc["Predicted_ref"], bins=AQI_EDGES, labels=AQI_LABELS).astype(str).str.strip()
        dfc["abs_error"] = (dfc["ref"] - dfc["Predicted_ref"]).abs()
        dfc = dfc.dropna(subset=need).copy().reset_index(drop=True)
        dfc["delta_used"] = dfc["AQI_group"].map(CANONICAL_DELTA).astype(float)
        dfc["reliable"] = (dfc["abs_error"] <= dfc["delta_used"]).astype(int)

        df_train = dfc[dfc["id"].isin(train_sensors)].copy()
        df_cal = dfc[dfc["id"].isin(cal_sensors)].copy()
        df_test = dfc[dfc["id"].isin(test_sensors)].copy()

        cols = CAT_FEATS + FULL_NUM_FEATS
        y_train = df_train["reliable"].values; y_cal = df_cal["reliable"].values; y_test = df_test["reliable"].values
        X_train = df_train[cols]; X_cal = df_cal[cols]; X_test = df_test[cols]

        pipe = build_trust_model(random_state=RANDOM_STATE+rep)
        pipe.fit(X_train, y_train)
        p_raw_test = pipe.predict_proba(X_test)[:,1]
        p_raw_cal = pipe.predict_proba(X_cal)[:,1]
        eps=1e-6
        logit_cal = np.log(np.clip(p_raw_cal,eps,1-eps)/np.clip(1-p_raw_cal,eps,1-eps)).reshape(-1,1)
        platt = LogisticRegression(max_iter=2000)
        if len(np.unique(y_cal))<2:
            p_cal_test = p_raw_test.copy()
        else:
            platt.fit(logit_cal, y_cal)
            logit_test = np.log(np.clip(p_raw_test,eps,1-eps)/np.clip(1-p_raw_test,eps,1-eps)).reshape(-1,1)
            p_cal_test = platt.predict_proba(logit_test)[:,1]

        prevalence = float(y_test.mean())
        dummy_pr_auc = prevalence
        model_pr_auc = safe_prauc(y_test, p_cal_test)
        model_roc_auc = safe_auc(y_test, p_cal_test)


        thresh_rows = []
        for t in THRESHOLDS:
            kept = p_cal_test >= t
            coverage = float(kept.mean())
            if kept.sum() > 0:
                far = float((y_test[kept] == 0).mean())
                actual_rel = float(y_test[kept].mean())
            else:
                far = np.nan; actual_rel = np.nan
            thresh_rows.append({"threshold": t, "coverage": coverage, "far": far, "actual_reliability": actual_rel})

        records.append({
            "rep": rep, "n_test": int(len(df_test)),
            "prevalence_reliable": prevalence, "dummy_pr_auc": dummy_pr_auc,
            "model_pr_auc": model_pr_auc, "model_roc_auc": model_roc_auc,
            "thresholds": thresh_rows,
        })
        print(f"rep {rep+1}/{n_repeats} done ({time.time()-t0:.0f}s) prevalence={prevalence:.4f} "
              f"model_pr={model_pr_auc:.4f} dummy_pr={dummy_pr_auc:.4f}", flush=True)

    prevs = np.array([r["prevalence_reliable"] for r in records])
    dummies = np.array([r["dummy_pr_auc"] for r in records])
    models = np.array([r["model_pr_auc"] for r in records])
    rocs = np.array([r["model_roc_auc"] for r in records])

    summary = {
        "per_repeat": records,
        "prevalence_reliable": {"mean": float(prevs.mean()), "std": float(prevs.std())},
        "dummy_pr_auc": {"mean": float(dummies.mean()), "std": float(dummies.std())},
        "model_pr_auc": {"mean": float(models.mean()), "std": float(models.std())},
        "model_roc_auc": {"mean": float(rocs.mean()), "std": float(rocs.std())},
    }
    for t in THRESHOLDS:
        covs = np.array([next(x["coverage"] for x in r["thresholds"] if x["threshold"]==t) for r in records])
        fars = np.array([next(x["far"] for x in r["thresholds"] if x["threshold"]==t) for r in records])
        actrel = np.array([next(x["actual_reliability"] for x in r["thresholds"] if x["threshold"]==t) for r in records])
        summary[f"threshold_{t}"] = {
            "coverage_mean": float(np.nanmean(covs)), "coverage_std": float(np.nanstd(covs)),
            "far_mean": float(np.nanmean(fars)), "far_std": float(np.nanstd(fars)),
            "actual_reliability_mean": float(np.nanmean(actrel)), "actual_reliability_std": float(np.nanstd(actrel)),
        }

    print(json.dumps({k:v for k,v in summary.items() if k!="per_repeat"}, indent=2), flush=True)
    json.dump(summary, open(f"{RESULTS_DIR}/rf_doubly_disjoint_safety_metrics_results.json","w"), indent=2)
    print("TOTAL TIME", time.time()-t0, flush=True)
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    run_sensor_identity_leak_safety_metrics()
