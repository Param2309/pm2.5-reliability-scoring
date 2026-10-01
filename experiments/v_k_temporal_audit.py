"""Section V-K. Historical-versus-future temporal audit."""

from config import DATA_DIR, RESULTS_DIR


def run_temporal_historical_future_leak_audit():
    import pandas as pd, numpy as np, json, time
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import roc_auc_score, average_precision_score
    from xgboost import XGBClassifier
    from scipy import stats

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
    HIST_FRAC = 0.8
    XGB_PARAMS = {'n_estimators': 1350, 'learning_rate': 0.029454474784661306, 'max_depth': 7,
                  'min_child_weight': 5.482646385370849, 'subsample': 0.7646717241827257,
                  'colsample_bytree': 0.9926123594067943, 'reg_lambda': 19.383944005992362,
                  'reg_alpha': 0.00010444341231664277, 'gamma': 2.202410950657256,
                  'scale_pos_weight': 3.898438563311057}

    def safe_prauc(y,p): return float(average_precision_score(y,p)) if len(np.unique(y))>1 else np.nan
    def safe_auc(y,p): return float(roc_auc_score(y,p)) if len(np.unique(y))>1 else np.nan

    t0 = time.time()
    raw = pd.read_csv(RAW_PATH)
    raw["hr"] = pd.to_datetime(raw["hr"], errors="coerce")
    raw = raw.dropna(subset=["hr","RH","PA","id","ref"]).copy()
    raw["id"] = raw["id"].astype(str)
    raw = raw.sort_values(["id","hr"]).reset_index(drop=True)
    print(f"raw rows: {len(raw)}, sensors: {raw['id'].nunique()}", flush=True)

    raw["row_in_sensor"] = raw.groupby("id").cumcount()
    sensor_counts = raw.groupby("id")["id"].transform("count")
    cutoff = np.ceil(HIST_FRAC * sensor_counts).astype(int)
    raw["is_future"] = raw["row_in_sensor"] >= cutoff
    per_sensor_summary = raw.groupby("id").agg(n=("id","size"), n_future=("is_future","sum"))
    print(per_sensor_summary, flush=True)
    print(f"total future rows (pre-feature-build): {raw['is_future'].sum()} / {len(raw)}", flush=True)

    hist_rows = raw[~raw["is_future"]].copy()
    X_tr = hist_rows[["RH","PA","id"]]
    y_tr = hist_rows["ref"]
    ct = ColumnTransformer([("encoder", OneHotEncoder(handle_unknown="ignore"), ["id"])], remainder="passthrough")
    X_tr_enc = ct.fit_transform(X_tr)
    sc = StandardScaler(with_mean=False)
    X_tr_scaled = sc.fit_transform(X_tr_enc)
    rf_hist = RandomForestRegressor(n_estimators=100, max_depth=None, random_state=RANDOM_STATE, n_jobs=2)
    rf_hist.fit(X_tr_scaled, y_tr)
    print(f"Stage 1 (historical-only) fit on {len(hist_rows)} rows, {hist_rows['id'].nunique()} sensors, "
          f"took {time.time()-t0:.0f}s", flush=True)

    X_all = raw[["RH","PA","id"]]
    X_all_enc = ct.transform(X_all)
    X_all_scaled = sc.transform(X_all_enc)
    raw["Predicted_ref_clean"] = rf_hist.predict(X_all_scaled).astype(float)

    pred_df = pd.read_csv(f"{DATA_DIR}/rf_pred_all.csv")
    pred_df["hr"] = pd.to_datetime(pred_df["hr"], errors="coerce")
    pred_df = pred_df.rename(columns={"Predicted_ref":"Predicted_ref_canonical"})
    raw = raw.merge(pred_df[["hr","id","Predicted_ref_canonical"]], on=["hr","id"], how="left")
    print(f"merged canonical Predicted_ref, missing: {raw['Predicted_ref_canonical'].isna().sum()}", flush=True)

    def build_features(df, pred_col, delta_map=CANONICAL_DELTA):
        d = df.sort_values(["id","hr"]).reset_index(drop=True).copy()
        d["Predicted_ref"] = d[pred_col]
        d["lag1_PA"] = d.groupby("id")["PA"].shift(1)
        d["dp_PA"] = d["PA"] - d["lag1_PA"]
        d["lag1_pred"] = d.groupby("id")["Predicted_ref"].shift(1)
        d["dp_pred"] = d["Predicted_ref"] - d["lag1_pred"]
        d["corr"] = d["Predicted_ref"] - d["PA"]
        d["abs_dp_PA"] = d["dp_PA"].abs()
        d["abs_dp_pred"] = d["dp_pred"].abs()
        for w in [3,6]:
            d[f"PA_roll_mean_{w}"] = d.groupby("id")["PA"].transform(lambda s: s.rolling(w,min_periods=1).mean())
            d[f"PA_roll_std_{w}"] = d.groupby("id")["PA"].transform(lambda s: s.rolling(w,min_periods=2).std())
            d[f"pred_roll_mean_{w}"] = d.groupby("id")["Predicted_ref"].transform(lambda s: s.rolling(w,min_periods=1).mean())
            d[f"pred_roll_std_{w}"] = d.groupby("id")["Predicted_ref"].transform(lambda s: s.rolling(w,min_periods=2).std())
        rstd_cols = [c for c in d.columns if "roll_std" in c]
        d[rstd_cols] = d[rstd_cols].fillna(0.0)
        d["AQI_group"] = pd.cut(d["Predicted_ref"], bins=AQI_EDGES, labels=AQI_LABELS).astype(str).str.strip()
        d["abs_error"] = (d["ref"] - d["Predicted_ref"]).abs()
        need = FULL_NUM_FEATS + CAT_FEATS + ["abs_error","is_future"]
        d = d.dropna(subset=need).copy().reset_index(drop=True)
        d["delta_used"] = d["AQI_group"].map(delta_map).astype(float)
        d["reliable"] = (d["abs_error"] <= d["delta_used"]).astype(int)
        return d

    df_clean_full = build_features(raw, "Predicted_ref_clean")
    df_canon_full = build_features(raw, "Predicted_ref_canonical")

    df_clean = df_clean_full[df_clean_full["is_future"]].copy().reset_index(drop=True)
    df_canon = df_canon_full[df_canon_full["is_future"]].copy().reset_index(drop=True)
    print(f"future-only skeleton: clean n={len(df_clean)} sensors={df_clean['id'].nunique()}, "
          f"canonical n={len(df_canon)} sensors={df_canon['id'].nunique()}", flush=True)

    df_clean["_key"] = df_clean["id"].astype(str) + "||" + df_clean["hr"].astype(str)
    df_canon["_key"] = df_canon["id"].astype(str) + "||" + df_canon["hr"].astype(str)
    common_keys = set(df_clean["_key"]) & set(df_canon["_key"])
    print(f"common future rows across both skeletons: {len(common_keys)}", flush=True)
    df_clean = df_clean[df_clean["_key"].isin(common_keys)].drop(columns=["_key"]).reset_index(drop=True)
    df_canon = df_canon[df_canon["_key"].isin(common_keys)].drop(columns=["_key"]).reset_index(drop=True)
    df_clean = df_clean.sort_values(["id","hr"]).reset_index(drop=True)
    df_canon = df_canon.sort_values(["id","hr"]).reset_index(drop=True)
    assert list(zip(df_clean["id"],df_clean["hr"])) == list(zip(df_canon["id"],df_canon["hr"])), "skeletons must align row-for-row"
    print(f"aligned skeleton size: {len(df_clean)}", flush=True)

    def build_trust_model(random_state):
        pre = ColumnTransformer([("aqi_ohe", OneHotEncoder(handle_unknown="ignore"), CAT_FEATS)], remainder="passthrough")
        clf = XGBClassifier(**XGB_PARAMS, objective="binary:logistic", eval_metric="logloss",
                             tree_method="hist", random_state=random_state, n_jobs=2)
        return Pipeline([("pre",pre),("clf",clf)])

    def fit_eval(df_train, df_cal, df_test, cols, random_state):
        y_train = df_train["reliable"].values; y_cal = df_cal["reliable"].values; y_test = df_test["reliable"].values
        X_train = df_train[cols]; X_cal = df_cal[cols]; X_test = df_test[cols]
        pipe = build_trust_model(random_state=random_state)
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
        return safe_prauc(y_test, p_cal_test), safe_auc(y_test, p_cal_test)

    cols = CAT_FEATS + FULL_NUM_FEATS
    n_repeats = 10
    clean_pr, clean_roc, canon_pr, canon_roc = [], [], [], []

    for rep in range(n_repeats):
        gss1r = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE+rep)
        idx_traincal, idx_test = next(gss1r.split(df_clean, groups=df_clean["id"]))
        df_traincal_ids = df_clean.iloc[idx_traincal]
        gss2r = GroupShuffleSplit(n_splits=1, test_size=0.2/(1-0.2), random_state=RANDOM_STATE+100+rep)
        idx_train, idx_cal = next(gss2r.split(df_traincal_ids, groups=df_traincal_ids["id"]))

        train_sensors = set(df_traincal_ids.iloc[idx_train]["id"].unique())
        cal_sensors = set(df_traincal_ids.iloc[idx_cal]["id"].unique())
        test_sensors = set(df_clean.iloc[idx_test]["id"].unique())
        assert not (train_sensors & cal_sensors) and not (train_sensors & test_sensors) and not (cal_sensors & test_sensors)

        dfc_train = df_clean[df_clean["id"].isin(train_sensors)]
        dfc_cal = df_clean[df_clean["id"].isin(cal_sensors)]
        dfc_test = df_clean[df_clean["id"].isin(test_sensors)]
        pr, roc = fit_eval(dfc_train, dfc_cal, dfc_test, cols, random_state=RANDOM_STATE+rep)
        clean_pr.append(pr); clean_roc.append(roc)

        dfn_train = df_canon[df_canon["id"].isin(train_sensors)]
        dfn_cal = df_canon[df_canon["id"].isin(cal_sensors)]
        dfn_test = df_canon[df_canon["id"].isin(test_sensors)]
        pr2, roc2 = fit_eval(dfn_train, dfn_cal, dfn_test, cols, random_state=RANDOM_STATE+rep)
        canon_pr.append(pr2); canon_roc.append(roc2)

        print(f"rep {rep+1}/{n_repeats} done ({time.time()-t0:.0f}s) "
              f"clean_pr={pr:.4f} canon_recomputed_pr={pr2:.4f} "
              f"train={len(train_sensors)} cal={len(cal_sensors)} test={len(test_sensors)} "
              f"n_test_rows={len(dfc_test)}", flush=True)

    clean_pr, canon_pr = np.array(clean_pr), np.array(canon_pr)
    clean_roc, canon_roc = np.array(clean_roc), np.array(canon_roc)
    diff_pr = canon_pr - clean_pr
    diff_roc = canon_roc - clean_roc

    def ci95(x):
        m = x.mean(); se = x.std(ddof=1)/np.sqrt(len(x))
        return m, m-1.96*se, m+1.96*se

    w_pr = stats.wilcoxon(canon_pr, clean_pr) if not np.allclose(canon_pr, clean_pr) else None
    w_roc = stats.wilcoxon(canon_roc, clean_roc) if not np.allclose(canon_roc, clean_roc) else None

    result = {
        "hist_frac": HIST_FRAC,
        "n_future_rows_aligned_skeleton": int(len(df_clean)),
        "temporally_clean_pr_auc": {"mean": float(clean_pr.mean()), "std": float(clean_pr.std()), "per_repeat": clean_pr.tolist()},
        "canonical_recomputed_future_only_pr_auc": {"mean": float(canon_pr.mean()), "std": float(canon_pr.std()), "per_repeat": canon_pr.tolist()},
        "temporally_clean_roc_auc": {"mean": float(clean_roc.mean()), "std": float(clean_roc.std()), "per_repeat": clean_roc.tolist()},
        "canonical_recomputed_future_only_roc_auc": {"mean": float(canon_roc.mean()), "std": float(canon_roc.std()), "per_repeat": canon_roc.tolist()},
        "paired_diff_pr_auc_canon_minus_clean": {"mean": float(diff_pr.mean()), "std": float(diff_pr.std()), "ci95": list(ci95(diff_pr)[1:])},
        "paired_diff_roc_auc_canon_minus_clean": {"mean": float(diff_roc.mean()), "std": float(diff_roc.std()), "ci95": list(ci95(diff_roc)[1:])},
        "wilcoxon_pr_auc": {"statistic": float(w_pr.statistic), "pvalue": float(w_pr.pvalue)} if w_pr else None,
        "wilcoxon_roc_auc": {"statistic": float(w_roc.statistic), "pvalue": float(w_roc.pvalue)} if w_roc else None,
        "sign_consistency_pr": int(np.sum(diff_pr > 0)),
        "sign_consistency_roc": int(np.sum(diff_roc > 0)),
    }
    print(json.dumps(result, indent=2), flush=True)
    json.dump(result, open(f"{RESULTS_DIR}/rf_historical_future_holdout_results.json","w"), indent=2)
    print("TOTAL TIME", time.time()-t0, flush=True)
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    run_temporal_historical_future_leak_audit()
