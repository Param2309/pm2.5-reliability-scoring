"""Sections IV-G and V-B. Safety-oriented evaluation with unreliable as the positive class, false-accept rates and Spearman check against raw-PA error."""

from config import DATA_DIR, RESULTS_DIR


def run_safety_diagnostics_naive_regime():
    import pandas as pd, numpy as np, json, time
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, average_precision_score
    from xgboost import XGBClassifier
    from scipy.stats import spearmanr

    AQI_EDGES = [-np.inf, 35.4, np.inf]
    AQI_LABELS = ["Non-smoky", "Smoky"]
    FULL_NUM_FEATS = ["RH","PA","Predicted_ref","lag1_PA","dp_PA","lag1_pred","dp_pred","corr",
                 "abs_dp_PA","abs_dp_pred","PA_roll_mean_3","PA_roll_std_3",
                 "pred_roll_mean_3","pred_roll_std_3","PA_roll_mean_6","PA_roll_std_6",
                 "pred_roll_mean_6","pred_roll_std_6"]
    CAT_FEATS = ["AQI_group"]
    RANDOM_STATE = 42
    THRESHOLDS = [0.5,0.7,0.8,0.9]
    XGB_PARAMS = {'n_estimators': 1350, 'learning_rate': 0.029454474784661306, 'max_depth': 7,
                  'min_child_weight': 5.482646385370849, 'subsample': 0.7646717241827257,
                  'colsample_bytree': 0.9926123594067943, 'reg_lambda': 19.383944005992362,
                  'reg_alpha': 0.00010444341231664277, 'gamma': 2.202410950657256,
                  'scale_pos_weight': 3.898438563311057}

    def safe_prauc(y,p): return float(average_precision_score(y,p)) if len(np.unique(y))>1 else np.nan
    def safe_auc(y,p): return float(roc_auc_score(y,p)) if len(np.unique(y))>1 else np.nan

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
        df["abs_error_raw_PA"] = (df["ref"] - df["PA"]).abs()
        need = FULL_NUM_FEATS + CAT_FEATS + ["abs_error"]
        df0 = df.dropna(subset=need).copy().reset_index(drop=True)
        return df0

    def build_trust_model(random_state):
        pre = ColumnTransformer([("aqi_ohe", OneHotEncoder(handle_unknown="ignore"), CAT_FEATS)], remainder="passthrough")
        clf = XGBClassifier(**XGB_PARAMS, objective="binary:logistic", eval_metric="logloss",
                             tree_method="hist", random_state=random_state, n_jobs=2)
        return Pipeline([("pre",pre),("clf",clf)])

    t0=time.time()
    pred_df = pd.read_csv(f"{DATA_DIR}/rf_pred_all.csv")
    df0 = build_features(pred_df)
    print(f"rows after feature build: {len(df0)}, sensors: {df0['id'].nunique()}", flush=True)

    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    idx_tc, idx_test0 = next(gss1.split(df0, groups=df0["id"]))
    traincal0 = df0.iloc[idx_tc]
    AQI_DELTA_MAP = traincal0.groupby("AQI_group")["abs_error"].mean().to_dict()
    AQI_DELTA_MAP["Smoky"] = float(traincal0.loc[traincal0["Predicted_ref"] > 55.4, "abs_error"].mean())
    print("delta map:", AQI_DELTA_MAP, flush=True)
    df0["delta_used"] = df0["AQI_group"].map(AQI_DELTA_MAP).astype(float)
    df0["reliable"] = (df0["abs_error"] <= df0["delta_used"]).astype(int)

    n_repeats = 10
    pr_reliable, roc_reliable = [], []
    pr_unreliable, roc_unreliable = [], []
    far_at_t = {t: [] for t in THRESHOLDS}
    cov_at_t = {t: [] for t in THRESHOLDS}
    baseline_pr, baseline_roc = [], []
    spearman_predref_ns, spearman_predref_s = [], []
    spearman_rawpa_ns, spearman_rawpa_s = [], []

    for rep in range(n_repeats):
        gss1r = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE+rep)
        idx_traincal, idx_test = next(gss1r.split(df0, groups=df0["id"]))
        df_traincal = df0.iloc[idx_traincal].copy()
        df_test = df0.iloc[idx_test].copy()
        gss2r = GroupShuffleSplit(n_splits=1, test_size=0.2/(1-0.2), random_state=RANDOM_STATE+100+rep)
        idx_train, idx_cal = next(gss2r.split(df_traincal, groups=df_traincal["id"]))
        df_train = df_traincal.iloc[idx_train].copy()
        df_cal = df_traincal.iloc[idx_cal].copy()

        y_train = df_train["reliable"].values; y_cal = df_cal["reliable"].values; y_test = df_test["reliable"].values
        cols = CAT_FEATS + FULL_NUM_FEATS
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

        pr_reliable.append(safe_prauc(y_test, p_cal_test))
        roc_reliable.append(safe_auc(y_test, p_cal_test))

        y_test_flip = 1 - y_test
        p_flip = 1 - p_cal_test
        pr_unreliable.append(safe_prauc(y_test_flip, p_flip))
        roc_unreliable.append(safe_auc(y_test_flip, p_flip))

        for t in THRESHOLDS:
            keep = p_cal_test >= t
            cov_at_t[t].append(float(keep.mean()))
            if keep.sum() > 0:
                far_at_t[t].append(float((y_test[keep]==0).mean()))
            else:
                far_at_t[t].append(np.nan)

        z = (df_test["abs_dp_PA"].values) + (df_test["PA_roll_std_3"].fillna(0).values)
        baseline_score = 1.0 / (1.0 + z)
        baseline_pr.append(safe_prauc(y_test, baseline_score))
        baseline_roc.append(safe_auc(y_test, baseline_score))

        for grp, ns_list, s_list in [("Non-smoky", spearman_predref_ns, spearman_rawpa_ns),
                                       ("Smoky", spearman_predref_s, spearman_rawpa_s)]:
            m = df_test["AQI_group"].values == grp
            if m.sum() >= 3:
                r1,_ = spearmanr(p_cal_test[m], df_test["abs_error"].values[m])
                r2,_ = spearmanr(p_cal_test[m], df_test["abs_error_raw_PA"].values[m])
                ns_list.append(r1); s_list.append(r2)

        print(f"rep {rep+1}/{n_repeats} done ({time.time()-t0:.0f}s) pr_rel={pr_reliable[-1]:.4f} "
              f"pr_unrel={pr_unreliable[-1]:.4f} baseline_pr={baseline_pr[-1]:.4f}", flush=True)

    def ms(v): return float(np.nanmean(v)), float(np.nanstd(v))

    result = {
        "pr_reliable_pos": ms(pr_reliable), "roc_reliable_pos": ms(roc_reliable),
        "pr_unreliable_pos": ms(pr_unreliable), "roc_unreliable_pos": ms(roc_unreliable),
        "false_accept_rate": {str(t): ms(far_at_t[t]) for t in THRESHOLDS},
        "coverage": {str(t): ms(cov_at_t[t]) for t in THRESHOLDS},
        "baseline_pr_auc": ms(baseline_pr), "baseline_roc_auc": ms(baseline_roc),
        "spearman_predref_error": {"Non-smoky": ms(spearman_predref_ns), "Smoky": ms(spearman_predref_s)},
        "spearman_rawPA_error": {"Non-smoky": ms(spearman_rawpa_ns), "Smoky": ms(spearman_rawpa_s)},
    }
    print(json.dumps(result, indent=2), flush=True)
    json.dump(result, open(f"{RESULTS_DIR}/rf_full_diagnostics_results.json","w"), indent=2)
    print("TOTAL TIME", time.time()-t0, flush=True)
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    run_safety_diagnostics_naive_regime()
