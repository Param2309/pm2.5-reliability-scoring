"""Section V-D. Ablated Stage 2 model versus the non-learned anomaly heuristic."""

from config import DATA_DIR, RESULTS_DIR


def run_heuristic_vs_ablated_comparison():
    import pandas as pd, numpy as np, json, time
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, average_precision_score
    from xgboost import XGBClassifier
    from scipy import stats

    AQI_EDGES = [-np.inf, 35.4, np.inf]
    AQI_LABELS = ["Non-smoky", "Smoky"]
    FULL_NUM_FEATS = ["RH","PA","Predicted_ref","lag1_PA","dp_PA","lag1_pred","dp_pred","corr",
                 "abs_dp_PA","abs_dp_pred","PA_roll_mean_3","PA_roll_std_3",
                 "pred_roll_mean_3","pred_roll_std_3","PA_roll_mean_6","PA_roll_std_6",
                 "pred_roll_mean_6","pred_roll_std_6"]
    CLEAN_ABLATED_NUM_FEATS = ["PA","RH","lag1_PA","dp_PA","abs_dp_PA",
                 "PA_roll_mean_3","PA_roll_std_3","PA_roll_mean_6","PA_roll_std_6"]
    FULL_CAT_FEATS = ["AQI_group"]
    RANDOM_STATE = 42
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
        need = FULL_NUM_FEATS + FULL_CAT_FEATS + ["abs_error"]
        df0 = df.dropna(subset=need).copy().reset_index(drop=True)
        return df0

    t0=time.time()
    pred_df = pd.read_csv(f"{DATA_DIR}/rf_pred_all.csv")
    df0 = build_features(pred_df)

    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    idx_tc, idx_test0 = next(gss1.split(df0, groups=df0["id"]))
    traincal0 = df0.iloc[idx_tc]
    AQI_DELTA_MAP = traincal0.groupby("AQI_group")["abs_error"].mean().to_dict()
    df0["delta_used"] = df0["AQI_group"].map(AQI_DELTA_MAP).astype(float)
    df0["reliable"] = (df0["abs_error"] <= df0["delta_used"]).astype(int)

    n_repeats = 10
    abl_pr, abl_roc, heur_pr, heur_roc = [], [], [], []

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
        cols = CLEAN_ABLATED_NUM_FEATS
        X_train = df_train[cols]; X_cal = df_cal[cols]; X_test = df_test[cols]

        clf = XGBClassifier(**XGB_PARAMS, objective="binary:logistic", eval_metric="logloss",
                             tree_method="hist", random_state=RANDOM_STATE+rep, n_jobs=2)
        pipe = Pipeline([("clf", clf)])
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
        abl_pr.append(safe_prauc(y_test, p_cal_test))
        abl_roc.append(safe_auc(y_test, p_cal_test))

        z = df_test["abs_dp_PA"].values + df_test["PA_roll_std_3"].fillna(0).values
        baseline_score = 1.0 / (1.0 + z)
        heur_pr.append(safe_prauc(y_test, baseline_score))
        heur_roc.append(safe_auc(y_test, baseline_score))

        print(f"rep {rep+1}/{n_repeats} done ({time.time()-t0:.0f}s) abl_pr={abl_pr[-1]:.4f} heur_pr={heur_pr[-1]:.4f}", flush=True)

    abl_pr, heur_pr = np.array(abl_pr), np.array(heur_pr)
    abl_roc, heur_roc = np.array(abl_roc), np.array(heur_roc)
    diff_pr = abl_pr - heur_pr
    diff_roc = abl_roc - heur_roc

    def ci95(x):
        m = x.mean(); se = x.std(ddof=1)/np.sqrt(len(x))
        return m, m-1.96*se, m+1.96*se

    w_pr = stats.wilcoxon(abl_pr, heur_pr) if not np.allclose(abl_pr, heur_pr) else None
    w_roc = stats.wilcoxon(abl_roc, heur_roc) if not np.allclose(abl_roc, heur_roc) else None

    result = {
        "ablated_pr_auc": {"mean": float(abl_pr.mean()), "std": float(abl_pr.std()), "per_repeat": abl_pr.tolist()},
        "heuristic_pr_auc": {"mean": float(heur_pr.mean()), "std": float(heur_pr.std()), "per_repeat": heur_pr.tolist()},
        "ablated_roc_auc": {"mean": float(abl_roc.mean()), "std": float(abl_roc.std()), "per_repeat": abl_roc.tolist()},
        "heuristic_roc_auc": {"mean": float(heur_roc.mean()), "std": float(heur_roc.std()), "per_repeat": heur_roc.tolist()},
        "paired_diff_pr_auc_abl_minus_heur": {"mean": float(diff_pr.mean()), "std": float(diff_pr.std()), "ci95": list(ci95(diff_pr)[1:])},
        "paired_diff_roc_auc_abl_minus_heur": {"mean": float(diff_roc.mean()), "std": float(diff_roc.std()), "ci95": list(ci95(diff_roc)[1:])},
        "wilcoxon_pr_auc": {"statistic": float(w_pr.statistic), "pvalue": float(w_pr.pvalue)} if w_pr else None,
        "wilcoxon_roc_auc": {"statistic": float(w_roc.statistic), "pvalue": float(w_roc.pvalue)} if w_roc else None,
    }
    print(json.dumps(result, indent=2), flush=True)
    json.dump(result, open(f"{RESULTS_DIR}/heuristic_vs_ablated_paired_results.json","w"), indent=2)
    print("TOTAL TIME", time.time()-t0, flush=True)
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    run_heuristic_vs_ablated_comparison()
