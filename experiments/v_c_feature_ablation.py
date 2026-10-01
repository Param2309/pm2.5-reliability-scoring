"""Section V-C. Paired feature ablation (Predicted_ref and all derived features removed), refit delta. Table XIV."""

from config import DATA_DIR, RESULTS_DIR


def run_feature_ablation_refit_delta():
    import pandas as pd, numpy as np, json, time
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
    from xgboost import XGBClassifier
    from scipy import stats

    AQI_EDGES = [-np.inf, 35.4, np.inf]
    AQI_LABELS = ["Non-smoky", "Smoky"]
    FULL_NUM_FEATS = ["RH","PA","Predicted_ref","lag1_PA","dp_PA","lag1_pred","dp_pred","corr",
                 "abs_dp_PA","abs_dp_pred","PA_roll_mean_3","PA_roll_std_3",
                 "pred_roll_mean_3","pred_roll_std_3","PA_roll_mean_6","PA_roll_std_6",
                 "pred_roll_mean_6","pred_roll_std_6"]
    FULL_CAT_FEATS = ["AQI_group"]

    CLEAN_ABLATED_NUM_FEATS = ["PA","RH","lag1_PA","dp_PA","abs_dp_PA",
                 "PA_roll_mean_3","PA_roll_std_3","PA_roll_mean_6","PA_roll_std_6"]
    CLEAN_ABLATED_CAT_FEATS = []

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

    def build_trust_model(cat_feats, random_state):
        clf = XGBClassifier(**XGB_PARAMS, objective="binary:logistic", eval_metric="logloss",
                             tree_method="hist", random_state=random_state, n_jobs=2)
        if cat_feats:
            pre = ColumnTransformer([("aqi_ohe", OneHotEncoder(handle_unknown="ignore"), cat_feats)], remainder="passthrough")
            return Pipeline([("pre",pre),("clf",clf)])
        else:
            return Pipeline([("clf",clf)])

    t0=time.time()
    pred_df = pd.read_csv(f"{DATA_DIR}/rf_pred_all.csv")
    df0 = build_features(pred_df)
    print(f"rows after feature build: {len(df0)}, sensors: {df0['id'].nunique()}", flush=True)

    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    idx_tc, idx_test0 = next(gss1.split(df0, groups=df0["id"]))
    traincal0 = df0.iloc[idx_tc]
    AQI_DELTA_MAP = traincal0.groupby("AQI_group")["abs_error"].mean().to_dict()
    print("delta map:", AQI_DELTA_MAP, flush=True)
    df0["delta_used"] = df0["AQI_group"].map(AQI_DELTA_MAP).astype(float)
    df0["reliable"] = (df0["abs_error"] <= df0["delta_used"]).astype(int)

    n_repeats = 10
    full_pr, full_roc = [], []
    abl_pr, abl_roc = [], []

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

        for cat_feats, num_feats, pr_list, roc_list in [
            (FULL_CAT_FEATS, FULL_NUM_FEATS, full_pr, full_roc),
            (CLEAN_ABLATED_CAT_FEATS, CLEAN_ABLATED_NUM_FEATS, abl_pr, abl_roc),
        ]:
            cols = cat_feats + num_feats
            X_train = df_train[cols]; X_cal = df_cal[cols]; X_test = df_test[cols]
            pipe = build_trust_model(cat_feats, random_state=RANDOM_STATE+rep)
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
            pr_list.append(safe_prauc(y_test, p_cal_test))
            roc_list.append(safe_auc(y_test, p_cal_test))

        print(f"rep {rep+1}/{n_repeats} done ({time.time()-t0:.0f}s) full_pr={full_pr[-1]:.4f} abl_pr={abl_pr[-1]:.4f}", flush=True)

    full_pr, abl_pr = np.array(full_pr), np.array(abl_pr)
    full_roc, abl_roc = np.array(full_roc), np.array(abl_roc)
    diff_pr = full_pr - abl_pr
    diff_roc = full_roc - abl_roc

    def ci95(x):
        m = x.mean(); se = x.std(ddof=1)/np.sqrt(len(x))
        return m, m-1.96*se, m+1.96*se

    w_pr = stats.wilcoxon(full_pr, abl_pr) if not np.allclose(full_pr, abl_pr) else None

    result = {
        "full_pr_auc": {"mean": float(full_pr.mean()), "std": float(full_pr.std())},
        "ablated_pr_auc": {"mean": float(abl_pr.mean()), "std": float(abl_pr.std())},
        "full_roc_auc": {"mean": float(full_roc.mean()), "std": float(full_roc.std())},
        "ablated_roc_auc": {"mean": float(abl_roc.mean()), "std": float(abl_roc.std())},
        "paired_diff_pr_auc": {"mean": float(diff_pr.mean()), "std": float(diff_pr.std()), "per_repeat": diff_pr.tolist()},
        "paired_diff_roc_auc": {"mean": float(diff_roc.mean()), "std": float(diff_roc.std()), "per_repeat": diff_roc.tolist()},
        "wilcoxon_pr_auc": {"statistic": float(w_pr.statistic), "pvalue": float(w_pr.pvalue)} if w_pr else None,
    }
    result["paired_diff_pr_auc"]["ci95"] = list(ci95(diff_pr)[1:])
    print(json.dumps(result, indent=2), flush=True)
    json.dump(result, open(f"{RESULTS_DIR}/rf_ablation_paired_clean_results.json","w"), indent=2)
    print("TOTAL TIME", time.time()-t0, flush=True)
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    run_feature_ablation_refit_delta()
