"""Section IV-G. Stage 2 classifier comparison with Stage 1 fixed (Table V)."""

from config import DATA_DIR, RESULTS_DIR


def run_stage2_classifier_compare():
    import pandas as pd, numpy as np, json, time
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
    from xgboost import XGBClassifier

    AQI_EDGES = [-np.inf, 35.4, np.inf]
    AQI_LABELS = ["Non-smoky", "Smoky"]
    CANONICAL_DELTA = {"Non-smoky": 0.981723249508968, "Smoky": 9.697103519851735}
    NUM_FEATS = ["RH","PA","Predicted_ref","lag1_PA","dp_PA","lag1_pred","dp_pred","corr",
                 "abs_dp_PA","abs_dp_pred","PA_roll_mean_3","PA_roll_std_3",
                 "pred_roll_mean_3","pred_roll_std_3","PA_roll_mean_6","PA_roll_std_6",
                 "pred_roll_mean_6","pred_roll_std_6"]
    CAT_FEATS = ["AQI_group"]
    COLS = CAT_FEATS + NUM_FEATS
    RANDOM_STATE = 42
    ECE_BINS = 15

    FIXED_PARAMS = {
        "LR": {"C": 0.0019, "class_weight": "balanced"},
        "RF": {"n_estimators": 500, "max_depth": None, "min_samples_leaf": 10,
               "max_features": "sqrt", "class_weight": "balanced"},
        "XGB": {'n_estimators': 1350, 'learning_rate': 0.029454474784661306, 'max_depth': 7,
                'min_child_weight': 5.482646385370849, 'subsample': 0.7646717241827257,
                'colsample_bytree': 0.9926123594067943, 'reg_lambda': 19.383944005992362,
                'reg_alpha': 0.00010444341231664277, 'gamma': 2.202410950657256,
                'scale_pos_weight': 3.898438563311057}
    }

    def ece_score(y_true, p, n_bins=15):
        bins = np.linspace(0,1,n_bins+1); ece, n = 0.0, len(p)
        for i in range(n_bins):
            lo,hi = bins[i],bins[i+1]
            mask = (p>=lo)&(p<hi) if i<n_bins-1 else (p>=lo)&(p<=hi)
            if mask.sum()==0: continue
            ece += (mask.sum()/n)*abs(y_true[mask].mean()-p[mask].mean())
        return float(ece)

    def safe_auc(y,p): return float(roc_auc_score(y,p)) if len(np.unique(y))>1 else np.nan
    def safe_prauc(y,p): return float(average_precision_score(y,p)) if len(np.unique(y))>1 else np.nan

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
    rstd = [c for c in pred_df.columns if "roll_std" in c]
    pred_df[rstd] = pred_df[rstd].fillna(0.0)
    pred_df["AQI_group"] = pd.cut(pred_df["Predicted_ref"], bins=AQI_EDGES, labels=AQI_LABELS).astype(str).str.strip()
    pred_df["abs_error"] = (pred_df["ref"] - pred_df["Predicted_ref"]).abs()
    pred_df["delta_used"] = pred_df["AQI_group"].map(CANONICAL_DELTA).astype(float)
    pred_df["reliable"] = (pred_df["abs_error"] <= pred_df["delta_used"]).astype(int)
    need = NUM_FEATS + CAT_FEATS + ["abs_error","reliable"]
    df0 = pred_df.dropna(subset=need).copy().reset_index(drop=True)
    print(f"rows={len(df0)} sensors={df0['id'].nunique()}", flush=True)

    def make_split(rep):
        gss1 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE+rep)
        idx_tc, idx_test = next(gss1.split(df0, groups=df0["id"]))
        df_tc = df0.iloc[idx_tc].copy(); df_test = df0.iloc[idx_test].copy()
        gss2 = GroupShuffleSplit(n_splits=1, test_size=0.2/(1-0.2), random_state=RANDOM_STATE+100+rep)
        idx_train, idx_cal = next(gss2.split(df_tc, groups=df_tc["id"]))
        df_train = df_tc.iloc[idx_train].copy(); df_cal = df_tc.iloc[idx_cal].copy()
        return df_train, df_cal, df_test

    def build_pipe(kind, random_state):
        p = FIXED_PARAMS[kind]
        pre = ColumnTransformer([("aqi_ohe", OneHotEncoder(handle_unknown="ignore"), CAT_FEATS)], remainder="passthrough")
        if kind == "LR":
            clf = LogisticRegression(max_iter=3000, C=p["C"], class_weight=p["class_weight"])
        elif kind == "RF":
            clf = RandomForestClassifier(n_estimators=p["n_estimators"], max_depth=p["max_depth"],
                                          min_samples_leaf=p["min_samples_leaf"], max_features=p["max_features"],
                                          class_weight=p["class_weight"], random_state=random_state, n_jobs=2)
        elif kind == "XGB":
            clf = XGBClassifier(n_estimators=p["n_estimators"], learning_rate=p["learning_rate"],
                                 max_depth=p["max_depth"], min_child_weight=p["min_child_weight"],
                                 subsample=p["subsample"], colsample_bytree=p["colsample_bytree"],
                                 reg_lambda=p["reg_lambda"], reg_alpha=p["reg_alpha"], gamma=p["gamma"],
                                 scale_pos_weight=p["scale_pos_weight"],
                                 objective="binary:logistic", eval_metric="logloss", tree_method="hist",
                                 random_state=random_state, n_jobs=2)
        return Pipeline([("pre",pre),("clf",clf)])

    t0=time.time()
    results = {}
    for kind in ["LR","RF","XGB"]:
        print(f"=== {kind} (fixed params, no tuning) ===", flush=True)
        raw_m = {"pr":[],"roc":[],"brier":[],"ece":[]}
        cal_m = {"pr":[],"roc":[],"brier":[],"ece":[]}
        for rep in range(10):
            df_train, df_cal, df_test = make_split(rep)
            X_train = df_train[COLS]; y_train = df_train["reliable"].values
            X_cal = df_cal[COLS]; y_cal = df_cal["reliable"].values
            X_test = df_test[COLS]; y_test = df_test["reliable"].values
            pipe = build_pipe(kind, RANDOM_STATE+rep)
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
            raw_m["pr"].append(safe_prauc(y_test,p_raw_test)); raw_m["roc"].append(safe_auc(y_test,p_raw_test))
            raw_m["brier"].append(brier_score_loss(y_test,p_raw_test)); raw_m["ece"].append(ece_score(y_test,p_raw_test,ECE_BINS))
            cal_m["pr"].append(safe_prauc(y_test,p_cal_test)); cal_m["roc"].append(safe_auc(y_test,p_cal_test))
            cal_m["brier"].append(brier_score_loss(y_test,p_cal_test)); cal_m["ece"].append(ece_score(y_test,p_cal_test,ECE_BINS))
            print(f"  {kind} rep {rep+1}/10 ({time.time()-t0:.0f}s) pr={cal_m['pr'][-1]:.3f} roc={cal_m['roc'][-1]:.3f}", flush=True)
        def m(d): return {k:(float(np.nanmean(v)),float(np.nanstd(v))) for k,v in d.items()}
        results[kind] = {"params": FIXED_PARAMS[kind], "raw": m(raw_m), "cal": m(cal_m)}
        print(f"{kind} DONE: cal={results[kind]['cal']}", flush=True)

    json.dump(results, open(f"{RESULTS_DIR}/stage2_classifier_compare_results.json","w"), indent=2)
    print("TOTAL TIME", time.time()-t0, flush=True)
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    run_stage2_classifier_compare()
