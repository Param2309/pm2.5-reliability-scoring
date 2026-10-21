"""R3 (cold start) with in-sample (paper) vs out-of-fold Stage 1 for the TRAIN sensors.
Same 10 sensor-disjoint repeats and seeds as internal/r3_coldstart.py.
A = paper pipeline (Stage 1 fit on train sensors, Predicted_ref in-sample for train sensors, delta from those residuals).
B = Stage 1 predictions for train sensors made out-of-fold (5-fold, by sensor); delta, labels, features from those.
    Cal/test sensors are unchanged (already out-of-sample for Stage 1 in R3).
Writes only to safety/out."""
import sys, time, json, pickle
import numpy as np, pandas as pd
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score
from scipy.stats import spearmanr
from common import *

t0 = time.time()
need = NUM_FEATS + CAT_FEATS + ["abs_error"]
raw = pd.read_csv(f"{DATA}/WildFireData.csv")
raw["hr"] = pd.to_datetime(raw["hr"], errors="coerce")
raw = raw.dropna(subset=["hr", "RH", "PA", "id", "ref"]).copy()
raw["id"] = raw["id"].astype(str)
sk = pd.read_csv(f"{DATA}/rf_pred_all.csv")
sk["hr"] = pd.to_datetime(sk["hr"], errors="coerce")
sk = sk.dropna(subset=["hr", "RH", "PA", "id", "ref", "Predicted_ref"]).copy()
sk["id"] = sk["id"].astype(str)
df0 = add_features(sk).dropna(subset=need).reset_index(drop=True)
print("rows", len(raw), "skeleton", len(df0), "sensors", raw["id"].nunique(), flush=True)

def rf_fit(rows):
    sc = StandardScaler(); X = sc.fit_transform(rows[["RH", "PA"]].values)
    rf = RandomForestRegressor(n_estimators=100, max_depth=None, random_state=RS, n_jobs=2).fit(X, rows["ref"].values)
    return sc, rf

def pr(y, s): return float(average_precision_score(y, s))
def roc(y, s): return float(roc_auc_score(y, s))

res = []
for rep in range(10):
    g1 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RS + rep)
    itc, ite = next(g1.split(df0, groups=df0["id"]))
    tc = df0.iloc[itc]; te = df0.iloc[ite]
    g2 = GroupShuffleSplit(n_splits=1, test_size=0.2 / 0.8, random_state=RS + 100 + rep)
    itr, ica = next(g2.split(tc, groups=tc["id"]))
    tr = set(tc.iloc[itr]["id"]); ca = set(tc.iloc[ica]["id"]); ts = set(te["id"])
    assert not (tr & ca) and not (tr & ts) and not (ca & ts)

    sc, rf1 = rf_fit(raw[raw["id"].isin(tr)])
    base = raw[["hr", "RH", "PA", "id", "ref"]].copy()
    base["Predicted_ref"] = rf1.predict(sc.transform(raw[["RH", "PA"]].values)).astype(float)
    # OOF predictions for train sensors
    oof = base["Predicted_ref"].copy()
    trrows = raw[raw["id"].isin(tr)]
    gkf = GroupKFold(n_splits=5)
    for a, b in gkf.split(trrows, groups=trrows["id"]):
        sc_k, rf_k = rf_fit(trrows.iloc[a])
        oof.loc[trrows.index[b]] = rf_k.predict(sc_k.transform(trrows.iloc[b][["RH", "PA"]].values))
    out = {"rep": rep, "n_train_sensors": len(tr), "n_cal": len(ca), "n_test": len(ts)}
    for name, pcol in [("A_insample", base["Predicted_ref"]), ("B_oof", oof)]:
        r = base.copy(); r["Predicted_ref"] = pcol.values
        dfc = add_features(r).dropna(subset=need).reset_index(drop=True)
        dm = dfc[dfc["id"].isin(tr)].groupby("AQI_group")["abs_error"].mean().to_dict()
        for l in AQI_LABELS:
            if l not in dm or pd.isna(dm[l]): dm[l] = CANON_DELTA[l]
        dfc["reliable"] = make_labels(dfc, "delta", dm)
        a_ = dfc[dfc["id"].isin(tr)]; b_ = dfc[dfc["id"].isin(ca)]; c_ = dfc[dfc["id"].isin(ts)].copy()
        _, pt = fit_score(a_, b_, c_, a_["reliable"].values, b_["reliable"].values, XGB_RF, RS + rep)
        y = c_["reliable"].values
        sp = spearmanr(pt, c_["abs_error"].values)[0]
        out[name] = dict(prev=float(y.mean()), prauc=pr(y, pt), roc=roc(y, pt),
                         pa_only_prauc=pr(y, -c_["PA"].values), predref_only_prauc=pr(y, -c_["Predicted_ref"].values),
                         spearman=float(sp), delta=dm,
                         train_resid_mae=float(a_["abs_error"].mean()), test_resid_mae=float(c_["abs_error"].mean()))
    res.append(out)
    pickle.dump(res, open(f"{SAFE}/out/oof_R3.pkl", "wb"))
    print(f"rep {rep+1}/10 t={time.time()-t0:.0f}s  A pr={out['A_insample']['prauc']:.3f} prev={out['A_insample']['prev']:.3f} | "
          f"B pr={out['B_oof']['prauc']:.3f} prev={out['B_oof']['prev']:.3f} | "
          f"dA={out['A_insample']['delta']} dB={out['B_oof']['delta']}", flush=True)
print("DONE", flush=True)
