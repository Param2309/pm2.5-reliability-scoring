"""R1 (naive fleet protocol) with the paper's in-sample Stage 1 predictions (A, rf_pred_all.csv) vs 5-fold
row-level out-of-fold Stage 1 predictions (B). Everything downstream mirrors stage2/pipeline.py (delta from the rep-0
train+cal split, Smoky delta over Predicted_ref > 55.4, 10 sensor-disjoint repeats, XGB_RF params).
Writes only to safety/out."""
import time, pickle, os
import numpy as np, pandas as pd
from sklearn.model_selection import GroupShuffleSplit, KFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import average_precision_score, roc_auc_score
from scipy.stats import spearmanr
from common import *

t0 = time.time()
need = NUM_FEATS + CAT_FEATS + ["abs_error"]
data = pd.read_csv(f"{DATA}/WildFireData.csv")
oof_path = f"{SAFE}/out/rf_oof_pred_R1.csv"
if os.path.exists(oof_path):
    oofdf = pd.read_csv(oof_path)
else:
    X = data[["RH", "PA", "id"]]
    ct = ColumnTransformer([("encoder", OneHotEncoder(), ["id"])], remainder="passthrough")
    Xs = StandardScaler(with_mean=False).fit_transform(ct.fit_transform(X))
    y = data["ref"].values
    oof = np.zeros(len(data))
    for k, (a, b) in enumerate(KFold(5, shuffle=True, random_state=RS).split(Xs)):
        rf = RandomForestRegressor(n_estimators=100, max_depth=None, random_state=RS, n_jobs=2).fit(Xs[a], y[a])
        oof[b] = rf.predict(Xs[b]); print("oof fold", k + 1, f"{time.time()-t0:.0f}s", flush=True)
    oofdf = pd.DataFrame({"hr": data["hr"], "RH": data["RH"], "PA": data["PA"], "id": data["id"], "ref": data["ref"], "Predicted_ref": oof})
    oofdf.to_csv(oof_path, index=False)
print("OOF stage1 MAE", float(np.abs(oofdf["ref"] - oofdf["Predicted_ref"]).mean()), flush=True)
A = pd.read_csv(f"{DATA}/rf_pred_all.csv")
print("stored (in-sample) stage1 MAE", float(np.abs(A["ref"] - A["Predicted_ref"]).mean()), flush=True)

def pr(y, s): return float(average_precision_score(y, s))
def run(pred_df, tag):
    p = pred_df.copy(); p["hr"] = pd.to_datetime(p["hr"], errors="coerce"); p["id"] = p["id"].astype(str)
    p = p.dropna(subset=["hr", "RH", "PA", "id", "ref", "Predicted_ref"])
    df0 = add_features(p).dropna(subset=need).reset_index(drop=True)
    g = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RS)
    itc, _ = next(g.split(df0, groups=df0["id"]))
    t = df0.iloc[itc]
    dm = t.groupby("AQI_group")["abs_error"].mean().to_dict()
    dm["Smoky"] = float(t.loc[t["Predicted_ref"] > 55.4, "abs_error"].mean())
    df0["reliable"] = make_labels(df0, "delta", dm)
    print(tag, "delta", dm, "prev", df0["reliable"].mean(), "rows", len(df0), flush=True)
    rows = []
    for rep in range(10):
        g1 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RS + rep)
        itc, ite = next(g1.split(df0, groups=df0["id"]))
        tc = df0.iloc[itc]; te = df0.iloc[ite]
        g2 = GroupShuffleSplit(n_splits=1, test_size=0.2 / 0.8, random_state=RS + 100 + rep)
        itr, ica = next(g2.split(tc, groups=tc["id"]))
        a = tc.iloc[itr]; b = tc.iloc[ica]
        _, pt = fit_score(a, b, te, a["reliable"].values, b["reliable"].values, XGB_RF, RS + rep)
        y = te["reliable"].values
        rows.append(dict(prev=float(y.mean()), prauc=pr(y, pt), roc=float(roc_auc_score(y, pt)),
                         pa_only_prauc=pr(y, -te["PA"].values), predref_only_prauc=pr(y, -te["Predicted_ref"].values),
                         spearman=float(spearmanr(pt, te["abs_error"].values)[0]), delta=dm))
        print(f"{tag} rep {rep+1}/10 t={time.time()-t0:.0f}s pr={rows[-1]['prauc']:.3f} prev={rows[-1]['prev']:.3f} pa_only={rows[-1]['pa_only_prauc']:.3f}", flush=True)
    return rows
res = {"A_insample": run(A, "A"), "B_oof": run(oofdf, "B")}
pickle.dump(res, open(f"{SAFE}/out/oof_R1.pkl", "wb"))
print("DONE", flush=True)
