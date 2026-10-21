"""R2 (temporally clean known-sensor) with the paper's canonical delta (A) vs a delta measured out-of-sample (B).
In R2 every Stage-2 row is a 'future' row, so Predicted_ref is already out-of-sample for all of them; the only
in-sample ingredient is the canonical delta (0.982 / 9.697), which came from R1 in-sample residuals.
B: per repeat, delta = mean |ref - Predicted_ref| by AQI group over the TRAIN sensors' future rows.
Same splits/seeds as internal/r2_temporal_audit.py. Writes only to safety/out."""
import time, pickle
import numpy as np, pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import average_precision_score, roc_auc_score
from scipy.stats import spearmanr
from common import *

t0 = time.time()
need = NUM_FEATS + CAT_FEATS + ["abs_error"]
raw = pd.read_csv(f"{DATA}/WildFireData.csv")
raw["hr"] = pd.to_datetime(raw["hr"], errors="coerce")
raw = raw.dropna(subset=["hr", "RH", "PA", "id", "ref"]).copy()
raw["id"] = raw["id"].astype(str)
raw = raw.sort_values(["id", "hr"]).reset_index(drop=True)
raw["row_in_sensor"] = raw.groupby("id").cumcount()
cnt = raw.groupby("id")["id"].transform("count")
raw["is_future"] = raw["row_in_sensor"] >= np.ceil(0.8 * cnt).astype(int)
hist = raw[~raw["is_future"]]
ct = ColumnTransformer([("encoder", OneHotEncoder(handle_unknown="ignore"), ["id"])], remainder="passthrough")
Xh = ct.fit_transform(hist[["RH", "PA", "id"]])
sc = StandardScaler(with_mean=False); Xh = sc.fit_transform(Xh)
rf = RandomForestRegressor(n_estimators=100, max_depth=None, random_state=RS, n_jobs=2).fit(Xh, hist["ref"].values)
raw["Predicted_ref"] = rf.predict(sc.transform(ct.transform(raw[["RH", "PA", "id"]]))).astype(float)
print("stage1 fit", len(hist), "rows", f"{time.time()-t0:.0f}s", flush=True)
dfc = add_features(raw[["hr", "RH", "PA", "id", "ref", "Predicted_ref", "is_future"]])
dfc = dfc.dropna(subset=need).reset_index(drop=True)
df = dfc[dfc["is_future"]].reset_index(drop=True)
print("future rows", len(df), "sensors", df["id"].nunique(), flush=True)

def pr(y, s): return float(average_precision_score(y, s))
res = []
for rep in range(10):
    g1 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RS + rep)
    itc, ite = next(g1.split(df, groups=df["id"]))
    tc = df.iloc[itc]
    g2 = GroupShuffleSplit(n_splits=1, test_size=0.2 / 0.8, random_state=RS + 100 + rep)
    itr, ica = next(g2.split(tc, groups=tc["id"]))
    tr = set(tc.iloc[itr]["id"]); ca = set(tc.iloc[ica]["id"]); ts = set(df.iloc[ite]["id"])
    out = {"rep": rep}
    dmB = df[df["id"].isin(tr)].groupby("AQI_group")["abs_error"].mean().to_dict()
    for l in AQI_LABELS:
        if l not in dmB or pd.isna(dmB[l]): dmB[l] = CANON_DELTA[l]
    for name, dm in [("A_canonical", CANON_DELTA), ("B_oos_delta", dmB)]:
        d = df.copy(); d["reliable"] = make_labels(d, "delta", dm)
        a = d[d["id"].isin(tr)]; b = d[d["id"].isin(ca)]; c = d[d["id"].isin(ts)]
        _, pt = fit_score(a, b, c, a["reliable"].values, b["reliable"].values, XGB_RF, RS + rep)
        y = c["reliable"].values
        out[name] = dict(prev=float(y.mean()), prauc=pr(y, pt), roc=float(roc_auc_score(y, pt)),
                         pa_only_prauc=pr(y, -c["PA"].values), predref_only_prauc=pr(y, -c["Predicted_ref"].values),
                         spearman=float(spearmanr(pt, c["abs_error"].values)[0]), delta=dm)
    res.append(out)
    pickle.dump(res, open(f"{SAFE}/out/oof_R2.pkl", "wb"))
    print(f"rep {rep+1}/10 t={time.time()-t0:.0f}s A pr={out['A_canonical']['prauc']:.3f} prev={out['A_canonical']['prev']:.3f} | "
          f"B pr={out['B_oos_delta']['prauc']:.3f} prev={out['B_oos_delta']['prev']:.3f}", flush=True)
print("DONE", flush=True)
