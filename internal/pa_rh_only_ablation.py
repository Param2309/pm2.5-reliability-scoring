#!/usr/bin/env python3
"""Level-only row for the feature ablation (R1, R2, R3) + external check.  ANALYSIS ONLY: no frozen artifact is written or changed, no headline table is touched.

Level-only model = same XGBoost (+ Platt on the calibration sensors) as the ablation, same hyper-parameters, same label, same 10 sensor-disjoint repeats
(GroupShuffleSplit 15 train / 6 calibration / 6 test sensors, seeds 42+rep and 142+rep), same per-regime delta and rows as the existing scripts, but the ONLY inputs are
the current-hour PA and RH (no lags, rolling stats, differences, Predicted_ref or anything derived from it, no AQI group, no sensor id).
The existing temporal-only ablation row already contains current PA and RH plus the temporal features, so it is left as it is; level-only is an additional row.

Regimes (code paths copied from the existing scripts, v_c_feature_ablation.py, v_k_temporal_audit.py, v_l_coldstart_reconstruction.py):
  R1  in-distribution: rf_pred_all.csv Predicted_ref, canonical delta (0.9817 / 9.6971)
  R2  known-sensor, temporal: Stage 1 refit on the first 80% of each sensor's rows, future-only rows, canonical delta, aligned with the canonical skeleton
  R3  cold-start: per repeat Stage 1 (RH, PA only, no id) refit on that repeat's 15 train sensors, delta re-derived from those 15 sensors
Order: (1) full model in all three regimes -> reproduction check against the paper (R1 0.902, R2 0.523, R3 0.399 PR-AUC; tolerance 0.001). If any check fails the script STOPS
before fitting any level-only model.  (2) level-only model in R1-R3.  (3) frozen-model check + external check.
Metrics per repeat on the calibrated (Platt) score: PR-AUC, pi (test base rate), ROC-AUC, Brier, ECE (15 bins, as in the pipeline). Excess PR-AUC = (mean PR - mean pi)/(1 - mean pi).
Paired difference full - level-only: mean, 95% interval = mean +/- 1.96*SE across the 10 repeats (same as the existing ablation scripts), and the number of repeats with full > level-only.
External check: ONE level-only model trained on the standard training sensors (rep-0 train sensors in R1, Platt on the rep-0 calibration sensors), frozen, then used to score the rows
already scored (out_w2020/scored_w2020.csv, out_la2025/scored_la2025.csv). Spearman(score, |ref-Predicted_ref|) and ROC-AUC for the reliable label, 24-hour block bootstrap
(5000 draws, seed 42), full-model numbers from the existing score next to it, plus a paired CI of the difference.
Usage:
  python level_only.py --raw WildFireData.csv --pred-all rf_pred_all.csv --models DIR_with_trust_pipe_and_platt --wc out_w2020/scored_w2020.csv --rub out_la2025/scored_la2025.csv --out out_level_only
  (--smoke N shrinks the XGBoost trees to N for a pipeline test ONLY; it prefixes every output with SMOKE_ and skips the paper check.)"""
import argparse, json, os, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, joblib
from scipy import stats
from scipy.stats import rankdata
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

AQI_EDGES, AQI_LABELS = [-np.inf, 35.4, np.inf], ["Non-smoky", "Smoky"]
FULL_NUM = ["RH", "PA", "Predicted_ref", "lag1_PA", "dp_PA", "lag1_pred", "dp_pred", "corr", "abs_dp_PA", "abs_dp_pred", "PA_roll_mean_3", "PA_roll_std_3",
            "pred_roll_mean_3", "pred_roll_std_3", "PA_roll_mean_6", "PA_roll_std_6", "pred_roll_mean_6", "pred_roll_std_6"]
FULL_CAT = ["AQI_group"]; LEVEL_NUM = ["PA", "RH"]
RS = 42; HIST_FRAC = 0.8; N_REP = 10; ECE_BINS = 15
CANON = {"Non-smoky": 0.981723249508968, "Smoky": 9.697103519851735}
XGB = {'n_estimators': 1350, 'learning_rate': 0.029454474784661306, 'max_depth': 7, 'min_child_weight': 5.482646385370849, 'subsample': 0.7646717241827257,
       'colsample_bytree': 0.9926123594067943, 'reg_lambda': 19.383944005992362, 'reg_alpha': 0.00010444341231664277, 'gamma': 2.202410950657256, 'scale_pos_weight': 3.898438563311057}
PAPER = {"R1": 0.902, "R2": 0.523, "R3": 0.399}; TOL = 0.001
T0 = pd.Timestamp("2020-01-01", tz="UTC")

def safe_pr(y, p): return float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else np.nan
def safe_auc(y, p): return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else np.nan
def ece_score(y, p, n_bins=ECE_BINS):
    b = np.linspace(0, 1, n_bins + 1); e = 0.0
    for i in range(n_bins):
        m = (p >= b[i]) & (p < b[i + 1]) if i < n_bins - 1 else (p >= b[i]) & (p <= b[i + 1])
        if m.sum(): e += m.sum() / len(p) * abs(y[m].mean() - p[m].mean())
    return float(e)
def ci95(x): m = x.mean(); se = x.std(ddof=1) / np.sqrt(len(x)); return float(m - 1.96 * se), float(m + 1.96 * se)

def feats(df, pred_col=None):
    """feature build used by every existing script (rows with no lag are dropped)."""
    d = df.copy()
    if pred_col: d["Predicted_ref"] = d[pred_col]
    d = d.sort_values(["id", "hr"]).reset_index(drop=True)
    d["lag1_PA"] = d.groupby("id")["PA"].shift(1); d["dp_PA"] = d["PA"] - d["lag1_PA"]
    d["lag1_pred"] = d.groupby("id")["Predicted_ref"].shift(1); d["dp_pred"] = d["Predicted_ref"] - d["lag1_pred"]
    d["corr"] = d["Predicted_ref"] - d["PA"]; d["abs_dp_PA"] = d["dp_PA"].abs(); d["abs_dp_pred"] = d["dp_pred"].abs()
    for w in (3, 6):
        d[f"PA_roll_mean_{w}"] = d.groupby("id")["PA"].transform(lambda s: s.rolling(w, min_periods=1).mean())
        d[f"PA_roll_std_{w}"] = d.groupby("id")["PA"].transform(lambda s: s.rolling(w, min_periods=2).std())
        d[f"pred_roll_mean_{w}"] = d.groupby("id")["Predicted_ref"].transform(lambda s: s.rolling(w, min_periods=1).mean())
        d[f"pred_roll_std_{w}"] = d.groupby("id")["Predicted_ref"].transform(lambda s: s.rolling(w, min_periods=2).std())
    rs = [c for c in d.columns if "roll_std" in c]; d[rs] = d[rs].fillna(0.0)
    d["AQI_group"] = pd.cut(d["Predicted_ref"], bins=AQI_EDGES, labels=AQI_LABELS).astype(str).str.strip()
    d["abs_error"] = (d["ref"] - d["Predicted_ref"]).abs()
    return d

def label(d, dmap):
    d = d.copy(); d["delta_used"] = d["AQI_group"].map(dmap).astype(float); d["reliable"] = (d["abs_error"] <= d["delta_used"]).astype(int); return d

def split(df, rep):
    a = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RS + rep); itc, ite = next(a.split(df, groups=df["id"]))
    tc = df.iloc[itc]; b = GroupShuffleSplit(n_splits=1, test_size=0.2 / 0.8, random_state=RS + 100 + rep); itr, ica = next(b.split(tc, groups=tc["id"]))
    return tc.iloc[itr]["id"].unique(), tc.iloc[ica]["id"].unique(), df.iloc[ite]["id"].unique()

def make_model(cat, rep, n_est):
    p = dict(XGB); p["n_estimators"] = n_est
    clf = XGBClassifier(**p, objective="binary:logistic", eval_metric="logloss", tree_method="hist", random_state=RS + rep, n_jobs=2)
    if cat: return Pipeline([("pre", ColumnTransformer([("aqi_ohe", OneHotEncoder(handle_unknown="ignore"), cat)], remainder="passthrough")), ("clf", clf)])
    return Pipeline([("clf", clf)])

def platt_fit(pipe, Xc, yc):
    eps = 1e-6; lg = lambda p: np.log(np.clip(p, eps, 1 - eps) / np.clip(1 - p, eps, 1 - eps)).reshape(-1, 1)
    pc = pipe.predict_proba(Xc)[:, 1]
    if len(np.unique(yc)) < 2: return None, lg
    pl = LogisticRegression(max_iter=2000); pl.fit(lg(pc), yc); return pl, lg

def fit_eval(tr, ca, te, cat, num, rep, n_est):
    cols = cat + num; pipe = make_model(cat, rep, n_est); pipe.fit(tr[cols], tr["reliable"].values)
    pl, lg = platt_fit(pipe, ca[cols], ca["reliable"].values); pr_ = pipe.predict_proba(te[cols])[:, 1]
    p = pr_.copy() if pl is None else pl.predict_proba(lg(pr_))[:, 1]; y = te["reliable"].values
    return dict(pr=safe_pr(y, p), roc=safe_auc(y, p), brier=float(brier_score_loss(y, p)), ece=ece_score(y, p), pi=float(y.mean())), pipe, pl, lg

# ---------------------------------------------------------------- regime generators: yield (rep, train, cal, test) frames
def regime_R1(pred_all):
    d = feats(pred_all.assign(hr=pd.to_datetime(pred_all["hr"], errors="coerce")).dropna(subset=["hr", "RH", "PA", "id", "ref", "Predicted_ref"]))
    d = d.dropna(subset=FULL_NUM + FULL_CAT + ["abs_error"]).reset_index(drop=True)
    dmap = CANON; d = label(d, dmap)   # canonical delta, as in v_e_delta_robustness.py (the headline 0.902); v_c's refit-delta run gives 0.899
    print(f"[R1] rows {len(d)}, sensors {d['id'].nunique()}, delta {dmap}", flush=True)
    for rep in range(N_REP):
        tr, ca, te = split(d, rep); yield rep, d[d["id"].isin(tr)], d[d["id"].isin(ca)], d[d["id"].isin(te)]

def regime_R2(raw, pred_all):
    raw = raw.copy(); raw["hr"] = pd.to_datetime(raw["hr"], errors="coerce"); raw = raw.dropna(subset=["hr", "RH", "PA", "id", "ref"]).copy()
    raw["id"] = raw["id"].astype(str); raw = raw.sort_values(["id", "hr"]).reset_index(drop=True)
    raw["row_in_sensor"] = raw.groupby("id").cumcount(); cnt = raw.groupby("id")["id"].transform("count")
    raw["is_future"] = raw["row_in_sensor"] >= np.ceil(HIST_FRAC * cnt).astype(int)
    hist = raw[~raw["is_future"]]
    ct = ColumnTransformer([("encoder", OneHotEncoder(handle_unknown="ignore"), ["id"])], remainder="passthrough"); Xe = ct.fit_transform(hist[["RH", "PA", "id"]])
    sc = StandardScaler(with_mean=False); Xs = sc.fit_transform(Xe)
    rf = RandomForestRegressor(n_estimators=100, max_depth=None, random_state=RS, n_jobs=2); rf.fit(Xs, hist["ref"])
    raw["Predicted_ref_clean"] = rf.predict(sc.transform(ct.transform(raw[["RH", "PA", "id"]]))).astype(float)
    p = pred_all.copy(); p["hr"] = pd.to_datetime(p["hr"], errors="coerce"); p["id"] = p["id"].astype(str)
    p = p.rename(columns={"Predicted_ref": "Predicted_ref_canonical"})
    raw = raw.merge(p[["hr", "id", "Predicted_ref_canonical"]], on=["hr", "id"], how="left")
    def build(pc):
        d = feats(raw, pc); d = d.dropna(subset=FULL_NUM + FULL_CAT + ["abs_error", "is_future"]).reset_index(drop=True); return label(d, CANON)
    cl, cn = build("Predicted_ref_clean"), build("Predicted_ref_canonical")
    cl = cl[cl["is_future"]].reset_index(drop=True); cn = cn[cn["is_future"]].reset_index(drop=True)
    key = lambda d: d["id"].astype(str) + "||" + d["hr"].astype(str); common = set(key(cl)) & set(key(cn))
    cl = cl[key(cl).isin(common)].sort_values(["id", "hr"]).reset_index(drop=True)
    print(f"[R2] aligned future-only skeleton rows {len(cl)}, sensors {cl['id'].nunique()}", flush=True)
    for rep in range(N_REP):
        tr, ca, te = split(cl, rep); yield rep, cl[cl["id"].isin(tr)], cl[cl["id"].isin(ca)], cl[cl["id"].isin(te)]

def regime_R3(raw, pred_all):
    raw = raw.copy(); raw["hr"] = pd.to_datetime(raw["hr"], errors="coerce"); raw = raw.dropna(subset=["hr", "RH", "PA", "id", "ref"]).copy(); raw["id"] = raw["id"].astype(str)
    sk = feats(pred_all.assign(hr=pd.to_datetime(pred_all["hr"], errors="coerce")).dropna(subset=["hr", "RH", "PA", "id", "ref", "Predicted_ref"]))
    sk = sk.dropna(subset=FULL_NUM + FULL_CAT + ["abs_error"]).reset_index(drop=True)
    print(f"[R3] skeleton rows {len(sk)}, sensors {sk['id'].nunique()}", flush=True)
    for rep in range(N_REP):
        tr, ca, te = split(sk, rep); trs, cas, tes = set(tr), set(ca), set(te)
        rt = raw[raw["id"].isin(trs)]; sc = StandardScaler(); Xs = sc.fit_transform(rt[["RH", "PA"]].values)
        rf = RandomForestRegressor(n_estimators=100, max_depth=None, random_state=RS, n_jobs=2); rf.fit(Xs, rt["ref"].values)
        d = raw[["hr", "RH", "PA", "id", "ref"]].copy(); d["Predicted_ref"] = rf.predict(sc.transform(raw[["RH", "PA"]].values)).astype(float)
        d = feats(d); d = d.dropna(subset=FULL_NUM + FULL_CAT + ["abs_error"]).reset_index(drop=True)
        dm = d[d["id"].isin(trs)].groupby("AQI_group")["abs_error"].mean().to_dict()
        for l in AQI_LABELS:
            if l not in dm or pd.isna(dm[l]): dm[l] = CANON[l]
        d = label(d, dm); yield rep, d[d["id"].isin(trs)], d[d["id"].isin(cas)], d[d["id"].isin(tes)]

# ---------------------------------------------------------------- external bootstrap (same draws as baselines.py / binwise_w2020.py)
def sp(a, b):
    if len(a) < 3: return np.nan
    ra, rb = rankdata(a), rankdata(b); return np.nan if ra.std() == 0 or rb.std() == 0 else float(np.corrcoef(ra, rb)[0, 1])
def auc(s, y):
    n1 = int(y.sum()); n0 = len(y) - n1
    return np.nan if n1 == 0 or n0 == 0 else float((rankdata(s)[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
def ci_(v):
    v = np.asarray(v, float); v = v[~np.isnan(v)]
    return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if len(v) >= 100 else (np.nan, np.nan)

def ext_boot(d, regime, S, nboot, seed, block=24):
    d = d.reset_index(drop=True); sel = (d["AQI_group"] == regime).values; err = d.err.values.astype(float); y = d.reliable.values.astype(int)
    pt = {(k, "rho"): sp(v[sel], err[sel]) for k, v in S.items()}; pt.update({(k, "roc"): auc(v[sel], y[sel]) for k, v in S.items()})
    mon = d["monitor_id"].astype(str).values; hh = np.floor((d["hr"] - T0).dt.total_seconds().values / 3600).astype(int); parts = []
    for mo in np.unique(mon):
        ix = np.where(mon == mo)[0]; pos = hh[ix] - hh[ix].min(); L = int(pos.max()) + 1; parts.append((ix, pos, L, max(0, L - block), int(np.ceil(L / block))))
    rng = np.random.default_rng(seed); dr = {(k, m): [] for k in S for m in ("rho", "roc")}
    for _ in range(nboot):
        sl = []
        for ix, pos, L, hi, nb in parts:
            st = rng.integers(0, hi + 1, nb); dz = np.zeros(L + block + 1, int); np.add.at(dz, st, 1); np.add.at(dz, st + block, -1)
            sl.append(np.repeat(ix, np.cumsum(dz)[:L][pos]))
        idx = np.concatenate(sl); idx = idx[sel[idx]]
        if len(idx) < 5: continue
        for k, v in S.items(): dr[(k, "rho")].append(sp(v[idx], err[idx])); dr[(k, "roc")].append(auc(v[idx], y[idx]))
    dr = {k: np.array(v, float) for k, v in dr.items()}; return pt, dr, int(sel.sum()), int(y[sel].sum())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True); ap.add_argument("--pred-all", required=True); ap.add_argument("--models", required=True)
    ap.add_argument("--wc", required=True); ap.add_argument("--rub", required=True); ap.add_argument("--out", default="out_level_only")
    ap.add_argument("--nboot", type=int, default=5000); ap.add_argument("--seed", type=int, default=42); ap.add_argument("--smoke", type=int, default=0)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); n_est = a.smoke if a.smoke else XGB["n_estimators"]; pre = "SMOKE_" if a.smoke else ""
    logf = open(os.path.join(a.out, pre + "level_only_summary.txt"), "w")
    def log(s=""): print(s, flush=True); logf.write(s + "\n"); logf.flush()
    t0 = time.time(); raw = pd.read_csv(a.raw); raw["id"] = raw["id"].astype(str); pred_all = pd.read_csv(a.pred_all); pred_all["id"] = pred_all["id"].astype(str)
    if a.smoke: log(f"*** SMOKE RUN: n_estimators={n_est}; numbers are NOT results ***")
    gens = {"R1": lambda: regime_R1(pred_all), "R2": lambda: regime_R2(raw, pred_all), "R3": lambda: regime_R3(raw, pred_all)}
    full, lev, r0 = {k: [] for k in gens}, {k: [] for k in gens}, {}
    cache = {k: [] for k in gens}
    log("PHASE 1: full model in R1, R2, R3 (reproduction check before any level-only model)")
    for k, g in gens.items():
        for rep, tr, ca, te in g():
            m, pipe, pl, lg = fit_eval(tr, ca, te, FULL_CAT, FULL_NUM, rep, n_est); full[k].append(m); cache[k].append((rep, tr, ca, te))
            if k == "R1" and rep == 0: r0 = dict(pipe=pipe, tr=tr, ca=ca, te=te)
            log(f"  {k} rep {rep + 1}/{N_REP} full PR {m['pr']:.4f} ROC {m['roc']:.4f} pi {m['pi']:.3f}  ({time.time() - t0:.0f}s)")
    ok = True; log("\nREPRODUCTION CHECK (full model, mean PR-AUC vs paper):")
    for k in gens:
        v = float(np.mean([m["pr"] for m in full[k]])); good = abs(v - PAPER[k]) <= TOL; ok &= good
        log(f"  {k}: reproduced {v:.4f} (sd {np.std([m['pr'] for m in full[k]]):.4f})  paper {PAPER[k]:.3f}  -> {'MATCH' if good else 'MISMATCH'}")
    if not ok and not a.smoke: log("\nSTOP: the full-model numbers do not match the paper. No level-only model was fitted."); sys.exit(1)
    log("\nPHASE 2: level-only (PA, RH only), same splits, labels, delta, hyper-parameters")
    for k in gens:
        for rep, tr, ca, te in cache[k]:
            m, *_ = fit_eval(tr, ca, te, [], LEVEL_NUM, rep, n_est); lev[k].append(m)
            log(f"  {k} rep {rep + 1}/{N_REP} level PR {m['pr']:.4f} ROC {m['roc']:.4f}  full PR {full[k][rep]['pr']:.4f}  ({time.time() - t0:.0f}s)")
    rows = []
    def agg(k, name, L):
        g = lambda f: np.array([m[f] for m in L]); pr, pi = g("pr"), g("pi")
        r = dict(kind="internal", setting=k, model=name, n_repeats=len(L), pr_auc_mean=pr.mean(), pr_auc_sd=pr.std(), pi_mean=pi.mean(), pi_sd=pi.std(),
                 excess_pr_auc=(pr.mean() - pi.mean()) / (1 - pi.mean()), roc_auc_mean=g("roc").mean(), roc_auc_sd=g("roc").std(), brier_mean=g("brier").mean(), brier_sd=g("brier").std(),
                 ece_mean=g("ece").mean(), ece_sd=g("ece").std()); return r
    summ = []
    for k in gens:
        rows += [agg(k, "full", full[k]), agg(k, "level_only", lev[k])]
        dp = np.array([f["pr"] - l["pr"] for f, l in zip(full[k], lev[k])]); dr = np.array([f["roc"] - l["roc"] for f, l in zip(full[k], lev[k])])
        lo, hi = ci95(dp); rlo, rhi = ci95(dr)
        rows.append(dict(kind="internal", setting=k, model="paired_diff_full_minus_level", n_repeats=N_REP, d_pr_mean=dp.mean(), d_pr_sd=dp.std(), d_pr_lo=lo, d_pr_hi=hi, d_pr_n_full_gt=int((dp > 0).sum()),
                         d_roc_mean=dr.mean(), d_roc_sd=dr.std(), d_roc_lo=rlo, d_roc_hi=rhi, d_roc_n_full_gt=int((dr > 0).sum())))
        f_, l_ = rows[-3], rows[-2]
        summ.append(f"{k}: full PR {f_['pr_auc_mean']:.3f}±{f_['pr_auc_sd']:.3f} vs level-only {l_['pr_auc_mean']:.3f}±{l_['pr_auc_sd']:.3f}; paired diff {dp.mean():+.3f} (95% {lo:+.3f} to {hi:+.3f}), full > level in {int((dp > 0).sum())}/10; "
                    f"ROC {f_['roc_auc_mean']:.3f} vs {l_['roc_auc_mean']:.3f}, diff {dr.mean():+.3f} ({rlo:+.3f} to {rhi:+.3f}), full > level in {int((dr > 0).sum())}/10; excess PR {f_['excess_pr_auc']:.3f} vs {l_['excess_pr_auc']:.3f}")
    # ---- frozen-model check and external level-only model
    log("\nPHASE 3: external check")
    cols = FULL_CAT + FULL_NUM; frozen = joblib.load(os.path.join(a.models, "trust_pipe.pkl"))
    dd = np.abs(frozen.predict_proba(r0["te"][cols])[:, 1] - r0["pipe"].predict_proba(r0["te"][cols])[:, 1])
    log(f"[frozen check] refit full model on rep-0 train sensors vs frozen trust_pipe.pkl, raw probabilities on the rep-0 test rows: max|diff|={dd.max():.2e}, median={np.median(dd):.2e}")
    log("   -> rep-0 train/calibration sensors " + ("ARE" if dd.max() < 1e-3 else "are NOT confirmed to be") + " the sensors the frozen Stage 2 was trained on" + ("" if dd.max() < 1e-3 else " (level-only external model still uses rep-0 sensors, labelled 'rep-0 sensors')"))
    lp = make_model([], 0, n_est); lp.fit(r0["tr"][LEVEL_NUM], r0["tr"]["reliable"].values)
    pl, lg = platt_fit(lp, r0["ca"][LEVEL_NUM], r0["ca"]["reliable"].values)
    level_score = lambda X: (lambda p: p if pl is None else pl.predict_proba(lg(p))[:, 1])(lp.predict_proba(X[LEVEL_NUM])[:, 1])
    def load(p):
        d = pd.read_csv(p); d["hr"] = pd.to_datetime(d["hr"], utc=True); d["id"] = d["id"].astype(str); d["monitor_id"] = d["monitor_id"].astype(str); return d
    wc, rub = load(a.wc), load(a.rub); rub = rub[rub["id"] == "180081"].copy(); wc = wc[(wc.monitor_id != "06-019-5001-3") | (wc["id"] == "18647")].copy()
    ext = []
    for nm, d, regs in (("WestCoast_pooled", wc, ("Non-smoky", "Smoky")), ("Rubidoux", rub, ("Non-smoky",))):
        d = d.reset_index(drop=True); d["level_score"] = level_score(d)
        for rg in regs:
            log(f"  bootstrapping {nm} / {rg} ...")
            S = {"full": d.score.values.astype(float), "level_only": d.level_score.values.astype(float)}
            pt, dr, n, nrel = ext_boot(d, rg, S, a.nboot, a.seed)
            for k in S:
                lo, hi = ci_(dr[(k, "rho")]); rlo, rhi = ci_(dr[(k, "roc")])
                rows.append(dict(kind="external", setting=f"{nm} / {rg}", model=k, n=n, n_reliable=nrel, rho=pt[(k, "rho")], rho_lo=lo, rho_hi=hi, roc_auc=pt[(k, "roc")], roc_lo=rlo, roc_hi=rhi))
            dl, dh = ci_(dr[("full", "rho")] - dr[("level_only", "rho")]); rl, rh = ci_(dr[("full", "roc")] - dr[("level_only", "roc")])
            rows.append(dict(kind="external", setting=f"{nm} / {rg}", model="paired_diff_full_minus_level", n=n, n_reliable=nrel, rho=pt[("full", "rho")] - pt[("level_only", "rho")], rho_lo=dl, rho_hi=dh,
                             roc_auc=pt[("full", "roc")] - pt[("level_only", "roc")], roc_lo=rl, roc_hi=rh))
            f_, l_ = rows[-3], rows[-2]; dd_ = rows[-1]
            def verdict(diff, lo, hi, better_positive):
                if lo != lo: return "cannot tell"
                if lo > 0 or hi < 0: return "FULL BETTER" if (diff > 0) == better_positive else "LEVEL-ONLY BETTER"
                return "no clear difference"
            summ.append(f"{nm} / {rg} (n={n}): Spearman full {f_['rho']:+.3f} [{f_['rho_lo']:+.3f},{f_['rho_hi']:+.3f}] vs level-only {l_['rho']:+.3f} [{l_['rho_lo']:+.3f},{l_['rho_hi']:+.3f}] -> "
                        f"{verdict(dd_['rho'], dd_['rho_lo'], dd_['rho_hi'], False)} (diff {dd_['rho']:+.3f} [{dd_['rho_lo']:+.3f},{dd_['rho_hi']:+.3f}]); "
                        f"ROC full {f_['roc_auc']:.3f} [{f_['roc_lo']:.3f},{f_['roc_hi']:.3f}] vs level-only {l_['roc_auc']:.3f} [{l_['roc_lo']:.3f},{l_['roc_hi']:.3f}] -> {verdict(dd_['roc_auc'], dd_['roc_lo'], dd_['roc_hi'], True)} "
                        f"(diff {dd_['roc_auc']:+.3f} [{dd_['roc_lo']:+.3f},{dd_['roc_hi']:+.3f}])")
    out = pd.DataFrame(rows); out.to_csv(os.path.join(a.out, pre + "level_only.csv"), index=False)
    log("\n================ SUMMARY (paired intervals: internal = mean ± 1.96 SE across 10 repeats; external = paired 24-h block bootstrap) ================")
    for s in summ: log(s)
    log(f"\nSaved {a.out}/{pre}level_only.csv and {pre}level_only_summary.txt  (total {time.time() - t0:.0f}s)")

main()
