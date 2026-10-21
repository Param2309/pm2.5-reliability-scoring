#!/usr/bin/env python3
"""Steps 3-6: rerun the frozen external validation with pm2.5_cf_1 instead of pm2.5_atm.
No retraining anywhere. Reads raw JSON from pull_purpleair_history.py. Writes only under --out.
Does not touch or need the API key.

Required inputs
  --raw      external_cf1/raw           (raw/<Missoula|NYC_Bronx|LA>/*.json)
  --ext      folder with Missoula_Wildfire_2021.csv, NYC_Bronx_June2023.csv, LA_Wildfire_Jan2025.csv  (hr,id,RH,PA,ref)
             and rf_pred_Missoula.csv, rf_pred_NYC_Bronx.csv, rf_pred_LA.csv (hr,id,RH,PA,ref,Predicted_ref)
  --models   folder with the FROZEN Stage 1 files rf_model.pkl, scaler.pkl, column_transformer.pkl
             and the frozen Stage 2 files trust_pipe.pkl, platt.pkl
"""
import argparse, glob, json, os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, joblib
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score, r2_score, mean_absolute_error

SITES = {"Missoula": ("Missoula_Wildfire_2021.csv", "rf_pred_Missoula.csv"),
         "NYC_Bronx": ("NYC_Bronx_June2023.csv", "rf_pred_NYC_Bronx.csv"),
         "LA": ("LA_Wildfire_Jan2025.csv", "rf_pred_LA.csv")}
CANON_DELTA = {"Non-smoky": 0.981723249508968, "Smoky": 9.697103519851735}
NUM_FEATS = ["RH","PA","Predicted_ref","lag1_PA","dp_PA","lag1_pred","dp_pred","corr","abs_dp_PA","abs_dp_pred",
             "PA_roll_mean_3","PA_roll_std_3","pred_roll_mean_3","pred_roll_std_3",
             "PA_roll_mean_6","PA_roll_std_6","pred_roll_mean_6","pred_roll_std_6"]
COLS = ["AQI_group"] + NUM_FEATS
NBOOT, BLOCK, SEED = 5000, 24, 42

def add_features(d):
    d = d.sort_values(["id","hr"]).reset_index(drop=True)
    d["lag1_PA"] = d.groupby("id")["PA"].shift(1); d["dp_PA"] = d["PA"] - d["lag1_PA"]
    d["lag1_pred"] = d.groupby("id")["Predicted_ref"].shift(1); d["dp_pred"] = d["Predicted_ref"] - d["lag1_pred"]
    d["corr"] = d["Predicted_ref"] - d["PA"]
    d["abs_dp_PA"] = d["dp_PA"].abs(); d["abs_dp_pred"] = d["dp_pred"].abs()
    for w in (3, 6):
        d[f"PA_roll_mean_{w}"] = d.groupby("id")["PA"].transform(lambda s: s.rolling(w, min_periods=1).mean())
        d[f"PA_roll_std_{w}"] = d.groupby("id")["PA"].transform(lambda s: s.rolling(w, min_periods=2).std())
        d[f"pred_roll_mean_{w}"] = d.groupby("id")["Predicted_ref"].transform(lambda s: s.rolling(w, min_periods=1).mean())
        d[f"pred_roll_std_{w}"] = d.groupby("id")["Predicted_ref"].transform(lambda s: s.rolling(w, min_periods=2).std())
    rs = [c for c in d.columns if "roll_std" in c]; d[rs] = d[rs].fillna(0.0)
    d["AQI_group"] = pd.cut(d["Predicted_ref"], bins=[-np.inf, 35.4, np.inf], labels=["Non-smoky","Smoky"]).astype(str)
    return d

def load_raw(site, rawdir):
    rows = []
    for f in sorted(glob.glob(os.path.join(rawdir, site, "*.json"))):
        j = json.load(open(f)); df = pd.DataFrame(j["data"], columns=j["fields"]); rows.append(df)
    if not rows: sys.exit(f"No raw JSON for {site} in {rawdir}")
    r = pd.concat(rows).drop_duplicates("time_stamp").sort_values("time_stamp").reset_index(drop=True)
    r["t"] = pd.to_datetime(r["time_stamp"], unit="s", utc=True)
    r["atm_ab"] = r[["pm2.5_atm_a","pm2.5_atm_b"]].mean(axis=1, skipna=True)
    r["cf1_ab"] = r[["pm2.5_cf_1_a","pm2.5_cf_1_b"]].mean(axis=1, skipna=True)
    r["atm_both"] = r[["pm2.5_atm_a","pm2.5_atm_b"]].notna().all(axis=1)
    r["cf1_both"] = r[["pm2.5_cf_1_a","pm2.5_cf_1_b"]].notna().all(axis=1)
    return r

def align(ex, raw, log):
    """Pick the whole-hour shift of the pulled timestamps that best reproduces the existing PA column."""
    best = None
    for sh in (-2, -1, 0, 1, 2):
        r = raw.copy(); r["hr"] = r["t"] + pd.Timedelta(hours=sh)
        m = ex.merge(r[["hr","atm_ab"]], on="hr", how="left")
        ok = m["atm_ab"].notna()
        if ok.sum() == 0: continue
        dif = (m.loc[ok, "PA"] - m.loc[ok, "atm_ab"]).abs()
        log(f"   shift {sh:+d} h: matched {ok.sum()}/{len(ex)}  median|diff|={dif.median():.4f}  max|diff|={dif.max():.4f}")
        key = (dif.median(), -ok.sum())
        if best is None or key < best[0]: best = (key, sh)
    return best[1]

# ---------- statistics ----------
def sp(a, b):
    if len(a) < 3: return np.nan
    ra, rb = rankdata(a), rankdata(b)
    if ra.std() == 0 or rb.std() == 0: return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])

def partial(s, e, pa):
    rs, re_, rp = rankdata(s), rankdata(e), rankdata(pa)
    res = lambda y: y - np.polyval(np.polyfit(rp, y, 1), rp)
    return float(np.corrcoef(res(rs), res(re_))[0, 1])

def run_stats(d, preds, do_boot=True):
    """d sorted by hr. preds: dict name->array. Returns list of dict rows."""
    d = d.sort_values("hr").reset_index(drop=True)
    n = len(d)
    cal = (d.ref - d.Predicted_ref).abs().values; raw = (d.ref - d.PA).abs().values
    errs = {"calibrated": cal, "raw_PA": raw}
    smoky = (d.Predicted_ref > 35.4).values
    masks = {"Non-smoky": ~smoky, "Smoky": smoky, "Overall": np.ones(n, bool)}
    P = {k: np.asarray(v, float) for k, v in preds.items()}
    keys = [(g, e, p) for g in masks for e in errs for p in P]
    point = {k: sp(P[k[2]][masks[k[0]]], errs[k[1]][masks[k[0]]]) for k in keys}
    boot = {k: [] for k in keys}; pboot = []
    ns = masks["Non-smoky"]
    ppoint = partial(P["score"][ns], cal[ns], d.PA.values[ns]) if ns.sum() > 5 else np.nan
    if do_boot:
        rng = np.random.default_rng(SEED); pool = np.arange(0, max(1, n - BLOCK + 1)); nb = int(np.ceil(n / BLOCK))
        for _ in range(NBOOT):
            st = rng.choice(pool, size=nb, replace=True)
            idx = np.concatenate([np.arange(s, min(s + BLOCK, n)) for s in st])[:n]
            for g, m in masks.items():
                mm = m[idx]
                if mm.sum() < 5:
                    for e in errs:
                        for p in P: boot[(g, e, p)].append(np.nan)
                    if g == "Non-smoky": pboot.append(np.nan)
                    continue
                ii = idx[mm]
                for e in errs:
                    for p in P: boot[(g, e, p)].append(sp(P[p][ii], errs[e][ii]))
                if g == "Non-smoky": pboot.append(partial(P["score"][ii], cal[ii], d.PA.values[ii]))
    out = []
    for k in keys:
        r = {"group": k[0], "error": k[1], "predictor": k[2], "n": int(masks[k[0]].sum()), "spearman": point[k]}
        if do_boot:
            v = np.array(boot[k]); v = v[~np.isnan(v)]
            r["ci_lo"], r["ci_hi"] = (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if len(v) else (np.nan, np.nan)
        out.append(r)
    pr = {"group": "Non-smoky", "error": "calibrated", "predictor": "score|PA (partial)", "n": int(ns.sum()), "spearman": ppoint}
    if do_boot:
        v = np.array(pboot); v = v[~np.isnan(v)]
        pr["ci_lo"], pr["ci_hi"] = (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if len(v) else (np.nan, np.nan)
    out.append(pr)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True); ap.add_argument("--ext", required=True); ap.add_argument("--models", required=True)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"))
    ap.add_argument("--ab-tol", type=float, default=0.15, help="max |A/B-mean atm PA - existing PA| allowed (existing values are rounded to 0.1)")
    ap.add_argument("--s1-tol", type=float, default=1e-3, help="max |recomputed - stored Predicted_ref| allowed when Stage 1 is applied to the existing PA")
    ap.add_argument("--continue-anyway", action="store_true", help="do not stop on a failed check (use only for smoke tests)")
    ap.add_argument("--no-boot", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    logf = open(os.path.join(a.out, "summary.txt"), "w")
    def log(s=""):
        print(s, flush=True); logf.write(s + "\n"); logf.flush()
    def stop(msg):
        log("STOP: " + msg)
        if not a.continue_anyway: sys.exit(1)
        log("   (--continue-anyway set, continuing)")

    M = lambda f: os.path.join(a.models, f)
    rf = joblib.load(M("rf_model.pkl")); sc = joblib.load(M("scaler.pkl")); ct = joblib.load(M("column_transformer.pkl"))
    tp = joblib.load(M("trust_pipe.pkl")); platt = joblib.load(M("platt.pkl"))
    for _, tr, _c in getattr(ct, "transformers_", []):
        if hasattr(tr, "handle_unknown"): tr.handle_unknown = "ignore"   # unseen external sensor ids -> all-zero id columns
    def stage1(df):
        X = df[["RH","PA","id"]].copy(); X["id"] = X["id"].astype(str)
        return np.asarray(rf.predict(sc.transform(ct.transform(X)))).reshape(-1).astype(float)
    def stage2(d):
        pr = tp.predict_proba(d[COLS])[:, 1]; e = 1e-6
        lg = np.log(np.clip(pr, e, 1 - e) / np.clip(1 - pr, e, 1 - e)).reshape(-1, 1)
        return platt.predict_proba(lg)[:, 1]

    allrows = []; summary = {}
    for site, (fx, fp) in SITES.items():
        log(f"\n================ {site} ================")
        ex = pd.read_csv(os.path.join(a.ext, fx)); ex["hr"] = pd.to_datetime(ex["hr"], utc=True); ex["id"] = ex["id"].astype(str)
        pred = pd.read_csv(os.path.join(a.ext, fp)); pred["hr"] = pd.to_datetime(pred["hr"], utc=True); pred["id"] = pred["id"].astype(str)
        raw = load_raw(site, a.raw)
        log(f"existing rows={len(ex)} (rf_pred rows={len(pred)}); pulled rows={len(raw)}; "
            f"pulled with one channel missing: atm {(~raw.atm_both).sum()}, cf1 {(~raw.cf1_both).sum()}")
        # step 3: reproduce the existing PA column from A/B atm
        sh = align(ex, raw, log); log(f"   -> using shift {sh:+d} h")
        raw["hr"] = raw["t"] + pd.Timedelta(hours=sh)
        keep = ["hr","atm_ab","cf1_ab","pm2.5_atm_a","pm2.5_atm_b","pm2.5_cf_1_a","pm2.5_cf_1_b","humidity","temperature"]
        m = ex.merge(raw[keep], on="hr", how="left")
        ok = m["atm_ab"].notna(); dif = (m.loc[ok, "PA"] - m.loc[ok, "atm_ab"]).abs()
        log(f"[step 3] A/B-mean atm vs existing PA: matched {ok.sum()}/{len(ex)}  max|diff|={dif.max():.4f}  mean|diff|={dif.mean():.4f}  rows with |diff|>0.1: {(dif>0.1).sum()}")
        rh = (m.loc[ok, "RH"] - m.loc[ok, "humidity"]).abs(); log(f"         pulled humidity vs existing RH: max|diff|={rh.max():.3f} (info only; existing RH is kept as model input)")
        if dif.max() > a.ab_tol or ok.mean() < 0.98: stop(f"{site}: A/B-averaged atm does not reproduce the existing PA within {a.ab_tol} (max {dif.max():.4f}, matched {ok.mean():.3f}).")
        # Stage 1 reproduction check on existing PA
        s1chk = stage1(pred); d1 = np.abs(s1chk - pred["Predicted_ref"].values)
        log(f"[stage1 check] frozen Stage 1 on existing PA vs stored Predicted_ref: max|diff|={d1.max():.2e}, median={np.median(d1):.2e}")
        if d1.max() > a.s1_tol: stop(f"{site}: Stage 1 as loaded does not reproduce stored Predicted_ref (max {d1.max():.3e}). Wrong model files or unseen-id handling differs.")
        # variants
        base = m[~m["atm_ab"].isna() & ~m["cf1_ab"].isna()].copy()
        log(f"rows with both pulled PA values available: {len(base)}/{len(ex)}")
        variants = {}
        v = pred.merge(m[["hr","pm2.5_atm_a","pm2.5_atm_b"]], on="hr", how="left")
        v["chA"], v["chB"] = v["pm2.5_atm_a"], v["pm2.5_atm_b"]; variants["atm_existing"] = v
        v = base.copy(); v["PA"] = v["atm_ab"]; v["Predicted_ref"] = stage1(v); v["chA"], v["chB"] = v["pm2.5_atm_a"], v["pm2.5_atm_b"]; variants["atm_repull"] = v
        v = base.copy(); v["PA"] = v["cf1_ab"]; v["Predicted_ref"] = stage1(v); v["chA"], v["chB"] = v["pm2.5_cf_1_a"], v["pm2.5_cf_1_b"]; variants["cf1"] = v
        for name, v in variants.items():
            d = v[["hr","id","RH","PA","ref","Predicted_ref","chA","chB"]].copy()
            d = d.dropna(subset=["hr","id","RH","PA","ref","Predicted_ref"])
            d = add_features(d).dropna(subset=["lag1_PA","dp_PA","lag1_pred","dp_pred"]).reset_index(drop=True)
            d["p"] = stage2(d)
            d["reliable"] = (( d.ref - d.Predicted_ref).abs() <= d["AQI_group"].map(CANON_DELTA)).astype(int)
            d["chdis"] = -(d.chA - d.chB).abs()
            d.to_csv(os.path.join(a.out, f"scored_{site}_{name}.csv"), index=False)
            auc = roc_auc_score(d.reliable, d.p) if d.reliable.nunique() == 2 else np.nan
            share = d.AQI_group.value_counts(normalize=True)
            info = {"rows": len(d), "stage1_R2": r2_score(d.ref, d.Predicted_ref), "stage1_MAE": mean_absolute_error(d.ref, d.Predicted_ref),
                    "reliable_rate": d.reliable.mean(), "stage2_ROC_AUC": auc,
                    "share_Non-smoky": share.get("Non-smoky", 0.0), "share_Smoky": share.get("Smoky", 0.0),
                    "share_ref_gt_35.4": float((d.ref > 35.4).mean()), "PA_mean": d.PA.mean(), "PA_max": d.PA.max(), "ref_max": d.ref.max()}
            summary[(site, name)] = info
            log(f"\n[{site} / {name}] " + ", ".join(f"{k}={val:.4f}" if isinstance(val, float) else f"{k}={val}" for k, val in info.items()))
            do_boot = (name != "atm_repull") and not a.no_boot
            if name == "cf1" or name == "atm_existing" or name == "atm_repull":
                d2 = d.sort_values("hr").reset_index(drop=True)   # run_stats works in hr order
                preds = {"score": d2.p.values, "-PA": -d2.PA.values, "-|A-B|": d2.chdis.values}
                for r in run_stats(d2, preds, do_boot):
                    r.update(site=site, variant=name); allrows.append(r)
    res = pd.DataFrame(allrows); res.to_csv(os.path.join(a.out, "spearman_all.csv"), index=False)
    pd.DataFrame(summary).T.to_csv(os.path.join(a.out, "site_metrics.csv"))
    # side-by-side print
    log("\n\n############ SIDE BY SIDE: atm_existing vs cf1 ############")
    for site in SITES:
        log(f"\n--- {site} ---")
        for k in ["rows","stage1_R2","stage1_MAE","reliable_rate","stage2_ROC_AUC","share_Non-smoky","share_Smoky"]:
            x, y = summary[(site,"atm_existing")][k], summary[(site,"cf1")][k]
            log(f"{k:18s} atm {x:10.4f}   cf1 {y:10.4f}")
        r = res[res.site == site]
        for g in ["Non-smoky","Smoky","Overall"]:
            for e in ["calibrated","raw_PA"]:
                for p in ["score","-PA","-|A-B|"]:
                    def cell(var):
                        q = r[(r.variant==var)&(r.group==g)&(r.error==e)&(r.predictor==p)].iloc[0]
                        ci = f" [{q.ci_lo:+.3f},{q.ci_hi:+.3f}]" if "ci_lo" in q and not np.isnan(q.get("ci_lo", np.nan)) else ""
                        return f"{q.spearman:+.3f}{ci} (n={q.n})"
                    log(f"{g:9s} vs {e:10s} {p:7s}  atm {cell('atm_existing')}   cf1 {cell('cf1')}")
        for var in ["atm_existing","cf1"]:
            q = r[(r.variant==var)&(r.predictor=="score|PA (partial)")].iloc[0]
            log(f"partial Spearman score~cal|PA non-smoky ({var}): {q.spearman:+.3f} [{q.get('ci_lo',np.nan):+.3f},{q.get('ci_hi',np.nan):+.3f}]")
    log("\nDone. Files in " + a.out)
main()
