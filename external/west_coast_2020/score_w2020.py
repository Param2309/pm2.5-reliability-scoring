#!/usr/bin/env python3
"""Score paired_w2020.csv (West Coast smoke, 7-25 Sep 2020) with the FROZEN pipeline. No refit, no retuning, nothing is trained.

Same construction as the Missoula / NYC Bronx / LA runs (run_cf1_analysis.py):
  Stage 1  frozen rf_model.pkl on [RH, PA, id] through scaler.pkl and column_transformer.pkl (unseen ids -> all-zero id columns)
  features lag1/dp/rolling on PA and Predicted_ref per sensor (shift by row), regime = Predicted_ref > 35.4
  Stage 2  frozen trust_pipe.pkl then platt.pkl on the logit
  label    reliable = |ref - Predicted_ref| <= delta, delta = 0.981723249508968 (non-smoky) / 9.697103519851735 (smoky), from training
PA in the file is the A/B mean of pm2.5_cf_1, which is the training PA field.

Rows used: ab_ok == True (main). The scripts of the earlier external runs applied no A/B agreement filter (the A/B channels were only used
for the -|A-B| baseline), so an all-rows version is also scored (point estimates, no bootstrap) as a sensitivity row.
Rows with ref > 500 are removed AFTER scoring (so lag features are not disturbed) in the "ref<=500" variant.

Usage
  python score_w2020.py --csv paired_w2020.csv --models MODELS_DIR --check rf_pred_Missoula.csv --out out_w2020 [--train WildFireData.csv]
MODELS_DIR holds the frozen rf_model.pkl (833 MB original), scaler.pkl, column_transformer.pkl, trust_pipe.pkl, platt.pkl.
The script stops if the frozen Stage 1 does not reproduce the stored Predicted_ref in --check to 1e-3.
Writes out_w2020/: scored_w2020.csv, summary_w2020.csv, overlap_check.txt, summary.txt. Does not read or need any API key."""
import argparse, os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, joblib
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score

DELTA = {"Non-smoky": 0.981723249508968, "Smoky": 9.697103519851735}
NUM = ["RH", "PA", "Predicted_ref", "lag1_PA", "dp_PA", "lag1_pred", "dp_pred", "corr", "abs_dp_PA", "abs_dp_pred",
       "PA_roll_mean_3", "PA_roll_std_3", "pred_roll_mean_3", "pred_roll_std_3", "PA_roll_mean_6", "PA_roll_std_6",
       "pred_roll_mean_6", "pred_roll_std_6"]
COLS = ["AQI_group"] + NUM
SHARED = "06-019-5001-3"

def add_features(d):
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
    d["AQI_group"] = pd.cut(d["Predicted_ref"], bins=[-np.inf, 35.4, np.inf], labels=["Non-smoky", "Smoky"]).astype(str)
    return d

def sp(a, b):
    if len(a) < 3: return np.nan
    ra, rb = rankdata(a), rankdata(b)
    if ra.std() == 0 or rb.std() == 0: return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])

def metrics(score, err, y):
    """n, base rate, PR-AUC, excess, ROC-AUC, Spearman(score, |Stage 1 error|)."""
    n = len(y); base = float(np.mean(y)) if n else np.nan
    ok = n > 0 and 0 < y.sum() < n
    pr = float(average_precision_score(y, score)) if ok else np.nan
    ro = float(roc_auc_score(y, score)) if ok else np.nan
    return {"n": n, "n_reliable": int(y.sum()) if n else 0, "base_rate": base, "pr_auc": pr, "excess_pr_auc": pr - base if ok else np.nan,
            "roc_auc": ro, "spearman_score_vs_abs_err": sp(score, err)}

def unit_arrays(d):
    cell = pd.factorize(d["monitor_id"].astype(str) + "|" + d["hr"].astype(str))[0]
    return {"score": d["score"].values, "err": d["err"].values, "y": d["reliable"].values, "smoky": (d["AQI_group"] == "Smoky").values,
            "cell": cell, "mon": d["monitor_id"].astype(str).values}

def bootstrap_smoky(d, nboot, seed):
    """Resample hours within monitor (all sensor rows of an hour move together); 95% percentile intervals for the smoky group."""
    A = unit_arrays(d); ncell = A["cell"].max() + 1
    mons = np.unique(A["mon"]); cells_of = {m: np.unique(A["cell"][A["mon"] == m]) for m in mons}
    rng = np.random.default_rng(seed); out = {k: [] for k in ("pr_auc", "excess_pr_auc", "roc_auc", "spearman_score_vs_abs_err", "base_rate")}
    for _ in range(nboot):
        cnt = np.zeros(ncell, int)
        for m in mons:
            cs = cells_of[m]; cnt[cs] = np.bincount(rng.integers(0, len(cs), len(cs)), minlength=len(cs))
        w = cnt[A["cell"]]; idx = np.repeat(np.arange(len(w)), w); idx = idx[A["smoky"][idx]]
        if len(idx) < 5: continue
        r = metrics(A["score"][idx], A["err"][idx], A["y"][idx])
        for k in out: out[k].append(r[k])
    ci = {}
    for k, v in out.items():
        v = np.array(v, float); v = v[~np.isnan(v)]
        ci[k] = (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), len(v)) if len(v) >= 100 else (np.nan, np.nan, len(v))
    return ci

def build_units(d):
    """sensors individually, monitors (shared monitor = both sensors pooled), pooled all rows, pooled with one sensor per monitor."""
    U = []
    for s, g in d.groupby("id"): U.append((f"sensor {s}", "sensor", g))
    for m, g in d.groupby("monitor_id"): U.append((f"monitor {m}" + (" (2 sensors)" if g["id"].nunique() > 1 else ""), "monitor", g))
    U.append(("pooled, all rows", "pooled", d))
    first = d.groupby("monitor_id")["id"].min()
    U.append(("pooled, one sensor per monitor", "pooled", d[d["id"] == d["monitor_id"].map(first)]))
    return U

def table(d, label, nboot, seed, boot):
    rows = []
    for name, level, g in build_units(d):
        sm = g["AQI_group"] == "Smoky"
        parts = {"Non-smoky": g[~sm], "Smoky": g[sm], "All": g}
        ci = bootstrap_smoky(g, nboot, seed) if boot and sm.sum() >= 10 else None
        for grp, x in parts.items():
            r = {"variant": label, "unit": name, "level": level, "group": grp, **metrics(x["score"].values, x["err"].values, x["reliable"].values)}
            if grp == "Smoky" and ci:
                for k, (lo, hi, nv) in ci.items():
                    r[k + "_lo"], r[k + "_hi"] = lo, hi
                r["boot_valid_draws"] = ci["pr_auc"][2]
            rows.append(r)
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True); ap.add_argument("--models", required=True); ap.add_argument("--check", required=True,
        help="rf_pred_Missoula.csv (hr,id,RH,PA,ref,Predicted_ref) used to confirm the frozen Stage 1 is the original")
    ap.add_argument("--out", default="out_w2020"); ap.add_argument("--train", default=None, help="WildFireData.csv (optional, for the overlap check)")
    ap.add_argument("--nboot", type=int, default=5000); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--s1-tol", type=float, default=1e-3); ap.add_argument("--continue-anyway", action="store_true")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    logf = open(os.path.join(a.out, "summary.txt"), "w")
    def log(s=""): print(s, flush=True); logf.write(s + "\n"); logf.flush()
    M = lambda f: os.path.join(a.models, f)
    rf = joblib.load(M("rf_model.pkl")); sc = joblib.load(M("scaler.pkl")); ct = joblib.load(M("column_transformer.pkl"))
    tp = joblib.load(M("trust_pipe.pkl")); platt = joblib.load(M("platt.pkl"))
    for _, tr, _c in getattr(ct, "transformers_", []):
        if hasattr(tr, "handle_unknown"): tr.handle_unknown = "ignore"
    stage1 = lambda df: np.asarray(rf.predict(sc.transform(ct.transform(df[["RH", "PA", "id"]].assign(id=df["id"].astype(str)))))).reshape(-1).astype(float)
    def stage2(d):
        p = tp.predict_proba(d[COLS])[:, 1]; e = 1e-6
        return platt.predict_proba(np.log(np.clip(p, e, 1 - e) / np.clip(1 - p, e, 1 - e)).reshape(-1, 1))[:, 1]

    # frozen Stage 1 check
    ck = pd.read_csv(a.check); dd = np.abs(stage1(ck.assign(id=ck["id"].astype(str))) - ck["Predicted_ref"].values)
    log(f"[stage1 check] frozen Stage 1 on {os.path.basename(a.check)} vs stored Predicted_ref: max|diff|={dd.max():.2e}, median={np.median(dd):.2e}")
    if dd.max() > a.s1_tol:
        log("STOP: Stage 1 as loaded does not reproduce the stored Predicted_ref. These are not the original frozen files.")
        if not a.continue_anyway: sys.exit(1)
        log("   (--continue-anyway set: numbers below are NOT frozen-pipeline results)")

    raw = pd.read_csv(a.csv)
    need = {"sensor_id", "monitor_id", "time_utc", "pa_a", "pa_b", "PA", "RH", "ab_ok", "ref"}
    if need - set(raw.columns): sys.exit(f"missing columns {need - set(raw.columns)}")
    raw["hr"] = pd.to_datetime(raw["time_utc"], utc=True); raw["id"] = raw["sensor_id"].astype(str); raw["ab_ok"] = raw["ab_ok"].astype(str).str.lower().eq("true")
    log(f"\nfile rows={len(raw)}  ab_ok True={raw.ab_ok.sum()}  False={(~raw.ab_ok).sum()}  sensors={raw.id.nunique()}  duplicate (sensor,hour)={raw.duplicated(['id','hr']).sum()}")
    dif = (raw["PA"] - (raw["pa_a"] + raw["pa_b"]) / 2).abs(); log(f"PA vs mean(pa_a, pa_b): max|diff|={dif.max():.3f}  (cf_1 A/B mean, rounded to 0.1)")
    ab = (raw.pa_a - raw.pa_b).abs(); pct = ab / ((raw.pa_a + raw.pa_b) / 2).replace(0, np.nan)
    flagged = (ab >= 5) & (pct >= 0.61)
    log(f"ab_ok vs the rule |A-B|>=5 and >=61% apart: agrees on {((~flagged) == raw.ab_ok).mean()*100:.1f}% of rows")
    sh = raw[raw.monitor_id == SHARED]; both = sh.groupby("hr")["id"].nunique()
    log(f"shared monitor {SHARED}: sensors {sorted(sh.id.unique())}, hours with both sensors {int((both == 2).sum())} of {len(both)}")

    def score(df, tag):
        df = df.dropna(subset=["hr", "id", "RH", "PA", "ref"]).copy()
        df["Predicted_ref"] = stage1(df); df = add_features(df)
        gap = df.groupby("id")["hr"].diff().dt.total_seconds().div(3600)
        n_gap = int((gap > 1).sum())
        df = df.dropna(subset=["lag1_PA", "dp_PA", "lag1_pred", "dp_pred"]).reset_index(drop=True)
        df["score"] = stage2(df); df["err"] = (df["ref"] - df["Predicted_ref"]).abs()
        df["reliable"] = (df["err"] <= df["AQI_group"].map(DELTA)).astype(int)
        log(f"[{tag}] scored rows={len(df)} (first row of each sensor dropped for lag features); rows whose lag spans a gap >1 h: {n_gap}")
        return df

    main_d = score(raw[raw.ab_ok], "ab_ok == True"); main_d["monitor_id"] = main_d["monitor_id"].astype(str)
    all_d = score(raw, "all rows (sensitivity)"); all_d["monitor_id"] = all_d["monitor_id"].astype(str)
    main_d.to_csv(os.path.join(a.out, "scored_w2020.csv"), index=False)

    log("\nPer sensor: rows, ref max, PA max, rows ref>500, rows predicted smoky, Stage 1 R2 and MAE vs raw-PA MAE")
    for s, g in main_d.groupby("id"):
        r2 = 1 - ((g.ref - g.Predicted_ref) ** 2).sum() / ((g.ref - g.ref.mean()) ** 2).sum()
        log(f"  {s:>6} mon {g.monitor_id.iloc[0]}: n={len(g)} ref_max={g.ref.max():.0f} PA_max={g.PA.max():.0f} ref>500={int((g.ref>500).sum())} "
            f"smoky={int((g.AQI_group=='Smoky').sum())} S1_R2={r2:.3f} S1_MAE={g.err.mean():.2f} rawPA_MAE={(g.ref-g.PA).abs().mean():.2f}")

    t = [table(main_d, "main: ab_ok, all ref", a.nboot, a.seed, True),
         table(main_d[main_d.ref <= 500], "ab_ok, ref<=500", a.nboot, a.seed, True),
         table(all_d, "sensitivity: all rows, no A/B filter", a.nboot, a.seed, False)]
    res = pd.concat(t, ignore_index=True); res.to_csv(os.path.join(a.out, "summary_w2020.csv"), index=False)

    f = lambda v: "   nan" if v != v else f"{v:6.3f}"
    for var in res.variant.unique():
        log(f"\n=========== {var} ===========")
        log(f"{'unit':38s}{'group':10s}{'n':>6s}{'base':>7s}{'PR':>7s}{'excess':>8s}{'ROC':>7s}{'rho':>7s}   smoky 95% CI (PR | excess | ROC | rho)")
        for _, r in res[res.variant == var].iterrows():
            ci = ""
            if r.group == "Smoky" and "pr_auc_lo" in res.columns and r.get("pr_auc_lo") == r.get("pr_auc_lo"):
                ci = "   " + " | ".join(f"[{r[k+'_lo']:+.3f},{r[k+'_hi']:+.3f}]" for k in ("pr_auc", "excess_pr_auc", "roc_auc", "spearman_score_vs_abs_err"))
            log(f"{r.unit:38s}{r.group:10s}{int(r.n):6d}{f(r.base_rate):>7s}{f(r.pr_auc):>7s}{f(r.excess_pr_auc):>8s}{f(r.roc_auc):>7s}{f(r.spearman_score_vs_abs_err):>7s}{ci}")

    log("\n=========== PLAIN VERDICT (smoky, main variant): does the score rank above the base rate? ===========")
    m = res[(res.variant == res.variant.unique()[0]) & (res.group == "Smoky")]
    for _, r in m.iterrows():
        if r.n_reliable < 10 or r.n - r.n_reliable < 10: v = f"too few readings of one class to say (n={int(r.n)}, reliable={int(r.n_reliable)}, fewer than 10 in a class)"
        elif r.get("excess_pr_auc_lo") == r.get("excess_pr_auc_lo"):
            lo, hi = r.excess_pr_auc_lo, r.excess_pr_auc_hi
            v = "ABOVE base rate (interval excludes 0)" if lo > 0 else "BELOW base rate (interval excludes 0)" if hi < 0 else "NOT distinguishable from the base rate (interval spans 0)"
        else: v = "no interval"
        log(f"  {r.unit:38s} n={int(r.n):4d} base={r.base_rate:.3f} PR={r.pr_auc:.3f} excess={r.excess_pr_auc:+.3f}  -> {v}")

    # overlap with training sensors
    ov = open(os.path.join(a.out, "overlap_check.txt"), "w")
    def lo_(s=""): log(s); ov.write(s + "\n")
    lo_("\n=========== OVERLAP WITH THE 27 TRAINING SENSORS ===========")
    if a.train and os.path.exists(a.train):
        w = pd.read_csv(a.train, parse_dates=["hr"]); tid = set(w["id"].astype(str)); new = set(raw["id"])
        lo_(f"by id: training ids {len(tid)}, new ids {sorted(new)}, intersection = {sorted(tid & new)}")
        s = w[(w.hr >= "2020-09-07") & (w.hr < "2020-09-26")]
        lo_("training sensors with rows in the same window (7-25 Sep 2020):")
        for (c, i), g in s.groupby(["city", "id"]): lo_(f"   {c:15s} {i:20s} rows={len(g):4d} ref_max={g.ref.max():.0f}")
        lo_("The training file has no coordinates or monitor ids, so location cannot be compared from it. Run locate_w2020.py for the new sensors' coordinates.")
    else: lo_("(pass --train WildFireData.csv to print this check)")
    log("\nDone. Files in " + a.out)

main()
