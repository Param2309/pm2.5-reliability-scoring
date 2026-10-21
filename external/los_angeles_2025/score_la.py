#!/usr/bin/env python3
"""Score paired_la2025.csv (collocated Los Angeles sensors, 1-16 Jan 2025) with the FROZEN pipeline and put the original Huntington result next to it.
No refit, no retuning, nothing is trained. Same construction as score_w2020.py / run_cf1_analysis.py:
  Stage 1 frozen rf_model.pkl on [RH, PA, id] (PA = cf_1 A/B mean); features lag1/dp/rolling per sensor; regime = Predicted_ref > 35.4
  Stage 2 frozen trust_pipe.pkl then platt.pkl on the logit; reliable = |ref - Predicted_ref| <= delta (0.981723249508968 non-smoky / 9.697103519851735 smoky)
Rows: ab_ok == True (same rule as the West Coast set: not flagged when |A-B| >= 5 ug/m3 and >= 61% apart). An all-rows version is scored as a sensitivity.
The original Huntington LA run applied no A/B filter, but it flags 0 of its 374 rows under this rule, so it is the same set either way.
POOLED RULE (stated before results): one sensor per monitor = the sensor nearest to its monitor among sensors with >= 24 scored hours (data availability only, never score).
Every sensor that returned data is reported individually. Intervals: 24-hour moving-block bootstrap within monitor (blocks on clock hours, ceil(span/24) blocks per monitor,
all rows of a drawn hour taken), 5000 draws, seed 42, 95% percentile.
Usage:
  python score_la.py --csv paired_la2025.csv --cand candidates_la2025.csv --all-cands discover_la2025_all.csv --models MODELS_DIR \
      --check rf_pred_Missoula.csv --huntington scored_LA_cf1.csv --out out_la2025
Writes out_la2025/: scored_la2025.csv, summary_la2025.csv, comparison_la2025.csv, binwise_la2025.csv, summary.txt. No API key needed."""
import argparse, os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, joblib
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score

DELTA = {"Non-smoky": 0.981723249508968, "Smoky": 9.697103519851735}
NUM = ["RH", "PA", "Predicted_ref", "lag1_PA", "dp_PA", "lag1_pred", "dp_pred", "corr", "abs_dp_PA", "abs_dp_pred", "PA_roll_mean_3", "PA_roll_std_3",
       "pred_roll_mean_3", "pred_roll_std_3", "PA_roll_mean_6", "PA_roll_std_6", "pred_roll_mean_6", "pred_roll_std_6"]
COLS = ["AQI_group"] + NUM
EDGES = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001]
LABELS = ["0.0--0.5", "0.5--0.6", "0.6--0.7", "0.7--0.8", "0.8--0.9", "0.9--1.0"]
MIN_N, MIN_CLASS, MIN_HOURS = 10, 10, 24
T0 = pd.Timestamp("2020-01-01", tz="UTC")
HUNT_MON, HUNT_ID, HUNT_DIST = "06-037-1103-3", "12272", 3889

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

def auc(s, y):
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0: return np.nan
    return float((rankdata(s)[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

def mets(s, e, y):
    n = len(y); nrel = int(y.sum()) if n else 0; base = nrel / n if n else np.nan
    ok = n > 0 and nrel > 0 and nrel < n
    pr = float(average_precision_score(y, s)) if ok else np.nan
    return {"n": n, "n_reliable": nrel, "base_rate": base, "pr_auc": pr, "excess_pr_auc": pr - base if ok else np.nan,
            "roc_auc": auc(s, y) if ok else np.nan, "spearman": sp(s, e)}

def block_boot(g, grp, nboot, seed, block=24):
    """24-hour moving-block bootstrap within monitor; metrics of group grp on the drawn rows."""
    sel = (g["AQI_group"] == grp).values; mon = g["monitor_id"].astype(str).values
    hh = np.floor((g["hr"] - T0).dt.total_seconds().values / 3600).astype(int)
    s, e, y = g["score"].values, g["err"].values, g["reliable"].values.astype(int); parts = []
    for m in np.unique(mon):
        ix = np.where(mon == m)[0]; pos = hh[ix] - hh[ix].min(); L = int(pos.max()) + 1
        parts.append((ix, pos, L, max(0, L - block), int(np.ceil(L / block))))
    rng = np.random.default_rng(seed); out = {"spearman": [], "roc_auc": [], "excess_pr_auc": [], "base_rate": []}
    for _ in range(nboot):
        sl = []
        for ix, pos, L, hi, nb in parts:
            st = rng.integers(0, hi + 1, nb); d = np.zeros(L + block + 1, int)
            np.add.at(d, st, 1); np.add.at(d, st + block, -1); cnt = np.cumsum(d)[:L]
            sl.append(np.repeat(ix, cnt[pos]))
        idx = np.concatenate(sl); idx = idx[sel[idx]]
        if len(idx) < 5: continue
        r = mets(s[idx], e[idx], y[idx])
        for k in out: out[k].append(r[k])
    ci = {}
    for k, v in out.items():
        v = np.array(v, float); v = v[~np.isnan(v)]
        ci[k] = (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if len(v) >= 100 else (np.nan, np.nan)
    return ci

def too_few(r): return r["n"] < MIN_N or r["n_reliable"] < MIN_CLASS or r["n"] - r["n_reliable"] < MIN_CLASS

def unit_rows(g, name, dist, nboot, seed, boot=True):
    """long-format rows (All / Non-smoky / Smoky) for one unit, with block-bootstrap intervals for the two regimes."""
    rows = []
    for grp in ("All", "Non-smoky", "Smoky"):
        x = g if grp == "All" else g[g["AQI_group"] == grp]
        r = {"unit": name, "dist_m": dist, "group": grp, **mets(x["score"].values, x["err"].values, x["reliable"].values.astype(int))}
        r["too_few_for_auc"] = bool(too_few(r))
        if grp != "All" and boot and len(x) >= MIN_N:
            for k, (lo, hi) in block_boot(g, grp, nboot, seed).items(): r[k + "_lo"], r[k + "_hi"] = lo, hi
        rows.append(r)
    return rows

def bin_table(g):
    g = g.copy(); g["bin"] = pd.cut(g["score"], EDGES, labels=LABELS, right=False); rows = []
    for lab in LABELS:
        x = g[g["bin"] == lab]; n = len(x)
        if n < MIN_N: rows.append({"bin": lab, "n": n, "mean_score": np.nan, "reliab": np.nan, "gap": np.nan, "mae": np.nan, "too_few": True})
        else:
            ms, rl = float(x.score.mean()), float(x.reliable.mean())
            rows.append({"bin": lab, "n": n, "mean_score": ms, "reliab": rl, "gap": ms - rl, "mae": float(x.err.mean()), "too_few": False})
    return pd.DataFrame(rows)

def trend(t):
    p = t[~t.too_few].reset_index(drop=True)
    if len(p) < 2: return None
    d = np.diff(p.mae.values)
    return {"bottom_bin": p.loc[0, "bin"], "bottom_mae": p.loc[0, "mae"], "top_bin": p.loc[len(p) - 1, "bin"], "top_mae": p.loc[len(p) - 1, "mae"],
            "steps_down": int((d < 0).sum()), "steps": len(d), "monotone": bool((d < 0).all()), "n_populated": len(p)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True); ap.add_argument("--cand", required=True); ap.add_argument("--all-cands", default=None)
    ap.add_argument("--models", required=True); ap.add_argument("--check", required=True); ap.add_argument("--huntington", required=True)
    ap.add_argument("--out", default="out_la2025"); ap.add_argument("--nboot", type=int, default=5000); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--s1-tol", type=float, default=1e-3); ap.add_argument("--continue-anyway", action="store_true")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    logf = open(os.path.join(a.out, "summary.txt"), "w")
    buf = []
    def log(s=""): print(s, flush=True); logf.write(s + "\n"); logf.flush(); buf.append(s)
    O = lambda f: os.path.join(a.out, f)
    M = lambda f: os.path.join(a.models, f)
    rf, sc, ct, tp, platt = (joblib.load(M(f)) for f in ("rf_model.pkl", "scaler.pkl", "column_transformer.pkl", "trust_pipe.pkl", "platt.pkl"))
    for _, tr, _c in getattr(ct, "transformers_", []):
        if hasattr(tr, "handle_unknown"): tr.handle_unknown = "ignore"
    stage1 = lambda df: np.asarray(rf.predict(sc.transform(ct.transform(df[["RH", "PA", "id"]].assign(id=df["id"].astype(str)))))).reshape(-1).astype(float)
    def stage2(d):
        p = tp.predict_proba(d[COLS])[:, 1]; e = 1e-6
        return platt.predict_proba(np.log(np.clip(p, e, 1 - e) / np.clip(1 - p, e, 1 - e)).reshape(-1, 1))[:, 1]
    ck = pd.read_csv(a.check); dd = np.abs(stage1(ck.assign(id=ck["id"].astype(str))) - ck["Predicted_ref"].values)
    log(f"[stage1 check] frozen Stage 1 on {os.path.basename(a.check)} vs stored Predicted_ref: max|diff|={dd.max():.2e}, median={np.median(dd):.2e}")
    if dd.max() > a.s1_tol:
        log("STOP: Stage 1 as loaded does not reproduce the stored Predicted_ref. These are not the original frozen files.")
        if not a.continue_anyway: sys.exit(1)
        log("   (--continue-anyway set: numbers below are NOT frozen-pipeline results)")

    # ---- inputs
    raw = pd.read_csv(a.csv); raw["hr"] = pd.to_datetime(raw["time_utc"], utc=True); raw["id"] = raw["sensor_id"].astype(str)
    raw["ab_ok"] = raw["ab_ok"].astype(str).str.lower().eq("true"); raw["monitor_id"] = raw["monitor_id"].astype(str)
    cand = pd.read_csv(a.cand).drop_duplicates("sensor_index"); cand["id"] = cand.sensor_index.astype(str)
    dist = dict(zip(cand.id, cand.dist_m)); names = dict(zip(cand.id, cand["name"]))
    log(f"\npaired rows {len(raw)}  ab_ok True {int(raw.ab_ok.sum())}  False {int((~raw.ab_ok).sum())}  sensors with data {raw.id.nunique()} of {len(cand)} candidates")
    miss = sorted(set(cand.id) - set(raw.id))
    if a.all_cands and os.path.exists(a.all_cands):
        allc = pd.read_csv(a.all_cands); inact = allc[~allc.active]
        for _, r in inact.iterrows(): log(f"  not active in window, not pulled: {r.sensor_index} {r['name']} ({r.monitor_id}, {r.dist_m} m, created {r.date_created}, last seen {r.last_seen})")
    for i in miss: log(f"  candidate with NO paired data: {i} {names.get(i)} ({cand[cand.id == i].monitor_id.iloc[0]}, {dist.get(i)} m)")
    log("sensors that returned data: " + ", ".join(f"{i} {names.get(i)} ({dist.get(i)} m)" for i in sorted(raw.id.unique(), key=lambda x: dist.get(x, 1e9))))

    def score(df, tag):
        df = df.dropna(subset=["hr", "id", "RH", "PA", "ref"]).copy(); df["Predicted_ref"] = stage1(df); df = add_features(df)
        gap = df.groupby("id")["hr"].diff().dt.total_seconds().div(3600); n_gap = int((gap > 1).sum())
        df = df.dropna(subset=["lag1_PA", "dp_PA", "lag1_pred", "dp_pred"]).reset_index(drop=True)
        df["score"] = stage2(df); df["err"] = (df["ref"] - df["Predicted_ref"]).abs()
        df["reliable"] = (df["err"] <= df["AQI_group"].map(DELTA)).astype(int)
        log(f"[{tag}] scored rows={len(df)}; rows whose lag spans a gap >1 h: {n_gap}"); return df
    main_d = score(raw[raw.ab_ok], "ab_ok == True"); all_d = score(raw, "all rows (sensitivity)")
    main_d.to_csv(os.path.join(a.out, "scored_la2025.csv"), index=False)
    zero = sorted(set(raw.id) - set(main_d.id), key=lambda x: dist.get(x, 1e9))
    for i in zero: log(f"  sensor {i} {names.get(i)} ({dist.get(i)} m): 0 rows pass ab_ok (of {int((raw.id==i).sum())}); not in the ab_ok tables, shown only in the all-rows sensitivity")
    # shift check per sensor (PA moved by k hours against the monitor's reference, built from the paired file)
    shiftrows = []
    refs = raw.drop_duplicates(["monitor_id", "hr"]).set_index(["monitor_id", "hr"])["ref"]
    log("\n[hour-shift check, Pearson(PA, ref), ab_ok rows]  sensor: shift -1 / 0 / +1")
    for i in sorted(main_d.id.unique(), key=lambda x: (dist.get(x, 1e9), x)):
        g = main_d[main_d.id == i]; out_ = []
        for k in (-1, 0, 1):
            j = g.assign(hr=g.hr + pd.Timedelta(hours=k)).set_index(["monitor_id", "hr"]).join(refs.rename("ref_s"), how="inner").dropna(subset=["ref_s"])
            out_.append(np.corrcoef(j.PA, j.ref_s)[0, 1] if len(j) > 10 else np.nan)
        shiftrows.append({"sensor": i, "name": names.get(i), "dist_m": dist.get(i), "r_shift_minus1": out_[0], "r_shift_0": out_[1], "r_shift_plus1": out_[2], "best_shift": [-1, 0, 1][int(np.nanargmax(out_))]})
        log(f"   {i:>7s} {str(names.get(i))[:22]:22s}" + " / ".join(f"{v:6.3f}" for v in out_) + ("   <- best shift is not 0" if int(np.nanargmax(out_)) != 1 else ""))

    pd.DataFrame(shiftrows).to_csv(O("shift_check_la2025.csv"), index=False)
    # ---- original Huntington run, same frame layout
    h = pd.read_csv(a.huntington); h["hr"] = pd.to_datetime(h["hr"], utc=True)
    H = pd.DataFrame({"hr": h.hr, "id": HUNT_ID, "monitor_id": HUNT_MON, "ref": h.ref, "PA": h.PA, "Predicted_ref": h.Predicted_ref, "pa_a": h.chA, "pa_b": h.chB,
                      "AQI_group": h.AQI_group, "score": h.p, "reliable": h.reliable.astype(int)}); H["err"] = (H.ref - H.Predicted_ref).abs()
    r0 = mets(H.score.values, H.err.values, H.reliable.values)
    log(f"\n[Huntington original, reproduced from {os.path.basename(a.huntington)}] n={r0['n']} base={r0['base_rate']:.3f} ROC-AUC={r0['roc_auc']:.3f} (reported 0.511)")
    dist[HUNT_ID] = HUNT_DIST; names[HUNT_ID] = "Huntington (original, 3.89 km)"

    # ---- channel balance + units
    def chan(g):
        A, B = g.pa_a.values, g.pa_b.values
        return {"mean_A": float(A.mean()), "mean_B": float(B.mean()), "A_over_B": float(A.mean() / B.mean()) if B.mean() else np.nan,
                "A_gt_B_share": float((A > B).mean()), "median_A_over_B": float(np.nanmedian(A[B > 0] / B[B > 0])) if (B > 0).any() else np.nan,
                "share_absdiff_ge5": float((np.abs(A - B) >= 5).mean()),
                "PA_median": float(g.PA.median()), "PA_max": float(g.PA.max()), "r_PA_ref": float(np.corrcoef(g.PA, g.ref)[0, 1]) if len(g) > 5 else np.nan}
    sens = sorted(main_d.id.unique(), key=lambda x: (dist.get(x, 1e9), x))
    long_rows, comp = [], []
    for i in [HUNT_ID] + sens:
        g = H if i == HUNT_ID else main_d[main_d.id == i]
        nraw = 374 if i == HUNT_ID else int((raw.id == i).sum()); nab = 374 if i == HUNT_ID else int(((raw.id == i) & raw.ab_ok).sum())
        log(f"  scoring/bootstrapping sensor {i} {names.get(i)} (n={len(g)}) ...")
        rows = unit_rows(g, f"sensor {i}", dist.get(i), a.nboot, a.seed)
        for r in rows: r.update({"monitor_id": g.monitor_id.iloc[0], "sensor": i, "name": names.get(i), "role": "original" if i == HUNT_ID else "collocated"})
        long_rows += rows
        c = {"sensor": i, "name": names.get(i), "monitor": g.monitor_id.iloc[0], "dist_m": dist.get(i), "role": "original" if i == HUNT_ID else "collocated",
             "n_paired": nraw, "n_ab_ok": nab, "n_scored": len(g), "ref_max": float(g.ref.max()), **chan(g)}
        comp.append((c, rows))
    # pooled: one sensor per monitor, nearest to its monitor among sensors with >= 24 scored hours
    cnt = main_d.groupby("id").size(); elig = [i for i in sens if cnt[i] >= MIN_HOURS]
    keep = {}
    for i in elig:
        m = main_d[main_d.id == i].monitor_id.iloc[0]
        if m not in keep or (dist.get(i, 1e9), i) < (dist.get(keep[m], 1e9), keep[m]): keep[m] = i
    pooled = main_d[main_d.id.isin(set(keep.values()))].copy()
    log(f"\nPOOLED RULE: one sensor per monitor, nearest to the monitor among sensors with >= {MIN_HOURS} scored hours -> " + ", ".join(f"{m}: {i} {names.get(i)} ({dist.get(i)} m)" for m, i in keep.items()))
    log("  scoring/bootstrapping pooled ...")
    prow = unit_rows(pooled, "POOLED collocated (one sensor per monitor)", np.nan, a.nboot, a.seed)
    for r in prow: r.update({"monitor_id": ",".join(keep), "sensor": ",".join(keep.values()), "name": "pooled", "role": "pooled"})
    long_rows += prow; comp.append(({"sensor": "pooled", "name": "pooled nearest per monitor", "monitor": ",".join(keep), "dist_m": np.nan, "role": "pooled",
                                     "n_paired": np.nan, "n_ab_ok": np.nan, "n_scored": len(pooled), **chan(pooled)}, prow))
    res = pd.DataFrame(long_rows); res.to_csv(os.path.join(a.out, "summary_la2025.csv"), index=False)

    # ---- comparison table
    f = lambda v, p=3: "  nan" if v != v else f"{v:.{p}f}"
    ci = lambda r, k: "" if r.get(k + "_lo") != r.get(k + "_lo") or k + "_lo" not in r else f" [{r[k+'_lo']:+.2f},{r[k+'_hi']:+.2f}]"
    log("\n=========== COMPARISON: original Huntington vs collocated sensors (ab_ok rows, frozen pipeline) ===========")
    log("--- channels (scored rows) ---")
    log(f"{'sensor':34s}{'dist m':>7s}{'paired':>7s}{'ab_ok':>6s}{'scored':>7s}{'meanA':>7s}{'meanB':>7s}{'A/B':>6s}{'A>B':>6s}{'|A-B|>=5':>9s}{'PAmed':>7s}{'PAmax':>7s}{'r(PA,ref)':>10s}")
    flat = []
    for c, rows in comp:
        log(f"{(str(c['sensor'])+' '+str(c['name']))[:34]:34s}{f(c['dist_m'],0):>7s}{f(c['n_paired'],0):>7s}{f(c['n_ab_ok'],0):>6s}{c['n_scored']:7d}"
            f"{c['mean_A']:7.1f}{c['mean_B']:7.1f}{c['A_over_B']:6.2f}{c['A_gt_B_share']:6.2f}{c['share_absdiff_ge5']:9.2f}{c['PA_median']:7.0f}{c['PA_max']:7.0f}{c['r_PA_ref']:10.3f}"
            + ("   <- PA median > 500 ug/m3 while the reference never exceeds %.0f: not ambient air (lab/test exposure)" % c.get('ref_max', 0) if c['PA_median'] > 500 else ""))
        for r in rows: flat.append({**{k: v for k, v in c.items()}, **r})
    pd.DataFrame(flat).to_csv(os.path.join(a.out, "comparison_la2025.csv"), index=False)
    pd.DataFrame([c for c, _ in comp]).to_csv(O("channels_la2025.csv"), index=False)
    log("\n--- Stage 2 (n | base rate | ROC-AUC [95% block CI] | excess PR-AUC | Spearman(score,|err|) [95% block CI]) ---")
    for grp in ("All", "Non-smoky", "Smoky"):
        log(f"\n  {grp}")
        for _, r in res[res.group == grp].iterrows():
            tag = "  (fewer than 10 rows in one class: ROC/PR shown but unreliable)" if r.too_few_for_auc and grp != "All" else ""
            log(f"   {(str(r.sensor)+' '+str(r['name']))[:34]:34s} n={int(r.n):5d} base={f(r.base_rate)} ROC={f(r.roc_auc)}{ci(r,'roc_auc')} exPR={f(r.excess_pr_auc)}{ci(r,'excess_pr_auc')} rho={f(r.spearman)}{ci(r,'spearman')}{tag}")
    # all-rows sensitivity (point estimates)
    log("\n--- sensitivity: all rows, no A/B filter (point estimates) ---"); sens_rows = []
    for i in sorted(all_d.id.unique(), key=lambda x: (dist.get(x, 1e9), x)):
        g = all_d[all_d.id == i]
        for grp in ("Non-smoky", "Smoky"):
            x = g[g.AQI_group == grp]; r = mets(x.score.values, x.err.values, x.reliable.values)
            log(f"   {i} {grp:10s} n={r['n']:5d} base={f(r['base_rate'])} ROC={f(r['roc_auc'])} rho={f(r['spearman'])}")
            sens_rows.append({"sensor": i, "name": names.get(i), "group": grp, **r})
    pd.DataFrame(sens_rows).to_csv(O("sensitivity_allrows_la2025.csv"), index=False)

    # ---- bin-wise tables
    log("\n=========== BIN-WISE TABLES (same bins and layout as binwise_w2020.py; MAE = mean |ref - Predicted_ref|; < 10 readings = too few) ===========")
    out = []
    def show(t, title, key):
        log(f"\n{title}"); log(f"{'Score bin':10s}{'Mean score':>12s}{'Reliab.':>9s}{'Gap':>9s}{'MAE':>9s}{'n':>7s}")
        for _, r in t.iterrows():
            log(f"{r['bin']:10s}{'too few':>12s}{'':>9s}{'':>9s}{'':>9s}{int(r.n):7d}" if r.too_few else f"{r['bin']:10s}{r.mean_score:12.3f}{r.reliab:9.3f}{r.gap:+9.3f}{r.mae:9.3f}{int(r.n):7d}")
        t = t.copy(); t.insert(0, "unit", key[0]); t.insert(1, "group", key[1]); out.append(t); return trend(t)
    tr = {}
    for grp in ("Non-smoky", "Smoky"):
        tr[("pooled", grp)] = show(bin_table(pooled[pooled.AQI_group == grp]), f"POOLED collocated, one sensor per monitor, {grp} (n={int((pooled.AQI_group==grp).sum())})", ("pooled", grp))
    for i in [HUNT_ID] + sens:
        g = H if i == HUNT_ID else main_d[main_d.id == i]
        for grp in ("Non-smoky", "Smoky"):
            tr[(i, grp)] = show(bin_table(g[g.AQI_group == grp]), f"sensor {i} {names.get(i)} ({dist.get(i)} m), {grp} (n={int((g.AQI_group==grp).sum())})", (f"sensor {i}", grp))
    pd.concat(out, ignore_index=True).to_csv(os.path.join(a.out, "binwise_la2025.csv"), index=False)

    # ---- plain verdict
    v0 = len(buf); log("\n=========== PLAIN VERDICT ===========")
    log("Pooled rule: " + ", ".join(f"{i} ({names.get(i)}, {dist.get(i)} m)" for i in keep.values()) + ".")
    for grp in ("Non-smoky", "Smoky"):
        r = res[(res.group == grp) & (res.role == "pooled")].iloc[0]; hr_ = res[(res.group == grp) & (res.role == "original")].iloc[0]
        nb = int(np.ceil((pooled.hr.max() - pooled.hr.min()).total_seconds() / 3600 / 24)) if len(pooled) else 0
        log(f"\n{grp}: collocated pooled n={int(r.n)} (reliable {int(r.n_reliable)}, base {f(r.base_rate)}), about {nb} 24-hour blocks in the window; original Huntington n={int(hr_.n)}, ROC {f(hr_.roc_auc)}, rho {f(hr_.spearman)}.")
        if r.too_few_for_auc:
            log(f"   TOO SMALL TO TELL: fewer than {MIN_CLASS} readings in one class; ROC/PR not computed for pooled {grp}. rho={f(r.spearman)}{ci(r,'spearman')}.")
        else:
            lo, hi = r.get("roc_auc_lo", np.nan), r.get("roc_auc_hi", np.nan); elo, ehi = r.get("excess_pr_auc_lo", np.nan), r.get("excess_pr_auc_hi", np.nan)
            v = "no interval" if lo != lo else ("ABOVE chance/base rate (interval excludes 0.5)" if lo > 0.5 else "BELOW chance (interval excludes 0.5)" if hi < 0.5 else "NOT distinguishable from chance (interval spans 0.5)")
            log(f"   ROC-AUC {f(r.roc_auc)} [{f(lo,2)},{f(hi,2)}] -> {v}; excess PR-AUC {f(r.excess_pr_auc)} [{f(elo,2)},{f(ehi,2)}]; rho {f(r.spearman)} [{f(r.get('spearman_lo',np.nan),2)},{f(r.get('spearman_hi',np.nan),2)}].")
        t = tr[("pooled", grp)]
        if t is None: log("   MAE by bin: fewer than 2 populated bins, cannot say.")
        else: log(f"   MAE bottom bin {t['bottom_bin']} {t['bottom_mae']:.3f} -> top bin {t['top_bin']} {t['top_mae']:.3f} ({'lower' if t['top_mae'] < t['bottom_mae'] else 'NOT lower'} at the top); falls at {t['steps_down']} of {t['steps']} steps -> {'monotone' if t['monotone'] else 'NOT monotone'}.")
    cc = [c for c, _ in comp if c["role"] == "collocated"]
    if cc:
        log("\nChannel imbalance: Huntington A/B = %.2f. Collocated sensors: " % comp[0][0]["A_over_B"] + "; ".join(f"{c['sensor']} {c['A_over_B']:.2f}" for c in cc) + ".")
        big = [c for c in cc if c["A_over_B"] > 1.3]
        log("   " + ("the imbalance also appears at: " + ", ".join(str(c["sensor"]) for c in big) if big else "no collocated sensor shows an A/B ratio above 1.3, so the Huntington imbalance does not repeat."))
    open(O("verdict_la2025.txt"), "w").write("\n".join(buf[v0:]) + "\n")
    # ---- manifest: everything a reviewer needs to reproduce or check the run
    import hashlib, platform, datetime, glob, sklearn
    md5 = lambda fn: hashlib.md5(open(fn, "rb").read()).hexdigest()
    try: import xgboost; xv = xgboost.__version__
    except Exception: xv = "n/a"
    L = ["MANIFEST, collocated Los Angeles Jan 2025 rerun", f"run at (UTC): {datetime.datetime.now(datetime.timezone.utc):%Y-%m-%d %H:%M:%S}",
         "command: " + " ".join(sys.argv), f"python {platform.python_version()}, numpy {np.__version__}, pandas {pd.__version__}, scikit-learn {sklearn.__version__}, xgboost {xv}",
         f"seed {a.seed}, bootstrap draws {a.nboot}, 24-hour moving-block bootstrap within monitor", f"stage 1 check: max|diff| {dd.max():.2e} (tolerance {a.s1_tol})" + ("  [NOT the frozen files]" if dd.max() > a.s1_tol else "  [frozen Stage 1 reproduced]"),
         "", "frozen model files (md5, bytes):"]
    L += [f"  {f}  {md5(M(f))}  {os.path.getsize(M(f))}" for f in ("rf_model.pkl", "scaler.pkl", "column_transformer.pkl", "trust_pipe.pkl", "platt.pkl")]
    L += ["", "inputs (md5, bytes):"] + [f"  {fn}  {md5(fn)}  {os.path.getsize(fn)}" for fn in (a.csv, a.cand, a.check, a.huntington) + ((a.all_cands,) if a.all_cands and os.path.exists(a.all_cands) else ())]
    rawdir = os.path.join(os.path.dirname(os.path.abspath(a.csv)), "raw_la2025")
    if os.path.isdir(rawdir): L += ["", "raw PurpleAir history pulled by ext_sites.py pull (md5, bytes):"] + [f"  {os.path.basename(fn)}  {md5(fn)}  {os.path.getsize(fn)}" for fn in sorted(glob.glob(rawdir + "/*.csv"))]
    L += ["", "outputs (md5, bytes):"] + [f"  {os.path.basename(fn)}  {md5(fn)}  {os.path.getsize(fn)}" for fn in sorted(glob.glob(O("*"))) if not fn.endswith("MANIFEST_la2025.txt")]
    open(O("MANIFEST_la2025.txt"), "w").write("\n".join(L) + "\n")
    log("\nDone. Files in " + a.out + ": scored_la2025.csv, summary_la2025.csv, comparison_la2025.csv, channels_la2025.csv, shift_check_la2025.csv, sensitivity_allrows_la2025.csv, binwise_la2025.csv, verdict_la2025.txt, summary.txt, MANIFEST_la2025.txt")

main()
