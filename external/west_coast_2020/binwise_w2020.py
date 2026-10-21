#!/usr/bin/env python3
"""Bin-wise tables for the Sep 2020 West Coast set, same layout as the internal bin-wise tables:
   Score bin | Mean score | Reliab. | Gap (mean score - reliab.) | MAE | n        bins 0.0-0.5, 0.5-0.6, ..., 0.9-1.0
Reads out_w2020/scored_w2020.csv written by score_w2020.py (ab_ok rows, all ref, frozen pipeline). Nothing is re-scored here.
MAE = mean |ref - Predicted_ref| (Stage 1 corrected value against the EPA reference). No raw-PA numbers.
A bin with fewer than 10 readings prints "too few" instead of values.
Pooled = one sensor per monitor; monitor 06-019-5001-3 is represented by sensor 18647. Bootstrap: 24-hour moving-block bootstrap within each monitor, 5000 draws, seed 42.
Usage: python binwise_w2020.py --scored out_w2020/scored_w2020.csv --out out_w2020/binwise_w2020.csv"""
import argparse
import numpy as np, pandas as pd
from scipy.stats import rankdata

EDGES = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001]
LABELS = ["0.0--0.5", "0.5--0.6", "0.6--0.7", "0.7--0.8", "0.8--0.9", "0.9--1.0"]
MIN_N = 10
SHARED, KEEP = "06-019-5001-3", "18647"

def sp(a, b):
    if len(a) < 3: return np.nan
    ra, rb = rankdata(a), rankdata(b)
    if ra.std() == 0 or rb.std() == 0: return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])

def bin_table(g):
    g = g.copy(); g["bin"] = pd.cut(g["score"], EDGES, labels=LABELS, right=False)
    rows = []
    for lab in LABELS:
        x = g[g["bin"] == lab]; n = len(x)
        if n < MIN_N: rows.append({"bin": lab, "n": n, "mean_score": np.nan, "reliab": np.nan, "gap": np.nan, "mae": np.nan, "too_few": True})
        else:
            ms, rl = float(x.score.mean()), float(x.reliable.mean())
            rows.append({"bin": lab, "n": n, "mean_score": ms, "reliab": rl, "gap": ms - rl, "mae": float(x.err.mean()), "too_few": False})
    return pd.DataFrame(rows)

def show(t, title):
    print(f"\n{title}")
    print(f"{'Score bin':10s}{'Mean score':>12s}{'Reliab.':>9s}{'Gap':>9s}{'MAE':>9s}{'n':>7s}")
    for _, r in t.iterrows():
        if r.too_few: print(f"{r['bin']:10s}{'too few':>12s}{'':>9s}{'':>9s}{'':>9s}{int(r.n):7d}")
        else: print(f"{r['bin']:10s}{r.mean_score:12.3f}{r.reliab:9.3f}{r.gap:+9.3f}{r.mae:9.3f}{int(r.n):7d}")

def trend(t):
    """top vs bottom populated bin, and whether MAE falls monotonically across populated bins."""
    p = t[~t.too_few].reset_index(drop=True)
    if len(p) < 2: return {"note": "fewer than 2 populated bins"}
    d = np.diff(p.mae.values)
    return {"bottom_bin": p.loc[0, "bin"], "bottom_mae": p.loc[0, "mae"], "top_bin": p.loc[len(p) - 1, "bin"], "top_mae": p.loc[len(p) - 1, "mae"],
            "steps_down": int((d < 0).sum()), "steps": len(d), "monotone": bool((d < 0).all()), "n_populated": len(p),
            "worst_rise": float(d.max()) if (d > 0).any() else 0.0}

def boot_rho(g, nboot, seed, block=24):
    """24-hour moving-block bootstrap within each monitor. A block is 24 consecutive clock hours starting at a random hour of the monitor's
    span; ceil(span/24) blocks are drawn per monitor with replacement, and every row in a drawn hour is taken (both sensors of the shared
    monitor move together). Blocks are drawn per monitor and stacked, so each monitor keeps its own number of hours. Then the group's rho is taken."""
    mon = g["monitor_id"].astype(str).values; hh = np.floor((g["hr"] - pd.Timestamp("2020-01-01", tz="UTC")).dt.total_seconds().values / 3600).astype(int)
    s, e = g.score.values, g.err.values; parts = []
    for m in np.unique(mon):
        ix = np.where(mon == m)[0]; pos = hh[ix] - hh[ix].min(); L = int(pos.max()) + 1
        parts.append((ix, pos, L, max(0, L - block), int(np.ceil(L / block))))
    rng = np.random.default_rng(seed); out = []
    for _ in range(nboot):
        sel = []
        for ix, pos, L, hi, nb in parts:
            st = rng.integers(0, hi + 1, nb); d = np.zeros(L + block + 1, int)
            np.add.at(d, st, 1); np.add.at(d, st + block, -1); cnt = np.cumsum(d)[:L]
            sel.append(np.repeat(ix, cnt[pos]))
        idx = np.concatenate(sel)
        if len(idx) >= 5: out.append(sp(s[idx], e[idx]))
    v = np.array(out, float); v = v[~np.isnan(v)]
    return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if len(v) >= 100 else (np.nan, np.nan)

def boot_rho_group(g, grp, nboot, seed, block=24):
    """Same block bootstrap, but blocks are laid on the hour axis of all of g's rows and rho is taken on the rows of group grp inside the draw."""
    gg = g.copy(); gg["_sel"] = (gg["AQI_group"] == grp).values
    mon = gg["monitor_id"].astype(str).values; hh = np.floor((gg["hr"] - pd.Timestamp("2020-01-01", tz="UTC")).dt.total_seconds().values / 3600).astype(int)
    s, e, sel_ = gg.score.values, gg.err.values, gg["_sel"].values; parts = []
    for m in np.unique(mon):
        ix = np.where(mon == m)[0]; pos = hh[ix] - hh[ix].min(); L = int(pos.max()) + 1
        parts.append((ix, pos, L, max(0, L - block), int(np.ceil(L / block))))
    rng = np.random.default_rng(seed); out = []
    for _ in range(nboot):
        sl = []
        for ix, pos, L, hi, nb in parts:
            st = rng.integers(0, hi + 1, nb); d = np.zeros(L + block + 1, int)
            np.add.at(d, st, 1); np.add.at(d, st + block, -1); cnt = np.cumsum(d)[:L]
            sl.append(np.repeat(ix, cnt[pos]))
        idx = np.concatenate(sl); idx = idx[sel_[idx]]
        if len(idx) >= 5: out.append(sp(s[idx], e[idx]))
    v = np.array(out, float); v = v[~np.isnan(v)]
    return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if len(v) >= 100 else (np.nan, np.nan)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--scored", default="out_w2020/scored_w2020.csv"); ap.add_argument("--out", default="out_w2020/binwise_w2020_block.csv")
    ap.add_argument("--nboot", type=int, default=5000); ap.add_argument("--seed", type=int, default=42); a = ap.parse_args()
    d = pd.read_csv(a.scored); d["hr"] = pd.to_datetime(d["hr"], utc=True); d["id"] = d["id"].astype(str); d["monitor_id"] = d["monitor_id"].astype(str)
    pooled = d[(d.monitor_id != SHARED) | (d.id == KEEP)].copy()
    print(f"scored rows {len(d)}; pooled (one sensor per monitor, {SHARED} -> sensor {KEEP}) rows {len(pooled)} from {pooled.monitor_id.nunique()} monitors")
    out, verdicts = [], {}
    def add(t, section, group):
        t = t.copy(); t.insert(0, "group", group); t.insert(0, "section", section); out.append(t)

    print("\n############ A. POOLED, one sensor per monitor ############")
    for grp in ("Non-smoky", "Smoky"):
        g = pooled[pooled.AQI_group == grp]; t = bin_table(g); show(t, f"{grp}  (n={len(g)})"); add(t, "pooled", grp)
        tr = trend(t); verdicts[grp] = tr
    print("\n############ B. Spearman rho, score vs |ref - Predicted_ref|, 95% interval, 24-hour block bootstrap within monitor ############")
    rho = []
    for grp in ("Non-smoky", "Smoky"):
        g = pooled[pooled.AQI_group == grp]; r = sp(g.score.values, g.err.values); lo, hi = boot_rho_group(pooled, grp, a.nboot, a.seed)
        print(f"{grp:10s} n={len(g):5d}  rho={r:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]"); rho.append({"section": "spearman", "group": grp, "n": len(g), "rho": r, "ci_lo": lo, "ci_hi": hi})
        verdicts[grp]["rho"], verdicts[grp]["ci"] = r, (lo, hi)
    out.append(pd.DataFrame(rho))

    print("\n############ C. Top vs bottom bin, and monotonicity of MAE across populated bins ############")
    for grp in ("Non-smoky", "Smoky"):
        v = verdicts[grp]
        if "note" in v: print(f"{grp}: {v['note']}"); continue
        print(f"{grp}: bottom bin {v['bottom_bin']} MAE {v['bottom_mae']:.3f}  ->  top bin {v['top_bin']} MAE {v['top_mae']:.3f}  "
              f"({'lower' if v['top_mae'] < v['bottom_mae'] else 'NOT lower'} at the top)")
        print(f"   MAE falls at {v['steps_down']} of {v['steps']} steps between populated bins -> " +
              ("monotone decrease" if v["monotone"] else f"NOT monotone (largest rise between neighbouring bins {v['worst_rise']:.3f})"))
    out.append(pd.DataFrame([{"section": "trend", "group": g, **{k: (str(x) if isinstance(x, tuple) else x) for k, x in v.items()}} for g, v in verdicts.items()]))

    print("\n############ D. Smoky only, per monitor (shared monitor shown with sensor 18647) ############")
    for m in sorted(d.monitor_id.unique()):
        g = pooled[(pooled.monitor_id == m) & (pooled.AQI_group == "Smoky")]; t = bin_table(g)
        show(t, f"monitor {m}  Smoky (n={len(g)})"); add(t, f"monitor {m}", "Smoky")

    print("\n############ E. Spearman rho per sensor, Smoky, 95% interval (24-hour block bootstrap within the sensor's monitor) ############")
    srows = []
    for sid, g in d[d.AQI_group == "Smoky"].groupby("id"):
        gg = d[d.id == sid]; allg = gg  # blocks are laid on the sensor's own hours; smoky rows are selected inside the bootstrap
        sm = gg[gg.AQI_group == "Smoky"]
        r = sp(sm.score.values, sm.err.values); lo, hi = boot_rho_group(gg, "Smoky", a.nboot, a.seed)
        print(f"sensor {sid:>6} (monitor {gg.monitor_id.iloc[0]})  n={len(sm):4d}  rho={r:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]")
        srows.append({"section": "spearman_sensor", "group": "Smoky", "sensor": sid, "monitor": gg.monitor_id.iloc[0], "n": len(sm), "rho": r, "ci_lo": lo, "ci_hi": hi})
    out.append(pd.DataFrame(srows))

    print("\n############ PLAIN ANSWER (smoky, pooled) ############")
    v = verdicts["Smoky"]; lo, hi = v["ci"]
    if "note" in v: print(v["note"])
    else:
        dec = v["top_mae"] < v["bottom_mae"]
        sig = hi < 0 if hi == hi else False
        print(f"rho={v['rho']:+.3f} [{lo:+.3f},{hi:+.3f}]; top-bin MAE {v['top_mae']:.3f} vs bottom-bin MAE {v['bottom_mae']:.3f}; monotone: {v['monotone']} ({v['steps_down']}/{v['steps']} steps down).")
        if dec and sig and v["monotone"]: print("YES: in smoke, MAE decreases with the score, and the interval excludes zero.")
        elif dec and sig: print("PARTLY: MAE is lower at the top than at the bottom and rho is negative with an interval that excludes zero, but it is NOT monotone across bins.")
        elif dec: print("WEAK: MAE is lower in the top bin, but rho's interval includes zero, so the trend is not established.")
        else: print("NO: in smoke, MAE does not decrease with the score.")
    pd.concat(out, ignore_index=True).to_csv(a.out, index=False); print("\nSaved", a.out)

main()
