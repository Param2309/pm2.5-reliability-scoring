"""Summary of nested-tuning results: per regime mean over repeats with 95% t-interval, old (paper params) vs new (nested-tuned),
direction agreement, excess PR-AUC (ratio of means), ablation gap, best params (min/median/max) for Table V.
Usage: python summarize.py  (reads json files from OUT_DIR, writes summary.txt, summary.json, per_repeat.csv there)"""
import json, glob, sys, csv, numpy as np
import os
OUT_DIR = os.environ.get("NESTED_OUT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "nested_out")   # same folder as nested_tune.py
from scipy import stats
OUT = sys.argv[1] if len(sys.argv) > 1 else OUT_DIR
REGS = [r for r in ("R1", "R2", "R3") if glob.glob(f"{OUT}/{r}_rep*.json")]
R = {r: [json.load(open(f)) for f in sorted(glob.glob(f"{OUT}/{r}_rep*.json"), key=lambda s: int(s.split("rep")[-1].split(".")[0]))] for r in REGS}
for r in ("R1", "R3"):   # attach ablation passes (separate files *abl_rep*.json)
    for f in glob.glob(f"{OUT}/{r}abl_rep*.json"):
        j = json.load(open(f))
        for x in R.get(r, []):
            if x["rep"] == j["rep"]: x["abl"] = j["abl"]
def ci(x):
    x = np.asarray(x, float); n = len(x); m = x.mean(); h = stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    return float(m), float(m - h), float(m + h)
def agree(d):
    d = np.asarray(d); s = np.sign(d.mean()); return int((np.sign(d) == s).sum()) if s != 0 else 0
def col(rs, arm, ver, k): return np.array([r[arm][ver][k] for r in rs])
lines = []; P = lambda *a: lines.append(" ".join(str(x) for x in a)); S = {}
P("n repeats per regime:", {r: len(R[r]) for r in REGS})
for r in REGS:
    rs = R[r]; S[r] = {}
    P(f"\n=== {r} (full model) ===")
    for k in ("pr", "base", "roc", "brier", "ece"):
        o, n = col(rs, "full", "old", k), col(rs, "full", "new", k); d = n - o
        S[r][k] = dict(old=ci(o), new=ci(n), diff=ci(d), agree=agree(d), n=len(d))
        P(f"{k:6s} old {S[r][k]['old'][0]:.4f} [{S[r][k]['old'][1]:.4f},{S[r][k]['old'][2]:.4f}]  new {S[r][k]['new'][0]:.4f} [{S[r][k]['new'][1]:.4f},{S[r][k]['new'][2]:.4f}]  "
          f"diff {S[r][k]['diff'][0]:+.4f} [{S[r][k]['diff'][1]:+.4f},{S[r][k]['diff'][2]:+.4f}] agree {S[r][k]['agree']}/{len(d)}")
    exo = lambda v: (col(rs, "full", v, "pr").mean() - col(rs, "full", v, "base").mean()) / (1 - col(rs, "full", v, "base").mean())
    pe = lambda v: (col(rs, "full", v, "pr") - col(rs, "full", v, "base")) / (1 - col(rs, "full", v, "base"))
    S[r]["excess"] = dict(old=float(exo("old")), new=float(exo("new")), old_rep_ci=ci(pe("old")), new_rep_ci=ci(pe("new")))
    P(f"excess PR-AUC (ratio of means) old {exo('old'):.4f}  new {exo('new'):.4f}  (per-repeat t-interval old {ci(pe('old'))[1]:.3f},{ci(pe('old'))[2]:.3f}; new {ci(pe('new'))[1]:.3f},{ci(pe('new'))[2]:.3f})")
    P("tuning CV PR-AUC (inside 15 train sensors) mean", f"{np.mean([x['full']['cv_pr'] for x in rs]):.4f}", " tune rows mean", int(np.mean([x['full']['tune_rows'] for x in rs])))
    if all("abl" in x for x in rs):
        S[r]["gap"] = {}
        for k in ("pr", "roc"):
            go = col(rs, "full", "old", k) - col(rs, "abl", "old", k); gn = col(rs, "full", "new", k) - col(rs, "abl", "new", k)
            S[r]["gap"][k] = dict(old=ci(go), new=ci(gn), diff=ci(gn - go), pos_old=int((go > 0).sum()), pos_new=int((gn > 0).sum()), agree=agree(gn - go))
            P(f"ablation gap ({k}) old {ci(go)[0]:+.4f} [{ci(go)[1]:+.4f},{ci(go)[2]:+.4f}] pos {int((go>0).sum())}/10 | new {ci(gn)[0]:+.4f} [{ci(gn)[1]:+.4f},{ci(gn)[2]:+.4f}] pos {int((gn>0).sum())}/10 | new-old {ci(gn-go)[0]:+.4f} [{ci(gn-go)[1]:+.4f},{ci(gn-go)[2]:+.4f}]")
        for ver in ("old", "new"):
            P(f"  ablated PR {ver}: {ci(col(rs,'abl',ver,'pr'))[0]:.4f}")
# R2 vs R3 ordering
for r2 in [x for x in ("R2",) if x in R]:
    if "R3" in R and len(R[r2]) == len(R["R3"]):
        for v in ("old", "new"):
            d = col(R[r2], "full", v, "pr") - col(R["R3"], "full", v, "pr")
            P(f"\n{r2} - R3 PR-AUC ({v}): {ci(d)[0]:+.4f} [{ci(d)[1]:+.4f},{ci(d)[2]:+.4f}] repeats with R2>R3 {int((d>0).sum())}/10;  excess {r2} {S[r2]['excess'][v]:.4f} vs R3 {S['R3']['excess'][v]:.4f}")
            S.setdefault("R2minusR3", {})[f"{r2}_{v}"] = ci(d)
# params
PN = ["n_estimators", "learning_rate", "max_depth", "min_child_weight", "subsample", "colsample_bytree", "reg_lambda", "reg_alpha", "gamma", "scale_pos_weight"]
S["params"] = {}
def ptab(tag, rs_list, arm):
    allp = [x[arm]["best"] for rs in rs_list for x in rs]
    P(f"\n--- best params {tag} / {arm} (n={len(allp)}): min, median, max ---")
    out = {}
    for p in PN:
        v = np.array([q[p] for q in allp], float); out[p] = dict(min=float(v.min()), median=float(np.median(v)), max=float(v.max()))
        P(f"{p:18s} {v.min():.6g}  {np.median(v):.6g}  {v.max():.6g}")
    S["params"][f"{tag}/{arm}"] = out
for r in REGS:
    ptab(r, [R[r]], "full")
    if all("abl" in x for x in R[r]): ptab(r, [R[r]], "abl")
pool = [R[r] for r in ("R1", "R2", "R3") if r in R]
if len(pool) == 3: ptab("pooled R1+R2+R3", pool, "full")
# changes above 0.01
P("\n=== mean changes new-old with |diff| > 0.01 ===")
for r in REGS:
    for k in ("pr", "roc", "brier", "ece"):
        if abs(S[r][k]["diff"][0]) > 0.01: P(f"{r} {k}: {S[r][k]['diff'][0]:+.4f}")
    if abs(S[r]["excess"]["new"] - S[r]["excess"]["old"]) > 0.01: P(f"{r} excess: {S[r]['excess']['new']-S[r]['excess']['old']:+.4f}")
    for k, g in S[r].get("gap", {}).items():
        if abs(g["diff"][0]) > 0.01: P(f"{r} ablation gap {k}: {g['diff'][0]:+.4f}")
open(f"{OUT}/summary.txt", "w").write("\n".join(lines)); json.dump(S, open(f"{OUT}/summary.json", "w"), indent=1)
print("\n".join(lines))

# per-repeat table
with open(f"{OUT}/per_repeat.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["regime", "rep", "arm", "version", "pr", "base", "roc", "brier", "ece", "cv_pr"])
    for r in REGS:
        for x in R[r]:
            for arm in ("full", "abl"):
                if arm in x:
                    for v in ("old", "new"):
                        m = x[arm][v]; w.writerow([r, x["rep"], arm, v, m["pr"], m["base"], m["roc"], m["brier"], m["ece"], x[arm]["cv_pr"] if v == "new" else ""])
