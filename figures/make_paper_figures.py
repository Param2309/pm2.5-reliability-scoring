"""Regenerates Figs. 2, 3 and 4 of the paper. Labels and layout only; plotted values come from the saved results you point to with --results.
No tuning, seeds or retraining happen here. The saved results (stage2/pipeline.py) are: R1 original protocol, 10 sensor-disjoint splits
(test sensors GroupShuffleSplit seed 42+rep, calibration sensors seed 42+100+rep, XGBoost seed 42+rep), reliability-diagram bin edges
np.linspace(0, 1, 11), Pareto grid np.linspace(0, 1, 51).
Before plotting the two combined figures it prints the coverage and MAE at t = 0.5/0.7/0.8/0.9 for every model, next to the
paper's values, and stops if any known paper value does not match (override with --force).
Naming rule: "reliability score" everywhere; no "trust", "pm_cal" or code variable names in any label.
Usage (from the repository root): python figures/make_paper_figures.py   (reads results/ and data/rf_pred_all.csv, writes the PNGs next to this script)"""
import os, json, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
DPI = 300
UG = "µg/m³"
CANDS = [("RF", "Random Forest"), ("LR", "Linear Regression"), ("NN", "Neural Network"), ("BJ", "Barkjohn")]
AQI = ["Non-smoky", "Smoky"]
GRID = np.linspace(0.0, 1.0, 51)
MARKED = [0.5, 0.7, 0.8, 0.9]
EXPECT = {"RF": {0.5: (0.84, 0.005), 0.7: (0.64, 0.005), 0.8: (0.47, 0.005), 0.9: (0.24, 0.005)},
          "LR": {0.5: (0.77, 0.01), 0.9: (0.28, 0.005)}, "NN": {0.9: (0.05, 0.005)}, "BJ": {0.9: (0.02, 0.005)}}
# sizes in inches (about the printed width in the paper, so text is really 8 pt / 9 pt / 7 pt when printed)
SIZE = {"combined_reliability_diagrams.png": (5.4, 5.4 * 3153 / 2524), "combined_pareto_curves.png": (5.0, 5.0 * 1665 / 2078),
        "rf_scatter_ref_vs_pmcal.png": (3.4, 3.0), "rf_calibration_error_hist.png": (3.4, 3.0)}
STYLE = {"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7,
         "text.color": "black", "axes.labelcolor": "black", "xtick.color": "black", "ytick.color": "black", "axes.edgecolor": "black",
         "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white", "axes.linewidth": 0.6,
         "lines.linewidth": 1.1, "lines.markersize": 3.2, "grid.linewidth": 0.4}

def load(folder, key):
    with open(os.path.join(folder, f"{key}_results.json")) as f: return json.load(f)

def check_overlaps(fig, name):
    """Report overlapping text boxes (titles, labels, legend text, annotations, tick labels). Cannot check legend-vs-data overlap."""
    fig.canvas.draw(); r = fig.canvas.get_renderer(); items = []
    for ax in fig.axes:
        for tag, t in [("title", ax.title), ("xlabel", ax.xaxis.label), ("ylabel", ax.yaxis.label)]:
            if t.get_text() and t.get_visible(): items.append((tag, t.get_text(), t.get_window_extent(r)))
        for t in ax.get_xticklabels() + ax.get_yticklabels():
            if t.get_text() and t.get_visible(): items.append(("tick", t.get_text(), t.get_window_extent(r)))
        lg = ax.get_legend()
        if lg:
            for t in lg.get_texts(): items.append(("legend", t.get_text(), t.get_window_extent(r)))
        for t in ax.texts: items.append(("label", t.get_text(), t.get_window_extent(r)))
    bad = []
    for i in range(len(items)):
        for j in range(i):
            a, b = items[i], items[j]
            if a[0] == "tick" and b[0] == "tick": continue
            if a[2].overlaps(b[2]): bad.append(f"{a[0]} '{a[1]}' x {b[0]} '{b[1]}'")
    print(f"OVERLAP CHECK {name}: " + ("no overlapping text" if not bad else f"{len(bad)} overlaps: " + "; ".join(bad[:8])), flush=True)

def declutter(fig, texts):
    """Nudge t = labels apart (even ones up, odd ones down) until their boxes no longer overlap."""
    fig.canvas.draw(); r = fig.canvas.get_renderer()
    for _ in range(80):
        moved = False
        bbs = [t.get_window_extent(r).expanded(1.08, 1.2) for t in texts]
        for i in range(len(texts)):
            for j in range(i):
                if bbs[i].overlaps(bbs[j]):
                    dx, dy = texts[i].xyann; texts[i].xyann = (dx, dy + (3 if i % 2 == 0 else -3))
                    bbs[i] = texts[i].get_window_extent(r).expanded(1.08, 1.2); moved = True
        if not moved: break

def save(fig, figs, name):
    os.makedirs(figs, exist_ok=True)
    check_overlaps(fig, name)
    p = os.path.join(figs, name)
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig); print("wrote", p, flush=True)

def verify(folder):
    print("\nVERIFICATION of the saved results against the paper (folder: %s)" % folder)
    print("Pareto coverage axis = mean over the 10 splits (stage2/pipeline.py), same as the table.\n")
    print(f"{'model':6s}{'t':>5s}{'cov table (10-split mean)':>28s}{'cov plotted dot':>18s}{'MAE table':>11s}{'MAE plotted':>13s}{'paper cov':>11s}  status")
    ok = True
    for key, _ in CANDS:
        j = load(folder, key); p = j["pareto"]; cov = np.array(p["coverage"], float); mae = np.array(p["mae_mean"], float)
        for t in MARKED:
            i = int(np.argmin(np.abs(GRID - t))); tab = j["threshold_sweep"][str(t)]
            exp = EXPECT.get(key, {}).get(t)
            if exp is None: st, pv = "(no paper value)", ""
            else:
                good = abs(tab["coverage"] - exp[0]) <= exp[1]; ok &= good; st, pv = ("match" if good else "MISMATCH"), f"{exp[0]:.2f}"
            print(f"{key:6s}{t:5.1f}{tab['coverage']:28.4f}{cov[i]:18.4f}{tab['mae']:11.3f}{mae[i]:13.3f}{pv:>11s}  {st}")
    d0 = load(folder, "NN")["reldiag"]["cal"]["Non-smoky"]
    pts = [(x, y) for x, y in zip(d0["x_mean"], d0["y_mean"]) if y == y]
    print(f"\nNN non-smoky calibrated curve, first point: x={pts[0][0]:.3f}, y={pts[0][1]:.3f}")
    print("ALL KNOWN PAPER VALUES MATCH" if ok else "SOME PAPER VALUES DO NOT MATCH: this is not the run behind the paper's figures", flush=True)
    return ok

REPLAY = os.environ.get("REPLAY_DIR")      # folder with {RF,LR,NN,BJ}_replay.json from replay_nb.py (the notebook's own code, bin centres on x)

def replay_reldiag(key):
    j = json.load(open(os.path.join(REPLAY, f"{key}_replay.json"))); xc = j["centers"]; out = {"raw": {}, "cal": {}}
    for kind in ("raw", "cal"):
        for g in AQI:
            a = np.array(j[kind][g], float)
            with np.errstate(all="ignore"):
                import warnings; warnings.simplefilter("ignore")
                out[kind][g] = {"x_mean": xc, "y_mean": np.nanmean(a, axis=0).tolist(), "y_std": np.nanstd(a, axis=0).tolist()}
    return out

def fig_reliability(folder, figs):
    name = "combined_reliability_diagrams.png"
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(len(CANDS), len(AQI), figsize=SIZE[name], dpi=DPI, sharey=True, layout="constrained")
        for r, (key, title) in enumerate(CANDS):
            rd = load(folder, key)["reldiag"] if not REPLAY else replay_reldiag(key)
            for c, g in enumerate(AQI):
                ax = axes[r, c]
                ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect")
                for kind, marker, label in [("raw", "o", "Raw score"), ("cal", "s", "Platt-calibrated score")]:
                    d = rd[kind][g]
                    x = np.array(d["x_mean"], float); y = np.array(d["y_mean"], float); s = np.array(d["y_std"], float)
                    ax.plot(x, y, marker=marker, label=label); ax.fill_between(x, y - s, y + s, alpha=0.2)
                ax.set_title(f"{title}, {g}"); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.grid(alpha=0.3)
                ax.legend(loc="upper left", framealpha=0.85)
                if c == 0: ax.set_ylabel("Fraction of readings\nthat are reliable")
                if r == len(CANDS) - 1: ax.set_xlabel("Mean reliability score (bin)")
        save(fig, figs, name)

def fig_pareto(folder, figs):
    name = "combined_pareto_curves.png"
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 2, figsize=SIZE[name], dpi=DPI, layout="constrained")
        all_labels = []
        for ax, (key, title) in zip(axes.ravel(), CANDS):
            p = load(folder, key)["pareto"]
            cov = np.array(p["coverage"], float); m = np.array(p["mae_mean"], float); sd = np.array(p["mae_std"], float)
            ax.plot(cov, m, color="C0")
            ax.fill_between(cov, np.maximum(m - sd, 0.0), m + sd, alpha=0.2, color="C0")      # band clipped at MAE >= 0
            labs = []
            for k, t in enumerate(MARKED):
                i = int(np.argmin(np.abs(GRID - t)))
                if np.isnan(cov[i]) or np.isnan(m[i]): continue
                ax.scatter(cov[i], m[i], s=11, color="black", zorder=3)
                labs.append(ax.annotate(f"t={t:.1f}", (cov[i], m[i]), xytext=((-4 if cov[i] > 0.75 else 4), 4 if k % 2 == 0 else -9), textcoords="offset points",
                                        fontsize=7, ha=("right" if cov[i] > 0.75 else "left"), va="bottom", zorder=4))
            all_labels.append(labs)
            ax.set_title(title); ax.grid(alpha=0.3); ax.set_ylim(bottom=0)
            ax.set_xlabel("Coverage"); ax.set_ylabel(f"MAE ({UG})")
            ax.legend(handles=[Line2D([0], [0], color="C0", label="Mean ± 1 std"),
                               Line2D([0], [0], marker="o", color="black", linestyle="", markersize=3.2, label="reliability score ≥ t")],
                      loc="best", framealpha=0.85)
        fig.supxlabel("Coverage is the fraction of readings retained", fontsize=7)
        for labs in all_labels: declutter(fig, labs)
        save(fig, figs, name)

def test_rows(pred_csv):
    pred = pd.read_csv(pred_csv)
    _, ti = train_test_split(np.arange(len(pred)), test_size=0.2, random_state=42)     # same split as the repo's fig1 script
    return pred.iloc[ti]

def fig_scatter(pred_csv, figs):
    name = "rf_scatter_ref_vs_pmcal.png"; t = test_rows(pred_csv)
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=SIZE[name], dpi=DPI, layout="constrained")
        ax.scatter(t["ref"], t["Predicted_ref"], s=1.5, alpha=0.4, rasterized=True)
        ax.plot([0, 100], [0, 100], color="tab:blue", linewidth=1.0)
        ax.set_xlim(0, 100); ax.set_ylim(0, 100)
        ax.set_xlabel(f"EPA reference PM2.5 ({UG})"); ax.set_ylabel(f"Calibrated PM2.5 ({UG})")
        ax.set_title("Calibrated vs EPA reference\n(held-out test rows)"); ax.grid(alpha=0.3)
        save(fig, figs, name)

def fig_hist(pred_csv, figs):
    name = "rf_calibration_error_hist.png"; t = test_rows(pred_csv)
    err_raw = (t["ref"] - t["PA"]).abs(); err_cal = (t["ref"] - t["Predicted_ref"]).abs()
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=SIZE[name], dpi=DPI, layout="constrained")
        ax.hist(err_raw, bins=50, range=(0, 250), alpha=0.6, label="Raw PurpleAir − reference")
        ax.hist(err_cal, bins=50, range=(0, 250), alpha=0.6, label="Calibrated − reference")
        ax.set_xlabel(f"Absolute error ({UG})"); ax.set_ylabel("Count")
        ax.set_title("Calibration reduces\nabsolute error"); ax.legend(loc="upper right", framealpha=0.85)
        save(fig, figs, name)

def find_pred():
    for root in (HERE, os.getcwd(), os.path.dirname(HERE)):
        for dp, dn, fn in os.walk(root):
            if "rf_pred_all.csv" in fn: return os.path.join(dp, "rf_pred_all.csv")
            if dp.count(os.sep) - root.count(os.sep) >= 4: dn[:] = []
    raise SystemExit("Cannot find rf_pred_all.csv: pass --pred PATH")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(HERE, "..", "results"), help="folder holding RF_results.json, LR_results.json, NN_results.json, BJ_results.json")
    ap.add_argument("--figs", default=HERE); ap.add_argument("--pred", default=os.path.join(HERE, "..", "data", "rf_pred_all.csv"))
    ap.add_argument("--only", nargs="*", default=["reliability", "pareto", "scatter", "hist"]); ap.add_argument("--force", action="store_true")
    A = ap.parse_args()
    if any(x in A.only for x in ("reliability", "pareto")):
        ok = verify(A.results)
        if not ok and not A.force:
            print("\nSTOPPED before the two combined figures (values differ from the paper). Use another --results folder, or add --force to plot anyway.")
            A.only = [x for x in A.only if x not in ("reliability", "pareto")]
        else:
            if "reliability" in A.only: fig_reliability(A.results, A.figs)
            if "pareto" in A.only: fig_pareto(A.results, A.figs)
    if any(x in A.only for x in ("scatter", "hist")):
        pred = A.pred or find_pred()
        if "scatter" in A.only: fig_scatter(pred, A.figs)
        if "hist" in A.only: fig_hist(pred, A.figs)
    print("DONE")
