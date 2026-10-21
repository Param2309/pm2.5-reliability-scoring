"""Sections V-F and V-G. Frozen Stage 2 on Missoula 2021 and NYC Bronx 2023 with 24-hour moving-block bootstrap (Tables XV-XX)."""

from config import DATA_DIR, RESULTS_DIR


def run_external_validation_block_bootstrap():
    import warnings
    warnings.filterwarnings("ignore")
    import numpy as np, pandas as pd, joblib, json
    from scipy.stats import spearmanr

    UP = DATA_DIR
    AQI_DELTA_MAP = {"Non-smoky": 0.981723249508968, "Smoky": 9.697103519851735}
    NUM_FEATS = ["RH","PA","Predicted_ref","lag1_PA","dp_PA","lag1_pred","dp_pred","corr",
                 "abs_dp_PA","abs_dp_pred","PA_roll_mean_3","PA_roll_std_3",
                 "pred_roll_mean_3","pred_roll_std_3","PA_roll_mean_6","PA_roll_std_6",
                 "pred_roll_mean_6","pred_roll_std_6"]
    CAT_FEATS = ["AQI_group"]
    FEATURE_COLS = CAT_FEATS + NUM_FEATS
    RNG_SEED = 42
    N_BOOT = 5000
    BLOCK_HOURS = 24

    trust_pipe = joblib.load(f"{UP}/trust_pipe.pkl")
    platt = joblib.load(f"{UP}/platt.pkl")

    def build_features(df):
        df = df.copy()
        df["hr"] = pd.to_datetime(df["hr"], utc=True, errors="coerce")
        df["id"] = df["id"].astype(str)
        for c in ["RH","PA","ref"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["hr","id","RH","PA","ref","Predicted_ref"]).copy()
        df = df.sort_values(["id","hr"]).reset_index(drop=True)

        df["lag1_PA"] = df.groupby("id")["PA"].shift(1)
        df["dp_PA"] = df["PA"] - df["lag1_PA"]
        df["lag1_pred"] = df.groupby("id")["Predicted_ref"].shift(1)
        df["dp_pred"] = df["Predicted_ref"] - df["lag1_pred"]
        df["corr"] = df["Predicted_ref"] - df["PA"]
        df["abs_dp_PA"] = df["dp_PA"].abs()
        df["abs_dp_pred"] = df["dp_pred"].abs()
        for w in [3,6]:
            df[f"PA_roll_mean_{w}"] = df.groupby("id")["PA"].transform(lambda s: s.rolling(w,min_periods=1).mean())
            df[f"PA_roll_std_{w}"] = df.groupby("id")["PA"].transform(lambda s: s.rolling(w,min_periods=2).std())
            df[f"pred_roll_mean_{w}"] = df.groupby("id")["Predicted_ref"].transform(lambda s: s.rolling(w,min_periods=1).mean())
            df[f"pred_roll_std_{w}"] = df.groupby("id")["Predicted_ref"].transform(lambda s: s.rolling(w,min_periods=2).std())
        rstd = [c for c in df.columns if "roll_std" in c]
        df[rstd] = df[rstd].fillna(0.0)
        df["AQI_group"] = pd.cut(df["Predicted_ref"], bins=[-np.inf,35.4,np.inf], labels=["Non-smoky","Smoky"]).astype(str)
        df = df.dropna(subset=["lag1_PA","dp_PA","lag1_pred","dp_pred"]).copy().reset_index(drop=True)

        df["trust_prob_raw"] = trust_pipe.predict_proba(df[FEATURE_COLS])[:,1]
        eps=1e-6
        logit_raw = np.log(np.clip(df["trust_prob_raw"],eps,1-eps)/np.clip(1-df["trust_prob_raw"],eps,1-eps)).values.reshape(-1,1)
        df["trust_prob_cal"] = platt.predict_proba(logit_raw)[:,1]

        df["abs_error_raw_PA"] = (df["ref"] - df["PA"]).abs()
        df["abs_error_rf"] = (df["ref"] - df["Predicted_ref"]).abs()
        df["delta_original"] = df["AQI_group"].map(AQI_DELTA_MAP).astype(float)
        df["reliable_original"] = (df["abs_error_rf"] <= df["delta_original"]).astype(int)
        return df

    def block_bootstrap_spearman_chrono(df_full, target_group, err_col, block_size=BLOCK_HOURS, n_boot=N_BOOT, seed=RNG_SEED):
        n = len(df_full)
        rng = np.random.default_rng(seed)
        x_full = df_full["trust_prob_cal"].values
        y_full = df_full[err_col].values
        grp_full = df_full["AQI_group"].values
        n_blocks_needed = int(np.ceil(n / block_size))
        starts_pool = np.arange(0, max(1, n - block_size + 1))
        boot_r = []
        for b in range(n_boot):
            starts = rng.choice(starts_pool, size=n_blocks_needed, replace=True)
            idx = np.concatenate([np.arange(s, min(s+block_size, n)) for s in starts])[:n]
            mask = grp_full[idx] == target_group
            if mask.sum() < 5:
                boot_r.append(np.nan)
                continue
            r, _ = spearmanr(x_full[idx][mask], y_full[idx][mask])
            boot_r.append(r)
        boot_r = np.array(boot_r)
        valid = boot_r[~np.isnan(boot_r)]
        lo, hi = np.percentile(valid, [2.5, 97.5])
        return float(lo), float(hi), valid, float(np.isnan(boot_r).mean())

    def topk_mae(df, score_col, err_col, ascending=False):
        order = np.argsort(-df[score_col].values) if not ascending else np.argsort(df[score_col].values)
        ae_sorted = df[err_col].values[order]
        unfilt = float(np.mean(df[err_col].values))
        out = {}
        for q in [0.5, 0.25, 0.10]:
            k = max(1, int(round(q*len(df))))
            out[str(q)] = {"mae": float(np.mean(ae_sorted[:k])),
                            "reduction_pct": float(100*(1-np.mean(ae_sorted[:k])/unfilt))}
        return out, unfilt

    results = {}
    for name, path in [("Missoula", f"{UP}/rf_pred_Missoula.csv"),
                        ("NYC_Bronx", f"{UP}/rf_pred_NYC_Bronx.csv")]:
        raw = pd.read_csv(path)
        df = build_features(raw)
        r_all, p_all = spearmanr(df["trust_prob_cal"], df["abs_error_rf"])
        print(f"\n{name}: n={len(df)} sensors={df['id'].nunique()} R2_predref="
              f"{1 - ((df['ref']-df['Predicted_ref'])**2).sum()/((df['ref']-df['ref'].mean())**2).sum():.4f} "
              f"overall Spearman r={r_all:.4f} p={p_all:.3e}", flush=True)
        site_result = {"n": int(len(df)), "overall_r": float(r_all), "overall_p": float(p_all), "groups": {}}
        for grp in ["Non-smoky","Smoky"]:
            sub = df[df["AQI_group"]==grp]
            if len(sub) < 5:
                print(f"  {grp}: n={len(sub)} too small, skipping bootstrap", flush=True)
                continue
            r_g, p_g = spearmanr(sub["trust_prob_cal"], sub["abs_error_rf"])
            lo, hi, boot_r, frac_dropped = block_bootstrap_spearman_chrono(df, grp, "abs_error_rf")
            frac_wrong_sign = float(np.mean(np.sign(boot_r) != np.sign(r_g))) if r_g != 0 and len(boot_r) else float("nan")
            print(f"  {grp}: n={len(sub)} r={r_g:.4f} p={p_g:.3e} CHRONO block-bootstrap 95% CI=[{lo:.4f},{hi:.4f}] "
                  f"frac_boot_wrong_sign={frac_wrong_sign:.4f}", flush=True)

            r_raw, p_raw = spearmanr(sub["trust_prob_cal"], sub["abs_error_raw_PA"])
            topk_raw, unfilt_raw = topk_mae(sub, "trust_prob_cal", "abs_error_raw_PA")
            topk_cal, unfilt_cal = topk_mae(sub, "trust_prob_cal", "abs_error_rf")
            print(f"    raw-PA-error: r={r_raw:.4f} p={p_raw:.3e} | top10% raw-PA MAE={topk_raw['0.1']['mae']:.3f} "
                  f"({topk_raw['0.1']['reduction_pct']:.1f}% redn) vs calibrated top10% MAE="
                  f"{topk_cal['0.1']['mae']:.3f} ({topk_cal['0.1']['reduction_pct']:.1f}% redn)", flush=True)

            site_result["groups"][grp] = {
                "n": int(len(sub)), "r": float(r_g), "p": float(p_g),
                "boot_ci95_chrono": [lo, hi], "frac_boot_wrong_sign": frac_wrong_sign,
                "frac_bootstrap_resamples_dropped": frac_dropped,
                "raw_PA_error": {"spearman_r": float(r_raw), "spearman_p": float(p_raw),
                                  "unfiltered_mae": unfilt_raw, "topk": topk_raw},
                "calibrated_error": {"unfiltered_mae": unfilt_cal, "topk": topk_cal},
            }
        results[name] = site_result

    json.dump(results, open(f"{RESULTS_DIR}/external_block_bootstrap_results.json","w"), indent=2)
    print("\nALL_DONE", flush=True)


if __name__ == "__main__":
    run_external_validation_block_bootstrap()
