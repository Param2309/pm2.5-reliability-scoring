"""Section IV-C. Stage 1 Barkjohn candidate: published EPA correction applied to raw PA and RH."""
import numpy as np
import pandas as pd

from config import DATA_DIR, RESULTS_DIR


def barkjohn_eq1(pa_cf1, rh):
    return 0.524 * pa_cf1 - 0.0862 * rh + 5.75


def barkjohn_eq3(pa_cf1):
    return 4.21e-4 * (pa_cf1 ** 2) + 0.392 * pa_cf1 + 3.44


def barkjohn_eq2(pa_cf1, rh):
    w = np.clip(0.0244 * pa_cf1 - 13.9, 0.0, 1.0)
    return w * barkjohn_eq3(pa_cf1) + (1.0 - w) * barkjohn_eq1(pa_cf1, rh)


def barkjohn_piecewise(pa_cf1, rh):
    pa_cf1 = np.asarray(pa_cf1, dtype=float)
    rh = np.asarray(rh, dtype=float)
    out = np.full(pa_cf1.shape, np.nan, dtype=float)

    m1 = pa_cf1 < 570
    m2 = (pa_cf1 >= 570) & (pa_cf1 < 611)
    m3 = pa_cf1 >= 611

    out[m1] = barkjohn_eq1(pa_cf1[m1], rh[m1])
    out[m2] = barkjohn_eq2(pa_cf1[m2], rh[m2])
    out[m3] = barkjohn_eq3(pa_cf1[m3])
    return out


def fit_stage1_barkjohn():
    data = pd.read_csv(f"{DATA_DIR}/WildFireData.csv")
    data = data.sort_values(["id", "hr"], kind="stable").reset_index(drop=True)
    out = pd.DataFrame({
        "hr": data["hr"], "id": data["id"], "RH": data["RH"], "PA": data["PA"], "ref": data["ref"],
        "Predicted_ref": barkjohn_piecewise(data["PA"].values, data["RH"].values),
    })
    mae = np.mean(np.abs(out["ref"] - out["Predicted_ref"]))
    print(f"Barkjohn MAE over all rows = {mae:.4f}")
    out.to_csv(f"{RESULTS_DIR}/bj_pred_all.csv", index=False)
    print("Saved bj_pred_all.csv", out.shape)


if __name__ == "__main__":
    fit_stage1_barkjohn()
