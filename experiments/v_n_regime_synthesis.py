"""Section V-N. Random Forest row of the four-regime synthesis (Table XXI)."""

from config import DATA_DIR, RESULTS_DIR


def regime_unified_table_row_citation():
    import json

    rf_results = {
        "model": "RF",
        "AQI_DELTA_MAP": {"Non-smoky": 0.981723249508968, "Smoky": 9.697103519851735},
        "threshold_sweep": {
            "0.5": {"coverage": 0.84, "mae": 0.938},
            "0.7": {"coverage": 0.64, "mae": 0.778},
            "0.8": {"coverage": 0.47, "mae": 0.687},
            "0.9": {"coverage": 0.24, "mae": 0.569},
        },
        "baseline_mae": 0.930,
        "binwise": [
            {"AQI_group":"Non-smoky","trust_bin":"0.0-0.5","mean_trust":0.368,"actual_reliability":0.369,"mean_mae":1.895,"samples":3501},
            {"AQI_group":"Non-smoky","trust_bin":"0.5-0.6","mean_trust":0.554,"actual_reliability":0.622,"mean_mae":1.094,"samples":2872},
            {"AQI_group":"Non-smoky","trust_bin":"0.6-0.7","mean_trust":0.654,"actual_reliability":0.765,"mean_mae":0.723,"samples":4975},
            {"AQI_group":"Non-smoky","trust_bin":"0.7-0.8","mean_trust":0.753,"actual_reliability":0.886,"mean_mae":0.466,"samples":7638},
            {"AQI_group":"Non-smoky","trust_bin":"0.8-0.9","mean_trust":0.846,"actual_reliability":0.954,"mean_mae":0.341,"samples":7868},
            {"AQI_group":"Non-smoky","trust_bin":"0.9-1.0","mean_trust":0.922,"actual_reliability":0.986,"mean_mae":0.256,"samples":1870},
            {"AQI_group":"Smoky","trust_bin":"0.0-0.5","mean_trust":0.340,"actual_reliability":0.286,"mean_mae":37.959,"samples":70},
            {"AQI_group":"Smoky","trust_bin":"0.5-0.6","mean_trust":0.556,"actual_reliability":0.548,"mean_mae":12.761,"samples":73},
            {"AQI_group":"Smoky","trust_bin":"0.6-0.7","mean_trust":0.654,"actual_reliability":0.628,"mean_mae":9.378,"samples":148},
            {"AQI_group":"Smoky","trust_bin":"0.7-0.8","mean_trust":0.760,"actual_reliability":0.859,"mean_mae":4.752,"samples":313},
            {"AQI_group":"Smoky","trust_bin":"0.8-0.9","mean_trust":0.842,"actual_reliability":0.966,"mean_mae":2.969,"samples":350},
            {"AQI_group":"Smoky","trust_bin":"0.9-1.0","mean_trust":0.906,"actual_reliability":0.875,"mean_mae":3.559,"samples":8},
        ]
    }
    cov = [1.0, 0.84, 0.64, 0.47, 0.24]
    mae = [0.930, 0.938, 0.778, 0.687, 0.569]
    order = sorted(range(len(cov)), key=lambda i: cov[i])
    rf_results["pareto"] = {
        "coverage": [cov[i] for i in order],
        "mae_mean": [mae[i] for i in order],
        "mae_std": [0.0]*len(cov)
    }
    rf_results["raw"] = {"pr":[0.902,0.038],"roc":[0.790,0.030],"brier":[None,None],"ece":[0.125,0.040]}
    rf_results["cal"] = {"pr":[0.902,0.038],"roc":[0.790,0.030],"brier":[0.157,0.025],"ece":[0.053,0.033]}
    rf_results["dummy"] = {"pr":[0.731,0.054],"roc":[0.500,0.000],"brier":[0.199,0.023]}

    with open(f"{RESULTS_DIR}/RF_reported_results.json","w") as f:
        json.dump(rf_results, f, indent=2)
    print("wrote RF_reported_results.json (from published paper tables, not re-derived)")


if __name__ == "__main__":
    regime_unified_table_row_citation()
