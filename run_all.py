"""Runs every stage of the paper in order. Pass step names to run a subset, e.g. `python run_all.py stage1_rf stage2`."""
import subprocess
import sys

STEPS = [
    ("stage1_rf", "stage1.random_forest"),
    ("stage1_lr", "stage1.linear_regression"),
    ("stage1_nn", "stage1.neural_network"),
    ("stage1_bj", "stage1.barkjohn"),
    ("stage2", "stage2.candidates"),
    ("iv_g", "experiments.iv_g_classifier_comparison"),
    ("v_b", "experiments.v_b_safety_diagnostics"),
    ("v_c", "experiments.v_c_feature_ablation"),
    ("v_d", "experiments.v_d_heuristic_baseline"),
    ("v_e", "experiments.v_e_delta_robustness"),
    ("v_f", "experiments.v_f_external_validation"),
    ("v_f_sites", "experiments.v_f_single_site_checks"),
    ("v_j_audit", "experiments.v_j_sensor_identity_audit"),
    ("v_j_ablation", "experiments.v_j_sensor_identity_ablation"),
    ("v_j_safety", "experiments.v_j_sensor_identity_safety"),
    ("v_k", "experiments.v_k_temporal_audit"),
    ("v_l_reconstruction", "experiments.v_l_coldstart_reconstruction"),
    ("v_l_coverage", "experiments.v_l_coldstart_coverage"),
    ("v_m", "experiments.v_m_quantile_retention"),
    ("v_n", "experiments.v_n_regime_synthesis"),
    ("table_01", "tables.table_01_aqi_distribution"),
    ("table_03", "tables.table_03_binwise_rmse"),
    ("tables_stage2", "tables.stage2_tables"),
    ("fig1", "figures.fig1_rf_calibration"),
    ("fig2", "figures.fig2_stage1_density"),
    ("fig3", "figures.fig3_reliability_diagrams"),
    ("fig4", "figures.fig4_coverage_mae_pareto"),
]

if __name__ == "__main__":
    selected = set(sys.argv[1:])
    for name, module in STEPS:
        if selected and name not in selected:
            continue
        print(f"=== {name}: {module}", flush=True)
        subprocess.run([sys.executable, "-m", module], check=True)
