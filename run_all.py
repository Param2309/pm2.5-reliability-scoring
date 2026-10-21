"""Runs every stage of the paper in order. Pass step names to run a subset, e.g. `python run_all.py stage1_rf stage2`."""
import subprocess
import sys

STEPS = [
    # Stage 1: calibration candidates (Section IV)
    ("stage1_rf", "stage1.random_forest"),
    ("stage1_lr", "stage1.linear_regression"),
    ("stage1_nn", "stage1.neural_network"),
    ("stage1_bj", "stage1.barkjohn"),
    # Stage 2: reliability classifier, R1 original protocol (Section IV, V-A)
    ("stage2", "stage2.candidates"),
    ("stage2_classifiers", "stage2.classifier_comparison"),
    ("table_01", "tables.table_01_aqi_distribution"),
    ("table_02", "tables.table_02_stage1_binwise_rmse"),
    ("tables_stage2", "tables.stage2_tables"),
    # Internal regimes R1, R2, R3 and the leakage audit (Section V)
    ("r1_safety", "internal.r1_safety_diagnostics"),
    ("r1_ablation", "internal.r1_feature_ablation"),
    ("r1_delta", "internal.r1_delta_robustness"),
    ("r2_temporal", "internal.r2_temporal_audit"),
    ("r3_coldstart", "internal.r3_coldstart"),
    ("r3_coverage", "internal.r3_coldstart_coverage"),
    ("identity_audit", "internal.sensor_identity_audit"),
    ("identity_ablation", "internal.sensor_identity_ablation"),
    ("identity_safety", "internal.sensor_identity_safety"),
    # External validation on the data in data/ (Section V)
    ("ext_missoula_nyc", "external.missoula_nyc_bronx"),
    ("ext_la", "external.los_angeles_single_site"),
]

if __name__ == "__main__":
    selected = set(sys.argv[1:])
    for name, module in STEPS:
        if selected and name not in selected:
            continue
        print(f"=== {name}: {module}", flush=True)
        subprocess.run([sys.executable, "-m", module], check=True)
