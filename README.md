# Two-Stage Reliability Scoring for PurpleAir PM2.5 Sensors

Reference implementation of the paper *A Two-Stage Reliability Scoring Framework for PurpleAir PM2.5 Sensors under Wildfire Smoke*. Every table and figure that is computed from data in the paper has a corresponding script in this repository.

## Method in brief

| Stage | Purpose | Implementation |
|---|---|---|
| Stage 1 | Calibrate the raw PurpleAir reading (`PA`) with relative humidity (`RH`) and sensor `id` to an EPA-equivalent estimate `Predicted_ref` | Random Forest (selected), Linear Regression, feed-forward neural network, Barkjohn published formula |
| Stage 2 | Score how reliable each calibrated reading is | XGBoost classifier on 19 features (no sensor id) followed by Platt scaling |

A reading is labelled *reliable* when `|ref - Predicted_ref| <= δ`, where δ is the mean Stage 1 absolute error in the reading's AQI group (Non-smoky: `Predicted_ref <= 35.4` µg/m³, Smoky: above). Evaluation is sensor-disjoint: ten repeated `GroupShuffleSplit` splits over sensors give 15 training, 6 calibration and 6 test sensors, with PR-AUC as the primary metric, ECE over 15 bins, and confidence intervals from paired per-repeat differences.

## Repository layout

```
.
├── config.py                  data and results locations
├── run_all.py                 runs every step in paper order
├── requirements.txt
├── data/                      input data and frozen artefacts (see data/README.md)
├── stage1/                    Section IV-C, IV-D
│   ├── random_forest.py
│   ├── linear_regression.py
│   ├── neural_network.py
│   └── barkjohn.py
├── stage2/                    Sections IV-E to IV-I, V-A
│   ├── pipeline.py            features, labels, XGBoost + Platt, sensor-disjoint evaluation
│   └── candidates.py          runs the pipeline for RF, LR, NN and Barkjohn
├── experiments/               Sections IV-G, V-B to V-N
├── tables/                    Tables I, III, IX-XIII
├── figures/                   Figures 1-4
└── results/                   all outputs are written here
```

## Paper-to-code map

| Paper item | Script | Output in `results/` |
|---|---|---|
| Table I, AQI distribution | `tables/table_01_aqi_distribution.py` | `table_01_aqi_distribution.csv` |
| Table II, Stage 1 comparison | `stage1/*.py` (Random Forest, Linear Regression, neural network, Barkjohn) | printed metrics, `*_pred_all.csv` |
| Table III, Stage 1 bin-wise RMSE | `tables/table_03_binwise_rmse.py` | `table_03_binwise_rmse.csv` |
| Fig. 1, Random Forest calibration | `figures/fig1_rf_calibration.py` | `fig1_rf_calibration.png` |
| Fig. 2, LR and NN density plots | `figures/fig2_stage1_density.py` | `fig2_stage1_density.png` |
| Tables IV, VI, features and XGBoost parameters | `stage2/pipeline.py`, `stage2/candidates.py` | |
| Table V, Stage 2 classifier comparison | `experiments/iv_g_classifier_comparison.py` | `stage2_classifier_compare_results.json` |
| Tables IX-XIII, internal Stage 2 evaluation | `stage2/candidates.py`, `tables/stage2_tables.py` | `{RF,LR,NN,BJ}_results.json`, `table_09` to `table_13` |
| Fig. 3, reliability diagrams | `figures/fig3_reliability_diagrams.py` | `fig3_reliability_diagrams.png` |
| Fig. 4, coverage-MAE Pareto curves | `figures/fig4_coverage_mae_pareto.py` | `fig4_coverage_mae_pareto.png` |
| Section V-B, safety-oriented evaluation, raw-PA Spearman check | `experiments/v_b_safety_diagnostics.py` | `rf_full_diagnostics_results.json` |
| Section V-C, feature ablation | `experiments/v_c_feature_ablation.py` | `rf_ablation_paired_clean_results.json` |
| Section V-D, heuristic baseline | `experiments/v_d_heuristic_baseline.py` | `heuristic_vs_ablated_paired_results.json` |
| Section V-E and Table XIV, robustness to δ | `experiments/v_e_delta_robustness.py` | `rf_ablation_paired_clean_origdelta_results.json` |
| Sections V-F, V-G and Tables XV-XX, external validation | `experiments/v_f_external_validation.py` | `external_block_bootstrap_results.json` |
| Sections V-F and V-I, Hoopa and LA single-sensor checks | `experiments/v_f_single_site_checks.py` | `Hoopa_results.json`, `LA_Wildfire_Jan2025_results.json` |
| Section V-J, sensor-identity audit | `experiments/v_j_sensor_identity_audit.py`, `v_j_sensor_identity_ablation.py`, `v_j_sensor_identity_safety.py` | `rf_doubly_disjoint_*_results.json` |
| Section V-K, temporal audit | `experiments/v_k_temporal_audit.py` | `rf_historical_future_holdout_results.json` |
| Section V-L, cold-start reconstruction | `experiments/v_l_coldstart_reconstruction.py`, `v_l_coldstart_coverage.py` | `rf_no_id_clean_delta_results.json`, `rf_no_id_clean_delta_coverage_results.json` |
| Section V-M, quantile retention | `experiments/v_m_quantile_retention.py` | `rf_no_id_clean_delta_quantile_results.json` |
| Section V-N and Table XXI, four evaluation regimes | `experiments/v_n_regime_synthesis.py` | `RF_reported_results.json` |

## Data availability

The hourly PurpleAir and EPA data are not redistributed in this repository. The EPA-paired PurpleAir dataset of Barkjohn et al. is available from the U.S. EPA (doi:10.23719/1522388). Additional PurpleAir readings come from the PurpleAir API (data from PurpleAir, purpleair.com) and are subject to PurpleAir's terms of use; the processed files used here are available from the authors on request. `data/README.md` lists every file that must be placed in `data/` before running the scripts. The frozen Stage 2 artefacts (`trust_pipe.pkl`, `platt.pkl`, `column_transformer.pkl`) are included.

## Environment

The frozen Stage 2 artefacts in `data/` were serialised with scikit-learn 1.5.1 and XGBoost 2.1.x, and loading them under newer scikit-learn releases fails. The pinned versions in `requirements.txt` were used for every result in this repository.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

TensorFlow is only needed for `stage1/neural_network.py`. All scripts are run from the repository root as modules.

## Running

```bash
python run_all.py                       # everything, in paper order
python run_all.py stage1_rf stage2      # selected steps (names are listed in run_all.py)
python -m experiments.v_c_feature_ablation
```

Outputs go to `results/` (override with `PA_RESULTS_DIR`); inputs are read from `data/` (override with `PA_DATA_DIR`). `stage1_lr` and `stage1_nn` must run before `stage2`. The Random Forest experiments read `data/rf_pred_all.csv` (not redistributed, see `data/README.md`) and do not depend on `stage1_rf`.

Approximate run times on a two-core machine:

| Step | Time |
|---|---|
| `stage1/random_forest.py` | 4-5 min |
| `stage1/linear_regression.py` | under 1 min |
| `stage1/neural_network.py` (100 epochs) | about 7 min |
| `stage2/candidates.py` | 2 min per candidate |
| `experiments/iv_g_classifier_comparison.py` | 15 min |
| `experiments/v_b` to `v_e`, `v_f`, `v_n` | 1-4 min each |
| `experiments/v_j_*` (ten Stage 1 refits each) | 35-40 min each |
| `experiments/v_k`, `v_l_*`, `v_m` | 4-8 min each |

## Reproducibility notes

**Random Forest predictions are the frozen ones.** The original Random Forest model file (`rf_model.pkl`, 833 MB) is not distributed. Its predictions were exported once with the original `scaler.pkl` and `column_transformer.pkl` and are required as `data/rf_pred_all.csv` (training sensors) and `data/rf_pred_{Missoula,NYC_Bronx,LA,Hoopa}.csv` (external sites); these files are not redistributed, see `data/README.md`. Every Random Forest experiment, Table I, Table III, Fig. 1 and the external validation read these files. `stage1/random_forest.py` refits the model with the same architecture and seed, reproduces the reported test metrics (MAE 2.356, R² 0.9859, RMSE 5.62) and writes `results/rf_pred_refit.csv`; this refit is used only to report Table II and is not used downstream, because it differs slightly from the frozen model on unseen sensors.

**δ(Smoky).** Non-smoky δ is the mean absolute Stage 1 error over training and calibration readings with `Predicted_ref` ≤ 35.4 (0.982 µg/m³). For the Random Forest, the canonical Smoky δ is the mean over the Unhealthy+ readings (`Predicted_ref` > 55.4) of the same split, 9.697 µg/m³ (`smoky_delta_unhealthy_only=True` in `stage2/pipeline.py`, used by `stage2/candidates.py` and `experiments/v_b_safety_diagnostics.py`; the other experiments carry it as a constant). The mean over the full Smoky group (`Predicted_ref` > 35.4) is 7.684 µg/m³ and is what the pipeline derives by default for the Linear Regression, neural network and Barkjohn candidates and in `v_c` and `v_d`; Section V-E compares the two.

**Table XI and XII.** The bin-wise tables are taken from the last (tenth) sensor-disjoint repeat, which is the repeat the reported counts (28,724 Non-smoky and 962 Smoky test rows) correspond to. `RF_results.json` also stores the bin-wise table pooled over all ten repeats (`binwise`).

**Random Forest row of Tables IX-XIII.** `stage2/candidates.py` runs the shared pipeline on the frozen predictions (`RF_results.json`), and Tables IX-XIII and Figs. 3 and 4 are built from that run. `experiments/v_n_regime_synthesis.py` separately stores the Random Forest row exactly as printed in the paper (`RF_reported_results.json`).

**Barkjohn predictions.** `stage1/barkjohn.py` applies the piecewise Barkjohn correction to raw PA and RH and reproduces `data/bj_pred_all.csv` to floating-point precision (MAE 3.0956 over all rows).

**Hoopa and Los Angeles.** `experiments/v_f_single_site_checks.py` applies the frozen Stage 2 to the frozen Random Forest predictions of each site. Hoopa gives a reliable rate of 48.55% and ROC-AUC 0.7654, and Los Angeles 10.96% and 0.494, matching the paper.

**Figure 1.** The figure script follows the caption of Fig. 1 (calibrated PM2.5 against EPA reference, and the absolute-error histograms for raw PA and calibrated PM2.5, with the display ranges stated there).

**Neural network and Table I.** Keras training is not bit-reproducible across TensorFlow builds, so re-running `stage1/neural_network.py` gives Stage 1 and δ values close to, but not identical with, the reported ones. Table I is computed from the frozen Random Forest predictions.

**Seeds.** Every random component is seeded (`random_state=42`, with `+rep` and `+100+rep` offsets for the two sensor-level splits of repetition `rep`). Third-decimal differences from the tables are possible across library builds.

## Optuna

Optuna is not run in this repository. The hyperparameters in the paper (Tables V and VI, and the per-candidate XGBoost parameters in `stage2/candidates.py`) are fixed values from the original tuning studies and are reused unchanged. Those studies were run in the original notebooks with a maximising objective over PR-AUC: the mean PR-AUC of the Stage 2 classifier across repeated sensor-level `GroupShuffleSplit` splits, using Optuna's TPE sampler and 15 to 20 trials per study. For the XGBoost classifier the search covered `n_estimators` 200-1200, `learning_rate` 0.01-0.3 (log), `max_depth` 3-12, `min_child_weight` 1-50 (log), `subsample` and `colsample_bytree` 0.6-1.0, `reg_lambda` 1e-3-100 (log), `reg_alpha` 1e-6-10 (log), `gamma` 0-10 and `scale_pos_weight` 0.5-30 (log). The Logistic Regression and Random Forest classifiers of Table V were searched over `C`, `class_weight`, `solver`, `penalty`, and over `n_estimators`, `max_depth`, `min_samples_leaf`, `max_features`, `class_weight`, respectively.

## License

The code is released under the MIT License (see `LICENSE`).
