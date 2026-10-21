# Data

The hourly PurpleAir and EPA reference data are not redistributed in this repository. Place the following files in this folder (or point `PA_DATA_DIR` to a folder that holds them) before running the scripts.

| File | Used by | Description |
|---|---|---|
| `WildFireData.csv` | Stage 1, Stage 2, all audits | Hourly PurpleAir (`PA`) and EPA reference (`ref`) PM2.5 with relative humidity (`RH`), sensor `id` and hour `hr`; 27 sensors, 123,695 rows (Table I). This is the EPA-paired PurpleAir dataset of Barkjohn et al. (2021) |
| `bj_pred_all.csv` | Barkjohn candidate | Output of `stage1/barkjohn.py` (columns `hr`, `id`, `RH`, `PA`, `ref`, `Predicted_ref`) |
| `rf_pred_all.csv` | All Random Forest experiments, Tables I and III, Fig. 1 | Frozen Random Forest `Predicted_ref` for every row of `WildFireData.csv` (columns `hr`, `RH`, `PA`, `id`, `ref`, `Predicted_ref`) |
| `rf_pred_Missoula.csv`, `rf_pred_NYC_Bronx.csv`, `rf_pred_LA.csv` | Sections V-F, V-I | Frozen Random Forest `Predicted_ref` for each external site (columns `hr`, `id`, `RH`, `PA`, `ref`, `Predicted_ref`) |
| `Missoula_Wildfire_2021.csv`, `NYC_Bronx_June2023.csv`, `LA_Wildfire_Jan2025.csv` | External sites | Hourly PurpleAir, relative humidity and co-located EPA reference PM2.5 for each site |

The files shipped in this folder are the frozen Stage 1 and Stage 2 artefacts:

| File | Used by | Description |
|---|---|---|
| `column_transformer.pkl` | Section V-F | Stage 1 one-hot encoder fitted on the training sensors |
| `platt.pkl` | Section V-F | Frozen Platt calibrator for the Stage 2 score |
| `trust_pipe.pkl` | Section V-F | Frozen Stage 2 XGBoost pipeline |

The three `.pkl` files were serialised with scikit-learn 1.5.1 and XGBoost 2.1.x and must be loaded with those versions (see `requirements.txt`).
