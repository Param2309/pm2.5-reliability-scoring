"""Section IV-D. Stage 1 Random Forest calibration (Table II, Fig. 1, Table III)."""

from config import DATA_DIR, RESULTS_DIR


def fit_stage1_random_forest_refit():
    import pandas as pd, numpy as np, time
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer

    data = pd.read_csv(f"{DATA_DIR}/WildFireData.csv")
    X = data[['RH', 'PA', 'id']]
    y = data['ref']

    column_transformer = ColumnTransformer([('encoder', OneHotEncoder(), ['id'])], remainder='passthrough')
    X_encoded = column_transformer.fit_transform(X)
    scaler = StandardScaler(with_mean=False)
    X_scaled = scaler.fit_transform(X_encoded)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    t0 = time.time()
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=None, random_state=42, n_jobs=2)
    rf_model.fit(X_train, y_train)
    print("fit took", time.time()-t0, "s", flush=True)

    y_pred_test = rf_model.predict(X_test)
    print("TEST SET  : MAE=%.4f R2=%.4f RMSE=%.4f" % (
        mean_absolute_error(y_test, y_pred_test), r2_score(y_test, y_pred_test),
        np.sqrt(mean_squared_error(y_test, y_pred_test))), flush=True)

    y_pred_train = rf_model.predict(X_train)
    print("TRAIN SET : MAE=%.4f R2=%.4f RMSE=%.4f" % (
        mean_absolute_error(y_train, y_pred_train), r2_score(y_train, y_pred_train),
        np.sqrt(mean_squared_error(y_train, y_pred_train))), flush=True)

    import joblib
    joblib.dump(rf_model, f"{RESULTS_DIR}/rf_model_refit.pkl")
    joblib.dump(column_transformer, f"{RESULTS_DIR}/rf_ct_refit.pkl")
    joblib.dump(scaler, f"{RESULTS_DIR}/rf_scaler_refit.pkl")

    y_pred_all = rf_model.predict(X_scaled)
    out = pd.DataFrame({
        "hr": data["hr"], "RH": data["RH"], "PA": data["PA"], "id": data["id"], "ref": data["ref"],
        "Predicted_ref": y_pred_all
    })
    out.to_csv(f"{RESULTS_DIR}/rf_pred_refit.csv", index=False)
    print("SAVED rf_pred_refit.csv", out.shape, flush=True)
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    fit_stage1_random_forest_refit()
