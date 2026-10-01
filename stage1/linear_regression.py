"""Section IV-C. Stage 1 Linear Regression calibration candidate (Table II)."""

from config import DATA_DIR, RESULTS_DIR


def fit_stage1_linear_regression():
    import pandas as pd, numpy as np
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    DATA = f"{DATA_DIR}/WildFireData.csv"
    data = pd.read_csv(DATA)
    print("Loaded", data.shape, data['id'].nunique(), "sensors")

    X = data[['RH','PA','id']]
    y = data['ref']

    column_transformer = ColumnTransformer([('encoder', OneHotEncoder(), ['id'])], remainder='passthrough')
    X_encoded = column_transformer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

    scaler = StandardScaler(with_mean=False)
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    lr = LinearRegression()
    lr.fit(X_train_s, y_train)
    y_pred = lr.predict(X_test_s)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"LR held-out test: MAE={mae:.4f} R2={r2:.4f} RMSE={rmse:.4f}")

    X_all_scaled = scaler.transform(X_encoded)
    pred_all_lr = lr.predict(X_all_scaled).reshape(-1).astype(float)

    out = pd.DataFrame({
        "hr": data["hr"], "RH": data["RH"], "PA": data["PA"], "id": data["id"], "ref": data["ref"],
        "Predicted_ref": pred_all_lr
    })
    out.to_csv(f"{RESULTS_DIR}/lr_pred_all.csv", index=False)
    print("Saved lr_pred_all.csv", out.shape)


if __name__ == "__main__":
    fit_stage1_linear_regression()
