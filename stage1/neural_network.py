"""Section IV-C. Stage 1 neural network calibration candidate (Table II)."""

from config import DATA_DIR, RESULTS_DIR


def fit_stage1_neural_network():
    import pandas as pd, numpy as np, time
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense

    np.random.seed(42)
    tf.random.set_seed(42)

    data = pd.read_csv(f'{DATA_DIR}/WildFireData.csv')
    X = data[['RH','PA','id']]
    y = data['ref']
    ct = ColumnTransformer([('encoder', OneHotEncoder(), ['id'])], remainder='passthrough')
    Xe = ct.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(Xe, y, test_size=0.2, random_state=42)
    sc = StandardScaler(with_mean=False)
    X_train_s = sc.fit_transform(X_train)
    X_test_s = sc.transform(X_test)
    X_train_d = np.asarray(X_train_s.todense()) if hasattr(X_train_s,'todense') else X_train_s
    X_test_d = np.asarray(X_test_s.todense()) if hasattr(X_test_s,'todense') else X_test_s

    model = Sequential([
        Dense(64, input_dim=X_train_d.shape[1], activation='relu'),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')

    t0=time.time()
    model.fit(X_train_d, y_train, epochs=100, batch_size=32, verbose=2, validation_split=0.2)
    print("100 epochs took", time.time()-t0, "seconds", flush=True)

    y_pred = model.predict(X_test_d).flatten()
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"NN held-out test: MAE={mae:.4f} R2={r2:.4f} RMSE={rmse:.4f}", flush=True)

    X_all_s = sc.transform(Xe)
    X_all_d = np.asarray(X_all_s.todense()) if hasattr(X_all_s,'todense') else X_all_s
    pred_all_nn = model.predict(X_all_d).flatten().astype(float)

    out = pd.DataFrame({
        "hr": data["hr"], "RH": data["RH"], "PA": data["PA"], "id": data["id"], "ref": data["ref"],
        "Predicted_ref": pred_all_nn
    })
    out.to_csv(f"{RESULTS_DIR}/nn_pred_all.csv", index=False)
    print("Saved nn_pred_all.csv", out.shape, flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    fit_stage1_neural_network()
