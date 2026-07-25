import json
import numpy as np
import pandas as pd

np.random.seed(42)

# ---------------------------------------------------------------------
# 1. Load and clean real data
# ---------------------------------------------------------------------
df = pd.read_csv("AB_NYC_2019.csv")
print(f"Raw rows: {len(df)}")

df = df[(df["price"] > 0) & (df["price"] <= 1000)].copy()
df["minimum_nights"] = df["minimum_nights"].clip(upper=30)
df["reviews_per_month"] = df["reviews_per_month"].fillna(0)
df = df.dropna(subset=["neighbourhood_group", "room_type"])
print(f"After cleaning: {len(df)}")

BOROUGHS = sorted(df["neighbourhood_group"].unique().tolist())
ROOM_TYPES = sorted(df["room_type"].unique().tolist())
print("Boroughs:", BOROUGHS)
print("Room types:", ROOM_TYPES)

# ---------------------------------------------------------------------
# 2. Feature engineering
# ---------------------------------------------------------------------
def borough_onehot(b):
    return [1.0 if b == x else 0.0 for x in BOROUGHS]

def room_onehot(r):
    return [1.0 if r == x else 0.0 for x in ROOM_TYPES]

numeric_cols = ["minimum_nights", "number_of_reviews", "availability_365", "reviews_per_month"]
X_numeric_raw = df[numeric_cols].values.astype(np.float64)
X_numeric_raw[:, 0] = np.log1p(X_numeric_raw[:, 0])  # minimum_nights
X_numeric_raw[:, 1] = np.log1p(X_numeric_raw[:, 1])  # number_of_reviews

X_cat = np.array([borough_onehot(b) + room_onehot(r) for b, r in zip(df["neighbourhood_group"], df["room_type"])])

# standardize numeric features using TRAIN stats (computed after split below)
y_log = np.log1p(df["price"].values.astype(np.float64))

X_all = np.concatenate([X_numeric_raw, X_cat], axis=1)
N_FEATURES = X_all.shape[1]
print(f"Feature dimension: {N_FEATURES}")

# ---------------------------------------------------------------------
# 3. Train/test split
# ---------------------------------------------------------------------
idx = np.random.permutation(len(X_all))
split = int(len(idx) * 0.9)
train_idx, test_idx = idx[:split], idx[split:]

X_train_raw, y_train = X_all[train_idx], y_log[train_idx]
X_test_raw, y_test = X_all[test_idx], y_log[test_idx]

num_mean = X_train_raw[:, :len(numeric_cols)].mean(axis=0)
num_std = X_train_raw[:, :len(numeric_cols)].std(axis=0) + 1e-8

def standardize(X):
    X = X.copy()
    X[:, :len(numeric_cols)] = (X[:, :len(numeric_cols)] - num_mean) / num_std
    return X

X_train = standardize(X_train_raw)
X_test = standardize(X_test_raw)
print(f"Train: {len(X_train)}, Test: {len(X_test)}")

# ---------------------------------------------------------------------
# 4. Deep feedforward network: N -> 32 (ReLU) -> 16 (ReLU) -> 1 (linear)
#    Trained with a hand-rolled ADAM optimizer (Unit-1 syllabus content)
# ---------------------------------------------------------------------
H1, H2 = 32, 16
rng = np.random.RandomState(3)

def he_init(fan_in, fan_out):
    return rng.randn(fan_in, fan_out) * np.sqrt(2.0 / fan_in)

params = {
    "W1": he_init(N_FEATURES, H1), "b1": np.zeros(H1),
    "W2": he_init(H1, H2), "b2": np.zeros(H2),
    "W3": he_init(H2, 1), "b3": np.zeros(1),
}

# Adam optimizer state
m = {k: np.zeros_like(v) for k, v in params.items()}
v = {k: np.zeros_like(v) for k, v in params.items()}
beta1, beta2, adam_eps = 0.9, 0.999, 1e-8
LR = 0.003


def forward(X):
    z1 = X @ params["W1"] + params["b1"]
    h1 = np.maximum(0, z1)
    z2 = h1 @ params["W2"] + params["b2"]
    h2 = np.maximum(0, z2)
    out = h2 @ params["W3"] + params["b3"]
    return z1, h1, z2, h2, out


def backward(X, z1, h1, z2, h2, out, y_batch):
    n = X.shape[0]
    d_out = (out[:, 0] - y_batch) / n  # dMSE/d(out), averaged over batch

    grads = {}
    grads["W3"] = h2.T @ d_out[:, None]
    grads["b3"] = d_out.sum(axis=0, keepdims=True).flatten()
    dh2 = d_out[:, None] @ params["W3"].T
    dz2 = dh2 * (z2 > 0)

    grads["W2"] = h1.T @ dz2
    grads["b2"] = dz2.sum(axis=0)
    dh1 = dz2 @ params["W2"].T
    dz1 = dh1 * (z1 > 0)

    grads["W1"] = X.T @ dz1
    grads["b1"] = dz1.sum(axis=0)
    return grads


def adam_step(grads, t):
    for k in params:
        m[k] = beta1 * m[k] + (1 - beta1) * grads[k]
        v[k] = beta2 * v[k] + (1 - beta2) * (grads[k] ** 2)
        m_hat = m[k] / (1 - beta1 ** t)
        v_hat = v[k] / (1 - beta2 ** t)
        params[k] -= LR * m_hat / (np.sqrt(v_hat) + adam_eps)


# ---------------------------------------------------------------------
# 5. Training loop (mini-batch, Adam)
# ---------------------------------------------------------------------
EPOCHS = 40
BATCH = 128
n_train = len(X_train)
t_step = 0

for epoch in range(EPOCHS):
    order = np.random.permutation(n_train)
    epoch_loss = 0.0
    for start in range(0, n_train, BATCH):
        batch_idx = order[start:start + BATCH]
        Xb, yb = X_train[batch_idx], y_train[batch_idx]

        z1, h1, z2, h2, out = forward(Xb)
        loss = np.mean((out[:, 0] - yb) ** 2)
        epoch_loss += loss * len(batch_idx)

        t_step += 1
        grads = backward(Xb, z1, h1, z2, h2, out, yb)
        adam_step(grads, t_step)

    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch {epoch+1}/{EPOCHS} - train MSE (log-price): {epoch_loss/n_train:.5f}")

# ---------------------------------------------------------------------
# 6. Evaluate on held-out test set, in REAL dollar terms
# ---------------------------------------------------------------------
_, _, _, _, test_out = forward(X_test)
pred_price = np.expm1(test_out[:, 0])
true_price = np.expm1(y_test)

mae = np.mean(np.abs(pred_price - true_price))
mape = np.mean(np.abs(pred_price - true_price) / true_price) * 100
print(f"\nTest MAE: ${mae:.2f}")
print(f"Test MAPE: {mape:.1f}%")

# baseline comparison: always predict the train-set mean price
baseline_pred = np.full_like(true_price, np.expm1(y_train.mean()))
baseline_mae = np.mean(np.abs(baseline_pred - true_price))
print(f"Baseline (predict mean price) MAE: ${baseline_mae:.2f}  <- model should beat this")

# ---------------------------------------------------------------------
# 7. Export
# ---------------------------------------------------------------------
export = {
    "boroughs": BOROUGHS,
    "room_types": ROOM_TYPES,
    "numeric_cols": numeric_cols,
    "num_mean": num_mean.tolist(),
    "num_std": num_std.tolist(),
    "weights": {k: val.tolist() for k, val in params.items()},
    "test_mae_usd": float(mae),
    "test_mape_pct": float(mape),
}
with open("price_model_weights.json", "w") as f:
    json.dump(export, f, indent=2)

print("\nSaved price_model_weights.json")