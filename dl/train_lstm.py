import json
import numpy as np

np.random.seed(42)

# ---------------------------------------------------------------------
# 1. Destination types and their seasonal curve parameters
# ---------------------------------------------------------------------
DEST_TYPES = {
    "beach":    {"amplitude": 0.25, "peak_day": 355, "keywords": ["beach", "coast", "island", "goa", "seaside"]},
    "hill":     {"amplitude": 0.20, "peak_day": 155, "keywords": ["hill", "mountain", "shimla", "manali", "hilltop"]},
    "city":     {"amplitude": 0.10, "peak_day": 0,   "keywords": ["city", "urban", "metro"]},
    "heritage": {"amplitude": 0.15, "peak_day": 300, "keywords": ["heritage", "historic", "temple", "fort", "palace"]},
    "desert":   {"amplitude": 0.30, "peak_day": 335, "keywords": ["desert", "rajasthan", "jaisalmer", "dune"]},
}
TYPE_NAMES = list(DEST_TYPES.keys())


def seasonal_value(day_of_year, amplitude, peak_day, noise_std=0.03, rng=None):
    base = 1.0 + amplitude * np.cos(2 * np.pi * (day_of_year - peak_day) / 365)
    if rng is not None:
        base += rng.normal(0, noise_std)
    return base


def make_series(type_name, n_years, rng):
    params = DEST_TYPES[type_name]
    days = np.arange(365 * n_years)
    doy = days % 365
    series = np.array([seasonal_value(d, params["amplitude"], params["peak_day"], rng=rng) for d in doy])
    return series, doy


# ---------------------------------------------------------------------
# 2. Build training windows: 14-day input sequence -> next-day target
#    Each timestep's input is [price_index, sin(day-of-year), cos(day-of-year)]
# ---------------------------------------------------------------------
SEQ_LEN = 14
INPUT_DIM = 3

def build_dataset(n_years=6, rng=None):
    X, Y = [], []
    for t in TYPE_NAMES:
        series, doy = make_series(t, n_years, rng)
        sin_d = np.sin(2 * np.pi * doy / 365)
        cos_d = np.cos(2 * np.pi * doy / 365)
        feats = np.stack([series, sin_d, cos_d], axis=1)  # (N, 3)
        for i in range(len(series) - SEQ_LEN - 1):
            X.append(feats[i:i + SEQ_LEN])
            Y.append(series[i + SEQ_LEN])
    return np.array(X), np.array(Y)


rng = np.random.RandomState(1)
X_train, Y_train = build_dataset(n_years=8, rng=rng)
X_test, Y_test = build_dataset(n_years=2, rng=np.random.RandomState(99))
print(f"Train sequences: {len(X_train)}, Test sequences: {len(X_test)}")

# ---------------------------------------------------------------------
# 3. Hand-rolled LSTM cell + output layer, with manual BPTT
# ---------------------------------------------------------------------
HIDDEN = 16
LR = 0.05
EPOCHS = 6

rng_init = np.random.RandomState(2)


def init_weights():
    z = INPUT_DIM + HIDDEN
    scale = 0.15
    return {
        "Wf": rng_init.randn(z, HIDDEN) * scale, "bf": np.zeros(HIDDEN),
        "Wi": rng_init.randn(z, HIDDEN) * scale, "bi": np.zeros(HIDDEN),
        "Wo": rng_init.randn(z, HIDDEN) * scale, "bo": np.zeros(HIDDEN),
        "Wg": rng_init.randn(z, HIDDEN) * scale, "bg": np.zeros(HIDDEN),
        "Wy": rng_init.randn(HIDDEN, 1) * scale, "by": np.zeros(1),
    }


W = init_weights()


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -30, 30)))


def lstm_forward(x_seq, W):
    """x_seq: (SEQ_LEN, INPUT_DIM). Returns cache for BPTT + final prediction."""
    h = np.zeros(HIDDEN)
    c = np.zeros(HIDDEN)
    cache = []
    for x_t in x_seq:
        z = np.concatenate([h, x_t])
        f = sigmoid(z @ W["Wf"] + W["bf"])
        i = sigmoid(z @ W["Wi"] + W["bi"])
        o = sigmoid(z @ W["Wo"] + W["bo"])
        g = np.tanh(z @ W["Wg"] + W["bg"])
        c_new = f * c + i * g
        h_new = o * np.tanh(c_new)
        cache.append((z, f, i, o, g, c, c_new, h, h_new))
        c, h = c_new, h_new
    y_pred = h @ W["Wy"] + W["by"]
    return y_pred, cache, h


def lstm_backward(cache, h_final, y_pred, y_true, W, lr):
    dWy = np.outer(h_final, (y_pred - y_true))
    dby = (y_pred - y_true)

    dh_next = (y_pred - y_true) * W["Wy"].flatten()
    dc_next = np.zeros(HIDDEN)

    grads = {k: np.zeros_like(v) for k, v in W.items()}
    grads["Wy"] = dWy
    grads["by"] = dby

    for t in reversed(range(len(cache))):
        z, f, i, o, g, c_prev, c_new, h_prev, h_new = cache[t]

        dh = dh_next
        do = dh * np.tanh(c_new)
        dc = dc_next + dh * o * (1 - np.tanh(c_new) ** 2)
        df = dc * c_prev
        di = dc * g
        dg = dc * i
        dc_prev = dc * f

        do_raw = do * o * (1 - o)
        df_raw = df * f * (1 - f)
        di_raw = di * i * (1 - i)
        dg_raw = dg * (1 - g ** 2)

        grads["Wo"] += np.outer(z, do_raw)
        grads["bo"] += do_raw
        grads["Wf"] += np.outer(z, df_raw)
        grads["bf"] += df_raw
        grads["Wi"] += np.outer(z, di_raw)
        grads["bi"] += di_raw
        grads["Wg"] += np.outer(z, dg_raw)
        grads["bg"] += dg_raw

        dz = (do_raw @ W["Wo"].T + df_raw @ W["Wf"].T + di_raw @ W["Wi"].T + dg_raw @ W["Wg"].T)
        dh_next = dz[:HIDDEN]
        dc_next = dc_prev

    for k in W:
        np.clip(grads[k], -5, 5, out=grads[k])
        W[k] -= lr * grads[k]


# ---------------------------------------------------------------------
# 4. Training loop
# ---------------------------------------------------------------------
n = len(X_train)
for epoch in range(EPOCHS):
    order = np.random.permutation(n)
    total_loss = 0.0
    for idx in order:
        y_pred, cache, h_final = lstm_forward(X_train[idx], W)
        loss = 0.5 * (y_pred[0] - Y_train[idx]) ** 2
        total_loss += loss
        lstm_backward(cache, h_final, y_pred, Y_train[idx], W, LR)
    print(f"Epoch {epoch+1}/{EPOCHS} - avg train loss: {total_loss/n:.6f}")

# ---------------------------------------------------------------------
# 5. Evaluate on held-out test sequences
# ---------------------------------------------------------------------
errors = []
for idx in range(len(X_test)):
    y_pred, _, _ = lstm_forward(X_test[idx], W)
    errors.append(abs(y_pred[0] - Y_test[idx]))
errors = np.array(errors)
print(f"\nTest MAE: {errors.mean():.4f}  (price-index units, e.g. 0.03 = ~3% typical error)")
print(f"Test max error: {errors.max():.4f}")

# spot-check: does it predict rising trend into a beach destination's peak season?
beach = DEST_TYPES["beach"]
test_rng = np.random.RandomState(555)
doy_start = 340  # early Dec, heading toward beach peak (day 355)
series_check = np.array([seasonal_value(d % 365, beach["amplitude"], beach["peak_day"], rng=test_rng) for d in range(doy_start, doy_start + SEQ_LEN)])
doy_check = np.arange(doy_start, doy_start + SEQ_LEN) % 365
feats_check = np.stack([series_check, np.sin(2*np.pi*doy_check/365), np.cos(2*np.pi*doy_check/365)], axis=1)
pred, _, _ = lstm_forward(feats_check, W)
true_next = seasonal_value((doy_start + SEQ_LEN) % 365, beach["amplitude"], beach["peak_day"])
print(f"\nBeach destination, 14 days into December (approaching peak season):")
print(f"  predicted next-day price index: {pred[0]:.3f}  (true seasonal curve: {true_next:.3f})")

# ---------------------------------------------------------------------
# 6. Export
# ---------------------------------------------------------------------
export = {
    "seq_len": SEQ_LEN,
    "hidden": HIDDEN,
    "dest_types": {k: {"amplitude": v["amplitude"], "peak_day": v["peak_day"], "keywords": v["keywords"]} for k, v in DEST_TYPES.items()},
    "weights": {k: v.tolist() for k, v in W.items()},
}
with open("lstm_weights.json", "w") as f:
    json.dump(export, f, indent=2)

print("\nSaved lstm_weights.json")