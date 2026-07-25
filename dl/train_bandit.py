import json
import numpy as np

np.random.seed(42)

CATEGORIES = ["Accommodation", "Food & Dining", "Local Transport", "Activities & Sightseeing", "Shopping & Misc."]
N_OUT = len(CATEGORIES)

CHIP_FEATURES = ["too_expensive", "too_much_walking", "more_food", "more_culture", "too_packed"]
CHIP_PROFILE = {
    "too_expensive":    [3, 2, 1, -3, -3],
    "too_much_walking": [8, 0, 0, -5, -3],
    "more_food":        [-3, 10, 0, -4, -3],
    "more_culture":     [2, 3, 0, 3, -8],
    "too_packed":       [8, 0, 0, -5, -3],
}
CHIP_INSTRUCTION = {
    "too_expensive": "favor free or low-cost attractions and budget-friendly meals",
    "too_much_walking": "reduce walking-heavy activities and space things out with more rest time",
    "more_food": "include more varied restaurant and food-experience stops",
    "more_culture": "prioritize museums, heritage sites, and cultural experiences",
    "too_packed": "space the day out with fewer stops and more free/rest time",
}

KEYWORD_FEATURES = ["shopping", "nightlife", "nature", "luxury", "family"]
KEYWORD_PROFILE = {
    "shopping": [-3, -2, 0, -5, 10],
    "nightlife": [-4, 5, -1, 5, -5],
    "nature": [-3, -2, 1, 10, -6],
    "luxury": [12, 3, 0, -5, -10],
    "family": [2, 4, 0, 4, -10],
}

FEATURES = CHIP_FEATURES + KEYWORD_FEATURES
N_FEATURES = len(FEATURES)

ALL_PROFILES = {**CHIP_PROFILE, **KEYWORD_PROFILE}
for k, v in ALL_PROFILES.items():
    assert sum(v) == 0, f"{k} profile does not sum to 0"
PROFILE_MATRIX = np.array([ALL_PROFILES[f] for f in FEATURES])  # 10 x 5


def target_for(feature_values):
    """Ideal blended delta vector. Uses a max(1, total) denominator rather
    than always dividing by the raw total: this way a single WEAK signal
    (e.g. luxury at 0.17 confidence) produces a proportionally SMALL
    adjustment, while multiple/strong signals (total >= 1) still blend
    as a proper weighted average rather than growing unboundedly."""
    feature_values = np.asarray(feature_values, dtype=np.float64)
    total = feature_values.sum()
    if total <= 1e-8:
        return np.zeros(N_OUT)
    denom = max(1.0, total)
    return (feature_values[:, None] * PROFILE_MATRIX).sum(axis=0) / denom


# ---------------------------------------------------------------------
# Reward normalizer (sampled, same style as v4)
# ---------------------------------------------------------------------
def sample_context(rng):
    chips = (rng.rand(len(CHIP_FEATURES)) < 0.3).astype(np.float64)
    # keywords: mostly 0 (no signal), sometimes a partial/strong score
    kw = np.zeros(len(KEYWORD_FEATURES))
    for i in range(len(KEYWORD_FEATURES)):
        if rng.rand() < 0.4:
            kw[i] = rng.uniform(0.15, 1.0)
    return np.concatenate([chips, kw])


rng_check = np.random.RandomState(0)
_sample_targets = np.array([target_for(sample_context(rng_check)) for _ in range(4000)])
MAX_DIST = np.linalg.norm(_sample_targets[:, None, :] - _sample_targets[None, :, :], axis=-1).max()
if MAX_DIST == 0:
    MAX_DIST = 1.0


def reward_for(action, feature_values):
    target = target_for(feature_values)
    dist = np.linalg.norm(action - target)
    return 1.0 - dist / MAX_DIST

# ---------------------------------------------------------------------
# Deep network: 10 -> 20 (ReLU) -> 14 (ReLU) -> 5 (linear, policy mean)
# ---------------------------------------------------------------------
H1, H2 = 20, 14
LR = 0.012
SIGMA_EXPLORE = 6.0
SIGMA_EXPLOIT = 1.5
EPISODES = 250000
eps_start, eps_end = 1.0, 0.05

W1 = np.random.randn(N_FEATURES, H1) * 0.35
b1 = np.zeros(H1)
W2 = np.random.randn(H1, H2) * 0.35
b2 = np.zeros(H2)
W3 = np.random.randn(H2, N_OUT) * 0.35
b3 = np.zeros(N_OUT)


def forward(x):
    z1 = x @ W1 + b1
    h1 = np.maximum(0, z1)
    z2 = h1 @ W2 + b2
    h2 = np.maximum(0, z2)
    mu = h2 @ W3 + b3
    return z1, h1, z2, h2, mu


def policy_gradient_step(x, z1, h1, z2, h2, mu, action, advantage, sigma_used, lr):
    global W1, b1, W2, b2, W3, b3
    d_mu = -advantage * (action - mu) / (sigma_used ** 2)
    d_mu = np.clip(d_mu, -8, 8)

    dW3 = np.outer(h2, d_mu)
    db3 = d_mu
    dh2 = d_mu @ W3.T
    dz2 = dh2 * (z2 > 0)
    dW2 = np.outer(h1, dz2)
    db2 = dz2
    dh1 = dz2 @ W2.T
    dz1 = dh1 * (z1 > 0)
    dW1 = np.outer(x, dz1)
    db1 = dz1

    W3 -= lr * dW3
    b3 -= lr * db3
    W2 -= lr * dW2
    b2 -= lr * db2
    W1 -= lr * dW1
    b1 -= lr * db1
    for w in (W1, W2, W3):
        np.clip(w, -10, 10, out=w)


train_rng = np.random.RandomState(7)
baseline = 0.5
baseline_lr = 0.01

for ep in range(EPISODES):
    eps = eps_start + (eps_end - eps_start) * (ep / EPISODES)

    roll = ep % 10
    if roll == 0:
        context = np.zeros(N_FEATURES)
    elif roll == 1:
        # pure single feature, full strength (1.0) - covers both chip and
        # keyword "clean signal" cases, including partial-strength keywords
        context = np.zeros(N_FEATURES)
        idx = train_rng.randint(N_FEATURES)
        context[idx] = 1.0 if idx < len(CHIP_FEATURES) else train_rng.choice([0.3, 0.6, 1.0])
    elif roll <= 3:
        context = sample_context(train_rng)
    else:
        context = sample_context(train_rng)

    z1, h1, z2, h2, mu = forward(context)
    mu = np.clip(mu, -40, 40)

    sigma_used = SIGMA_EXPLORE if train_rng.rand() < eps else SIGMA_EXPLOIT
    action = mu + train_rng.randn(N_OUT) * sigma_used

    r = reward_for(action, context)
    advantage = r - baseline
    baseline += baseline_lr * (r - baseline)

    policy_gradient_step(context, z1, h1, z2, h2, mu, action, advantage, sigma_used, LR)

# ---------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------
test_rng = np.random.RandomState(123)
n_test = 3000
rewards = []
worst = 1.0
worst_info = None
for _ in range(n_test):
    ctx = sample_context(test_rng)
    _, _, _, _, mu = forward(ctx)
    r = reward_for(mu, ctx)
    rewards.append(r)
    if r < worst:
        worst = r
        worst_info = (ctx.round(2).tolist(), mu.round(2).tolist(), target_for(ctx).round(2).tolist())

print(f"Sampled average reward: {np.mean(rewards):.4f}")
print(f"Sampled worst-case reward: {worst:.4f}")
print(f"Worst case -> context: {dict(zip(FEATURES, worst_info[0]))}")
print(f"  policy mean: {worst_info[1]}")
print(f"  ideal blend: {worst_info[2]}")

# specific check: partial-confidence keyword score (e.g. LLM unsure, 0.5)
ctx = np.zeros(N_FEATURES)
ctx[FEATURES.index("luxury")] = 0.5
_, _, _, _, mu = forward(ctx)
print("\n'luxury' at 0.5 confidence (partial signal):")
print("  policy mean:", mu.round(2).tolist())
print("  ideal blend:", target_for(ctx).round(2).tolist())

ctx2 = np.zeros(N_FEATURES)
ctx2[FEATURES.index("luxury")] = 1.0
_, _, _, _, mu2 = forward(ctx2)
print("'luxury' at 1.0 confidence (full signal):")
print("  policy mean:", mu2.round(2).tolist())

# ---------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------
export = {
    "features": FEATURES,
    "chip_features": CHIP_FEATURES,
    "keyword_features": KEYWORD_FEATURES,
    "categories": CATEGORIES,
    "chip_instructions": CHIP_INSTRUCTION,
    "W1": W1.tolist(), "b1": b1.tolist(),
    "W2": W2.tolist(), "b2": b2.tolist(),
    "W3": W3.tolist(), "b3": b3.tolist(),
}

with open("bandit_weights.json", "w") as f:
    json.dump(export, f, indent=2)

print("\nSaved bandit_weights.json (v5 - continuous keyword scores, for LLM-based scoring)")