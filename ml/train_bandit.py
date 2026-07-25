"""
Voyager AI - Feedback Adjustment Model (v4)
----------------------------------------------
v3 was a genuine deep RL contextual bandit, but its input was limited to
the 5 preset feedback chips - free-text feedback reached the Groq prompt
but never reached this trained model, so typing something like "more
luxury" or "shopping-focused" moved the itinerary TEXT but never moved
the Budget Breakdown NUMBERS.

v4 fixes this by extending the context from 5 -> 10 features:
    5 original chip flags (unchanged)
  + 5 new KEYWORD categories detected in the free-text box via simple
    substring matching (shopping, nightlife, nature/outdoors, luxury,
    family-friendly)

Each keyword category gets its own hand-designed "ideal profile" vector,
exactly like the chips did in v2/v3. The blended training target for any
context is still the elementwise average of ALL active features' profiles
(chips AND keywords together) - so a user who selects "Too packed" AND
types "we want something more luxury" gets a properly blended adjustment
reflecting both signals, not just whichever one happened to be a chip.

Same training method as v3: 2 hidden layers (deep), epsilon-greedy
exploration, REINFORCE policy-gradient updates against a simulated
reward (1 - normalized distance to the ideal blended target).
"""

import json
import numpy as np

np.random.seed(42)

CATEGORIES = ["Accommodation", "Food & Dining", "Local Transport", "Activities & Sightseeing", "Shopping & Misc."]
N_OUT = len(CATEGORIES)

# ---------------------------------------------------------------------
# 1. Chip features (unchanged from v3)
# ---------------------------------------------------------------------
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

# ---------------------------------------------------------------------
# 2. NEW: keyword features, detected from the free-text feedback box
#    via simple case-insensitive substring matching (deterministic,
#    not a learned classifier - keeps this separate from the ML model,
#    whose job is only to turn "which features are active" into a
#    budget-delta blend, same as it always did for chips).
# ---------------------------------------------------------------------
KEYWORD_FEATURES = ["shopping", "nightlife", "nature", "luxury", "family"]

KEYWORD_TRIGGERS = {
    "shopping": ["shopping", "shop", "mall", "market", "boutique", "souvenir"],
    "nightlife": ["nightlife", "bar", "club", "party", "pub", "night out"],
    "nature": ["nature", "hiking", "hike", "outdoor", "outdoors", "wildlife", "trek"],
    "luxury": ["luxury", "premium", "5-star", "5 star", "upscale", "high-end", "high end"],
    "family": ["family", "kids", "kid-friendly", "children", "child friendly"],
}

KEYWORD_PROFILE = {
    "shopping": [-3, -2, 0, -5, 10],
    "nightlife": [-4, 5, -1, 5, -5],
    "nature": [-3, -2, 1, 10, -6],
    "luxury": [12, 3, 0, -5, -10],
    "family": [2, 4, 0, 4, -10],
}

# ---------------------------------------------------------------------
# 3. Combine into one feature list + profile matrix
# ---------------------------------------------------------------------
FEATURES = CHIP_FEATURES + KEYWORD_FEATURES
N_FEATURES = len(FEATURES)

ALL_PROFILES = {**CHIP_PROFILE, **KEYWORD_PROFILE}
for k, v in ALL_PROFILES.items():
    assert sum(v) == 0, f"{k} profile does not sum to 0"

PROFILE_MATRIX = np.array([ALL_PROFILES[f] for f in FEATURES])  # 10 x 5


def target_for(context_bits):
    """Ideal blended delta vector: average of all ACTIVE features' own profiles
    (chips and keywords treated identically here)."""
    active_idx = [i for i, b in enumerate(context_bits) if b]
    if not active_idx:
        return np.zeros(N_OUT)
    return PROFILE_MATRIX[active_idx].mean(axis=0)


# ---------------------------------------------------------------------
# Reward normalizer: sample many random contexts rather than enumerate
# all 2^10 = 1024 (fine either way at this size, but sampling is what
# we'll also use for training density, so stay consistent)
# ---------------------------------------------------------------------
rng_check = np.random.RandomState(0)
_sample_targets = np.array([
    target_for(rng_check.randint(0, 2, size=N_FEATURES)) for _ in range(4000)
])
MAX_DIST = np.linalg.norm(_sample_targets[:, None, :] - _sample_targets[None, :, :], axis=-1).max()
if MAX_DIST == 0:
    MAX_DIST = 1.0


def reward_for(action, context_bits):
    target = target_for(context_bits)
    dist = np.linalg.norm(action - target)
    return 1.0 - dist / MAX_DIST

# ---------------------------------------------------------------------
# Deep network: 10 -> 20 (ReLU) -> 14 (ReLU) -> 5 (linear, policy mean)
# (widened slightly vs v3 since the input space is now larger)
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


# ---------------------------------------------------------------------
# Training: sample random contexts (biased toward fewer active features,
# since realistic feedback rarely activates most of 10 flags at once -
# but still cover the full space so it generalizes)
# ---------------------------------------------------------------------
baseline = 0.5
baseline_lr = 0.01

for ep in range(EPISODES):
    eps = eps_start + (eps_end - eps_start) * (ep / EPISODES)

    # curriculum: guarantee exposure to the highest-magnitude cases
    # (pure single-feature contexts), which are otherwise under-represented
    # relative to blended multi-feature contexts under plain Bernoulli sampling
    roll = ep % 10
    if roll == 0:
        bits = np.zeros(N_FEATURES)
    elif roll == 1:
        bits = np.ones(N_FEATURES)
    elif roll <= 3:
        bits = np.zeros(N_FEATURES)
        bits[np.random.randint(N_FEATURES)] = 1  # pure single-feature case
    else:
        bits = (np.random.rand(N_FEATURES) < 0.3).astype(np.float64)

    context = bits

    z1, h1, z2, h2, mu = forward(context)
    mu = np.clip(mu, -40, 40)

    sigma_used = SIGMA_EXPLORE if np.random.rand() < eps else SIGMA_EXPLOIT
    action = mu + np.random.randn(N_OUT) * sigma_used

    r = reward_for(action, context)
    advantage = r - baseline
    baseline += baseline_lr * (r - baseline)

    policy_gradient_step(context, z1, h1, z2, h2, mu, action, advantage, sigma_used, LR)

# ---------------------------------------------------------------------
# Evaluate on a large random sample (2^10 = 1024 is cheap too - do both)
# ---------------------------------------------------------------------
test_rng = np.random.RandomState(123)
n_test = 2000
rewards = []
worst = 1.0
worst_info = None
for _ in range(n_test):
    bits = (test_rng.rand(N_FEATURES) < 0.3).astype(np.float64)
    _, _, _, _, mu = forward(bits)
    r = reward_for(mu, bits)
    rewards.append(r)
    if r < worst:
        worst = r
        worst_info = ([FEATURES[j] for j, v in enumerate(bits) if v], mu.round(2).tolist(), target_for(bits).round(2).tolist())

print(f"Sampled average reward: {np.mean(rewards):.4f}")
print(f"Sampled worst-case reward: {worst:.4f}")
print(f"Worst case -> active: {worst_info[0]}")
print(f"  policy mean: {worst_info[1]}")
print(f"  ideal blend: {worst_info[2]}")

# specific check: a chip + a keyword together
ctx = np.zeros(N_FEATURES)
ctx[CHIP_FEATURES.index("too_packed")] = 1
ctx[N_FEATURES - len(KEYWORD_FEATURES) + KEYWORD_FEATURES.index("luxury")] = 1
_, _, _, _, mu = forward(ctx)
print("\n'too_packed' chip + 'luxury' keyword:")
print("  policy mean:", mu.round(2).tolist())
print("  ideal blend:", target_for(ctx).round(2).tolist())

# ---------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------
export = {
    "features": FEATURES,
    "chip_features": CHIP_FEATURES,
    "keyword_features": KEYWORD_FEATURES,
    "keyword_triggers": KEYWORD_TRIGGERS,
    "categories": CATEGORIES,
    "chip_instructions": CHIP_INSTRUCTION,
    "W1": W1.tolist(), "b1": b1.tolist(),
    "W2": W2.tolist(), "b2": b2.tolist(),
    "W3": W3.tolist(), "b3": b3.tolist(),
}

with open("bandit_weights.json", "w") as f:
    json.dump(export, f, indent=2)

print("\nSaved bandit_weights.json (v4 - 10 features: 5 chips + 5 free-text keyword categories)")