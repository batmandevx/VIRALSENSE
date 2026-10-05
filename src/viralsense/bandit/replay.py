"""Shared bandit machinery: policies, offline replay (Li et al., 2011), bootstrap confidence intervals.

Replay walks logged events in time order and only counts an event when the policy picks the
logged arm. It is unbiased only if the logging policy was uniform random; creators were not, so
results compare policies under that bias and are not causal uplift estimates.
"""
import numpy as np
import pandas as pd
from mabwiser.mab import MAB, LearningPolicy

POLICIES = ["linucb", "lin_ts", "thompson", "eps_greedy", "best_fixed", "random"]
CONTEXTUAL = {"linucb", "lin_ts"}


class Policy:
    """Uniform wrapper so every policy is trained and replayed the same way."""

    def __init__(self, kind: str, arms: list, alpha: float, seed: int):
        self.kind, self.arms, self.rng = kind, list(arms), np.random.default_rng(seed)
        lp = {"linucb": LearningPolicy.LinUCB(alpha=alpha),
              "lin_ts": LearningPolicy.LinTS(alpha=alpha),
              "thompson": LearningPolicy.ThompsonSampling(),  # Beta-Bernoulli: reward must be 0/1
              "eps_greedy": LearningPolicy.EpsilonGreedy(epsilon=0.1)}.get(kind)
        self.mab = MAB(self.arms, lp, seed=seed) if lp is not None else None

    def fit(self, a, r, X):
        if self.kind == "best_fixed":
            means = pd.Series(r).groupby(np.asarray(a)).mean().reindex(self.arms).fillna(0)
            self.best = means.idxmax()
        elif self.mab is not None:
            self.mab.fit(list(a), list(r), X if self.kind in CONTEXTUAL else None)

    def predict(self, x):
        if self.kind == "random":
            return self.arms[int(self.rng.integers(len(self.arms)))]
        if self.kind == "best_fixed":
            return self.best
        return self.mab.predict(x[None, :]) if self.kind in CONTEXTUAL else self.mab.predict()

    def update(self, a, r, x):
        if self.mab is None:
            return
        if self.kind in CONTEXTUAL:
            self.mab.partial_fit([a], [r], x[None, :])
        else:
            self.mab.partial_fit([a], [r])


def replay(policy: Policy, a, r, X) -> dict:
    a, r = np.asarray(a), np.asarray(r)
    rewards, curve = [], []
    for t in range(len(a)):
        if policy.predict(X[t]) == a[t]:
            rewards.append(r[t])
            policy.update(a[t], r[t], X[t])
        curve.append(np.mean(rewards) if rewards else np.nan)
    return {"matched": len(rewards), "rewards": np.array(rewards, dtype=float),
            "mean_reward": float(np.mean(rewards)) if rewards else np.nan, "curve": curve}


def bootstrap_lift(rewards: np.ndarray, baseline: float, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% CI of (mean matched reward - baseline) by resampling the matched events."""
    if len(rewards) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = rng.choice(rewards, (n_boot, len(rewards)), replace=True).mean(1)
    lo, hi = np.percentile(means - baseline, [2.5, 97.5])
    return float(lo), float(hi)


def evaluate(train_a, train_r, Xtr, test_a, test_r, Xte, arms, alpha: float, seed: int, n_seeds: int):
    """Fit every policy on training logs, replay on the test stream. Returns (runs, summary, curves)."""
    logged = float(np.mean(test_r))
    rows, curves, pooled = [], {}, {}
    for kind in POLICIES:
        pooled[kind] = []
        for s in range(n_seeds):
            pol = Policy(kind, arms, alpha, seed + s)
            pol.fit(train_a, train_r, Xtr)
            res = replay(pol, test_a, test_r, Xte)
            rows.append({"policy": kind, "seed": s, "matched_events": res["matched"], "mean_reward": res["mean_reward"]})
            pooled[kind].append(res["rewards"])
            if s == 0:
                curves[kind] = res["curve"]
    runs = pd.DataFrame(rows)
    summ = runs.groupby("policy").agg(mean_reward=("mean_reward", "mean"), std_reward=("mean_reward", "std"),
                                      matched_events=("matched_events", "mean")).reindex(POLICIES).reset_index()
    # CI from one run (seed 0): pooling seeds would repeat identical events for deterministic policies
    # and shrink the interval artificially.
    ci = {k: bootstrap_lift(v[0], logged, seed=seed) for k, v in pooled.items()}
    summ["lift_vs_logged"] = summ["mean_reward"] - logged
    summ["lift_ci_low"] = summ["policy"].map(lambda k: ci[k][0])
    summ["lift_ci_high"] = summ["policy"].map(lambda k: ci[k][1])
    summ["ci_excludes_zero"] = (summ["lift_ci_low"] > 0) | (summ["lift_ci_high"] < 0)
    summ.loc[len(summ)] = {"policy": "logged (what creators did)", "mean_reward": logged, "std_reward": np.nan,
                           "matched_events": len(test_r), "lift_vs_logged": 0.0, "lift_ci_low": np.nan,
                           "lift_ci_high": np.nan, "ci_excludes_zero": False}
    return runs, summ, curves
