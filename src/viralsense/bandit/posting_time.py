"""Phase 4a: LinUCB posting-time recommender (MABWiser), evaluated offline by replay.

    python -m viralsense.bandit.posting_time

Arms: 8 three-hour UTC slots. Context: post content + follower tier + account history
(never the hour itself). Reward: 1 if the post was labelled Moderate or Viral.

Replay (Li et al., 2011) walks the test accounts' posts in time order; an event counts only
when the policy picks the slot the post was actually published in. It is unbiased only when
the logging policy chose slots uniformly at random. Real creators did not, so the numbers
compare policies under that bias and are not a causal estimate of uplift.
"""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from viralsense.bandit.replay import POLICIES, Policy, evaluate  # noqa: F401  (Policy: unpickling)
from viralsense.features.build import select
from viralsense.models.common import load_table
from viralsense.plots import MUTED, SERIES, save
from viralsense.utils import ensure_dir, load_config, seed_everything

CTX_PCA = 8


def slot_of(hour, n_slots: int) -> np.ndarray:
    return (np.asarray(hour) // (24 // n_slots)).astype(int)


def slot_label(s: int, n_slots: int) -> str:
    w = 24 // n_slots
    return f"{s * w:02d}:00-{(s + 1) * w:02d}:00 UTC"


def context_columns(df: pd.DataFrame) -> list[str]:
    return ["follower_tier", "acct_past_median_er", "acct_n_past_posts"] + select(df.columns, "content_scalar", "clip", "txt")


def make_context(columns: list[str], seed: int) -> ColumnTransformer:
    num = [c for c in columns if c.startswith(("img_", "cap_", "acct_"))]
    parts = [
        ("tier", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["follower_tier"]),
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
    ]
    for p in ("clip", "txt"):
        cols = [c for c in columns if c.startswith(p + "_")]
        if cols:
            parts.append((p, Pipeline([("pca", PCA(CTX_PCA, random_state=seed)), ("sc", StandardScaler())]), cols))
    return ColumnTransformer(parts)


def reward_of(labels: pd.Series) -> np.ndarray:
    return labels.isin(["Moderate", "Viral"]).astype(int).to_numpy()


def main(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    seed = cfg["seed"]
    seed_everything(seed)
    b = cfg["bandit"]
    rep = Path(cfg["paths"]["reports"])
    tables, figs = ensure_dir(rep / "tables"), ensure_dir(rep / "figures")

    df = load_table(cfg)
    df["slot"] = slot_of(df["meta_hour"], b["n_slots"])
    df["reward"] = reward_of(df["label"])
    posts = pd.read_parquet(Path(cfg["paths"]["processed"]) / "posts.parquet")[["post_id", "timestamp"]]
    df = df.merge(posts, on="post_id").sort_values("timestamp", kind="stable")

    logged = df.groupby(["split", "slot"]).agg(n_posts=("reward", "size"), mean_reward=("reward", "mean")).reset_index()
    logged["slot_label"] = logged["slot"].map(lambda s: slot_label(s, b["n_slots"]))
    logged.to_csv(tables / "bandit_logged_slots.csv", index=False)

    cols = context_columns(df)
    tr, te = df[df.split == "train"], df[df.split == "test"]
    # Only slots seen in the training logs can be learned or replay-evaluated; an arm with no data
    # keeps LinUCB's full exploration bonus and, under replay, is never corrected.
    arms = sorted(int(a) for a in tr["slot"].unique())
    if len(arms) < b["n_slots"]:
        print(f"WARNING: only {len(arms)} of {b['n_slots']} slots appear in training logs; arms = {arms}")
    ctx = make_context(cols, seed).fit(tr[cols])
    Xtr, Xte = ctx.transform(tr[cols]), ctx.transform(te[cols])

    runs, summary, curves = evaluate(tr["slot"].to_numpy(), tr["reward"].to_numpy(), Xtr, te["slot"].to_numpy(),
                                     te["reward"].to_numpy(), Xte, arms, b["alpha"], seed, b["n_seeds"])
    pd.DataFrame({k: pd.Series(v) for k, v in curves.items()}).rename_axis("event").reset_index().to_csv(
        tables / "bandit_replay_curves.csv", index=False)
    runs.to_csv(tables / "bandit_replay_runs.csv", index=False)
    summary.to_csv(tables / "bandit_replay_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 3.6))
    for i, (k, c) in enumerate(curves.items()):
        ax.plot(np.arange(1, len(c) + 1), c, color=SERIES[i], label=k)
    ax.axhline(te["reward"].mean(), color=MUTED, linestyle="--", linewidth=1, label="logged average")
    ax.set_xlabel("test events streamed (time order)")
    ax.set_ylabel("running mean reward on matched events")
    ax.set_title("Replay evaluation of posting-time policies (seed 0)")
    ax.legend(fontsize=8, ncols=3)
    save(fig, figs / "bandit_replay.png")

    full = Policy("linucb", arms, b["alpha"], seed)
    full.fit(tr["slot"].to_numpy(), tr["reward"].to_numpy(), Xtr)
    joblib.dump({"policy": full, "context": ctx, "columns": cols, "n_slots": b["n_slots"]},
                ensure_dir(cfg["paths"]["artifacts"]) / "bandit.joblib")
    print(summary.round(4).to_string(index=False))
    return summary


def recommend(bundle: dict, row: pd.DataFrame) -> pd.DataFrame:
    """Expected reward per slot for one post, best first."""
    x = bundle["context"].transform(row[bundle["columns"]])
    exp = bundle["policy"].mab.predict_expectations(x)
    exp = exp[0] if isinstance(exp, list) else exp
    out = pd.DataFrame({"slot": list(exp), "expected_reward": list(exp.values())})
    out["slot_label"] = out["slot"].map(lambda s: slot_label(s, bundle["n_slots"]))
    return out.sort_values("expected_reward", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    main()
