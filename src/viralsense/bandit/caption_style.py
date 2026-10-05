"""Phase 4c: caption-style bandit (idea from the team's Colab notebook, rebuilt on the real data).

    python -m viralsense.bandit.caption_style

Arms: caption length (short/medium/long, training tertiles) x hashtags (0 / 1-5 / 6+) x question (yes/no).
Context: what is fixed before the caption is written (image, follower tier, account history).
Reward: 1 if the post was Moderate or Viral. Evaluated by replay with bootstrap CIs (see replay.py);
per-tier Beta posteriors give the recommendations shown in the dashboard.
"""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import beta

from viralsense.bandit.posting_time import make_context, reward_of
from viralsense.bandit.replay import evaluate
from viralsense.features.build import select
from viralsense.models.common import load_table
from viralsense.plots import MUTED, SERIES, save
from viralsense.utils import ensure_dir, load_config, seed_everything

TAG_BINS, TAG_LABELS = [-1, 0, 5, 10_000], ["0 tags", "1-5 tags", "6+ tags"]


def style_arm(cap_len, n_tags, has_q, len_edges) -> np.ndarray:
    length = np.asarray(pd.cut(np.asarray(cap_len, float), [-1, *len_edges, 1e9], labels=["short", "medium", "long"]),
                        dtype=str)
    tags = np.asarray(pd.cut(np.asarray(n_tags, float), TAG_BINS, labels=TAG_LABELS), dtype=str)
    q = np.where(np.asarray(has_q) == 1, "question", "no question")
    return np.char.add(np.char.add(np.char.add(np.char.add(length, " · "), tags), " · "), q)


def context_columns(df: pd.DataFrame) -> list[str]:
    return ["follower_tier", "acct_past_median_er", "acct_n_past_posts"] + \
        [c for c in select(df.columns, "content_scalar") if c.startswith("img_")] + select(df.columns, "clip")


def posterior_by_tier(tr: pd.DataFrame) -> pd.DataFrame:
    g = tr.groupby(["follower_tier", "arm"])["reward"].agg(n="size", successes="sum").reset_index()
    a, b = 1 + g["successes"], 1 + g["n"] - g["successes"]
    g["posterior_mean"] = a / (a + b)
    g["ci_low"], g["ci_high"] = beta.ppf(0.025, a, b), beta.ppf(0.975, a, b)
    return g.sort_values(["follower_tier", "posterior_mean"], ascending=[True, False])


def main(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    seed = cfg["seed"]
    seed_everything(seed)
    b = cfg["bandit"]
    rep = Path(cfg["paths"]["reports"])
    tables, figs = ensure_dir(rep / "tables"), ensure_dir(rep / "figures")

    df = load_table(cfg)
    ts = pd.read_parquet(Path(cfg["paths"]["processed"]) / "posts.parquet", columns=["post_id", "timestamp"])
    df = df.merge(ts, on="post_id").sort_values("timestamp", kind="stable")
    df["reward"] = reward_of(df["label"])
    tr0 = df[df.split == "train"]
    len_edges = np.quantile(tr0["cap_len"], [1 / 3, 2 / 3]).tolist()
    df["arm"] = style_arm(df["cap_len"], df["cap_hashtag_count"], df["cap_has_question"], len_edges)
    tr, te = df[df.split == "train"], df[df.split == "test"]
    arms = sorted(tr["arm"].unique())

    cols = context_columns(df)
    ctx = make_context(cols, seed).fit(tr[cols])
    runs, summary, curves = evaluate(tr["arm"].to_numpy(), tr["reward"].to_numpy(), ctx.transform(tr[cols]),
                                     te["arm"].to_numpy(), te["reward"].to_numpy(), ctx.transform(te[cols]),
                                     arms, b["alpha"], seed, b["n_seeds"])
    runs.to_csv(tables / "caption_bandit_runs.csv", index=False)
    summary.to_csv(tables / "caption_bandit_summary.csv", index=False)
    pd.DataFrame({k: pd.Series(v) for k, v in curves.items()}).rename_axis("event").reset_index().to_csv(
        tables / "caption_bandit_curves.csv", index=False)

    post = posterior_by_tier(tr)
    post.to_csv(tables / "caption_style_by_tier.csv", index=False)
    arm_stats = df.groupby(["split", "arm"])["reward"].agg(n_posts="size", reward_rate="mean").reset_index()
    arm_stats.to_csv(tables / "caption_bandit_arms.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for i, (k, c) in enumerate(curves.items()):
        ax.plot(np.arange(1, len(c) + 1), c, color=SERIES[i], label=k)
    ax.axhline(te["reward"].mean(), color=MUTED, linestyle="--", linewidth=1, label="logged average")
    ax.set_xlabel("test events streamed (time order)")
    ax.set_ylabel("running mean reward on matched events")
    ax.set_title("Replay evaluation of caption-style policies (seed 0)")
    ax.legend(fontsize=8, ncols=3)
    save(fig, figs / "caption_bandit_replay.png")

    joblib.dump({"len_edges": len_edges, "posterior": post, "arms": arms},
                ensure_dir(cfg["paths"]["artifacts"]) / "caption_style.joblib")
    print(summary.round(4).to_string(index=False))
    return summary


if __name__ == "__main__":
    main()
