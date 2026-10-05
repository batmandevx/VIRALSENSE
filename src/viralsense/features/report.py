"""Phase 1 report: sample composition and feature sanity (training accounts only for anything label-related).

    python -m viralsense.features.report
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

from viralsense.features.build import feature_groups
from viralsense.plots import CLASS_COLORS, MUTED, SERIES, save
from viralsense.utils import ensure_dir, load_config


def main(cfg: dict | None = None) -> None:
    cfg = cfg or load_config()
    proc, rep = Path(cfg["paths"]["processed"]), Path(cfg["paths"]["reports"])
    tables, figs = ensure_dir(rep / "tables"), ensure_dir(rep / "figures")
    posts = pd.read_parquet(proc / "posts.parquet")
    feats = pd.read_parquet(proc / "features.parquet")

    comp = posts.groupby("split").agg(posts=("post_id", "size"), accounts=("account", "nunique"),
                                      median_followers=("followers", "median"),
                                      median_engagement_rate=("engagement_rate", "median"),
                                      first_post=("timestamp", "min"), last_post=("timestamp", "max"))
    comp.to_csv(tables / "phase1_sample_composition.csv")

    g = feature_groups(feats.columns)
    scalar = g["metadata"] + g["content_scalar"]
    health = pd.DataFrame({
        "missing_share": feats[scalar].isna().mean(), "n_unique": feats[scalar].nunique(),
        "mean": feats[scalar].mean(), "std": feats[scalar].std(), "min": feats[scalar].min(), "max": feats[scalar].max()})
    health.loc["clip_* (512)", ["missing_share", "std"]] = [feats[g["clip"]].isna().mean().mean(), feats[g["clip"]].std().mean()]
    health.loc["txt_* (384)", ["missing_share", "std"]] = [feats[g["txt"]].isna().mean().mean(), feats[g["txt"]].std().mean()]
    health.to_csv(tables / "phase1_feature_health.csv")

    tr = feats[feats.split == "train"]
    y = tr["label"].map({"Low": 0, "Moderate": 1, "Viral": 2})
    log_er = np.log10(tr["engagement_rate"] + 1e-6)
    assoc = pd.DataFrame({
        "spearman_vs_label": [spearmanr(tr[c], y, nan_policy="omit").statistic for c in scalar],
        "spearman_vs_log_er": [spearmanr(tr[c], log_er, nan_policy="omit").statistic for c in scalar],
        **{f"mean_{k}": [tr.loc[tr.label == k, c].mean() for c in scalar] for k in ("Low", "Moderate", "Viral")},
    }, index=scalar).sort_values("spearman_vs_label", key=np.abs, ascending=False)
    assoc.to_csv(tables / "phase1_feature_label_association_train.csv")

    pca_rows = []
    for name in ("clip", "txt"):
        p = PCA(cfg["features"]["pca_dims"], random_state=cfg["seed"]).fit(tr[g[name]])
        pca_rows.append({"block": name, "dims_in": len(g[name]), "dims_out": p.n_components_,
                         "explained_variance": p.explained_variance_ratio_.sum()})
    pd.DataFrame(pca_rows).to_csv(tables / "phase1_pca_variance.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.4, 0.3 * len(assoc) + 1))
    a = assoc["spearman_vs_label"].iloc[::-1]
    ax.barh(a.index, a.values, color=[SERIES[1] if v > 0 else SERIES[0] for v in a.values], height=0.6)
    ax.axvline(0, color=MUTED, linewidth=1)
    ax.set_xlabel("Spearman ρ with label (Low < Moderate < Viral), training accounts")
    ax.set_title("Single-feature association with the label")
    ax.grid(axis="y", visible=False)
    save(fig, figs / "phase1_feature_label_association.png")

    fig, ax = plt.subplots(figsize=(6, 3.6))
    for t in sorted(posts.follower_tier.unique()):
        v = np.log10(posts.loc[posts.follower_tier == t, "engagement_rate"] + 1e-6)
        ax.hist(v, bins=60, histtype="step", linewidth=2, color=SERIES[t], label=f"tier {t + 1}")
    ax.set_xlabel("log10 engagement rate")
    ax.set_ylabel("posts")
    ax.set_title("Engagement rate by follower tier")
    ax.legend()
    save(fig, figs / "phase1_engagement_by_tier.png")

    fig, ax = plt.subplots(figsize=(6, 3.4))
    share = pd.crosstab(posts.timestamp.dt.year, posts.label, normalize="index")[["Low", "Moderate", "Viral"]]
    left = np.zeros(len(share))
    for c in share.columns:
        ax.barh(share.index.astype(str), share[c], left=left, color=CLASS_COLORS[c], label=c, height=0.6,
                edgecolor="white", linewidth=1)
        left += share[c].to_numpy()
    ax.set_xlabel("share of posts")
    ax.set_title("Label mix by posting year (follower snapshot effect)")
    ax.legend(ncols=3, loc="lower right")
    ax.grid(axis="y", visible=False)
    save(fig, figs / "phase1_label_by_year.png")

    print(comp.to_string())
    print(health.round(4).to_string())
    print(assoc.round(3).to_string())
    print(pd.DataFrame(pca_rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
