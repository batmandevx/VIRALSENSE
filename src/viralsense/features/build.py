"""Assemble the feature matrix and guard it against target leakage.

    python -m viralsense.features.build

Raw CLIP/MiniLM embeddings are stored; PCA to 32 dims lives in the model pipeline
(``make_preprocessor``) so it is fit on training folds only.
"""
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from viralsense.features.metadata import metadata_features
from viralsense.utils import load_config, seed_everything

ID_COLS = ["post_id", "account", "split", "fold", "label", "engagement_rate"]


class LeakageError(AssertionError):
    pass


def leaky_columns(columns, cfg: dict) -> list[str]:
    f = cfg["features"]
    bad = set(f["forbidden_tokens"])
    # Jury ratings never see engagement (tests/test_jury.py); "comment_trigger" only names a factor.
    jury = {f"jury_{k}_{s}" for k in cfg["jury"]["factors"] for s in ("mean", "std")}
    allowed = set(f["history_whitelist"]) | jury
    return [c for c in columns if c not in allowed and bad & set(c.lower().split("_"))]


def check_no_leakage(X: pd.DataFrame, cfg: dict) -> None:
    leaks = leaky_columns(X.columns, cfg)
    if leaks:
        raise LeakageError(f"target-derived columns in feature matrix: {leaks}")


def feature_groups(columns) -> dict[str, list[str]]:
    cols = list(columns)
    g = {
        "clip": [c for c in cols if c.startswith("clip_")],
        "txt": [c for c in cols if c.startswith("txt_")],
        "jury": [c for c in cols if c.startswith("jury_")],
    }
    g["metadata"] = [c for c in cols if c.startswith(("meta_", "acct_")) or c == "follower_tier"]
    g["content_scalar"] = [c for c in cols if c.startswith(("img_", "cap_"))]
    return g


def select(columns, *groups: str) -> list[str]:
    g = feature_groups(columns)
    return [c for name in groups for c in g[name]]


def make_preprocessor(columns, pca_dims: int, seed: int, scale: bool = True) -> ColumnTransformer:
    g = feature_groups(columns)
    num = g["metadata"] + g["content_scalar"] + g["jury"]
    num_steps = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    parts = [("num", Pipeline(num_steps), num)] if num else []
    for name in ("clip", "txt"):
        if g[name]:
            parts.append((name, PCA(n_components=min(pca_dims, len(g[name])), random_state=seed), g[name]))
    return ColumnTransformer(parts, remainder="drop")


def build(cfg: dict | None = None, posts: pd.DataFrame | None = None) -> pd.DataFrame:
    from viralsense.features.text import text_features
    from viralsense.features.visual import visual_features

    cfg = cfg or load_config()
    seed_everything(cfg["seed"])
    proc = Path(cfg["paths"]["processed"])
    posts = posts if posts is not None else pd.read_parquet(proc / "posts.parquet")

    feats = posts[ID_COLS].copy()
    for part in (metadata_features(posts), visual_features(posts, cfg), text_features(posts, cfg)):
        feats = feats.merge(part, on="post_id", validate="one_to_one")
    check_no_leakage(feats.drop(columns=ID_COLS), cfg)
    feats.to_parquet(proc / "features.parquet", index=False)
    print(f"features: {feats.shape[0]} rows x {feats.shape[1] - len(ID_COLS)} columns")
    return feats


if __name__ == "__main__":
    build()
