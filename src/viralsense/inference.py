"""Featurise a new, unpublished post exactly like the training data."""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from viralsense.features.metadata import metadata_features
from viralsense.features.text import text_features
from viralsense.features.visual import visual_features


def load_label_spec(cfg: dict) -> dict:
    spec = json.loads((Path(cfg["paths"]["processed"]) / "label_spec.json").read_text())
    spec["cuts"] = {int(k): v for k, v in spec["cuts"].items()}
    return spec


def follower_tier(followers: float, spec: dict) -> int:
    return int(np.searchsorted(spec["tier_edges"], followers, side="right"))


def history_row(past_median_er: float | None = None, n_past_posts: int = 0, history: dict | None = None) -> dict:
    """All account-history inputs for a new post. Unknown values stay NaN and are imputed with the
    training medians inside the model pipeline. With only a typical engagement rate, recent = typical
    and trend = 0 (no evidence of change)."""
    from viralsense.data.prepare import HISTORY_COLS

    h = {c: np.nan for c in HISTORY_COLS}
    h["acct_n_past_posts"] = n_past_posts
    if past_median_er is not None:
        h.update(acct_past_median_er=past_median_er, acct_recent_median_er=past_median_er, acct_er_trend=0.0)
    h.update({k: v for k, v in (history or {}).items() if k in h})
    return h


def post_frame(image_path: str, captions: list[str], followers: float, when: datetime, spec: dict,
               past_median_er: float | None = None, n_past_posts: int = 0, history: dict | None = None) -> pd.DataFrame:
    when = when if when.tzinfo else when.replace(tzinfo=timezone.utc)
    n = len(captions)
    h = history_row(past_median_er, n_past_posts, history)
    return pd.DataFrame({
        "post_id": [f"new_{i}" for i in range(n)], "image_path": [image_path] * n, "caption": captions,
        "timestamp": pd.to_datetime([when] * n, utc=True), "followers": [followers] * n,
        "follower_tier": [follower_tier(followers, spec)] * n, **{k: [v] * n for k, v in h.items()},
    })


def featurize(posts: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """One feature row per caption; the image is embedded once and shared."""
    vis = visual_features(posts.head(1), cfg).drop(columns="post_id")
    vis = pd.concat([vis] * len(posts), ignore_index=True)
    txt = text_features(posts, cfg).drop(columns="post_id")
    meta = metadata_features(posts).reset_index(drop=True)
    return pd.concat([meta, vis, txt], axis=1)


def load_model(path) -> dict:
    """Load a saved model bundle for inference in a process that also runs torch (CLIP).

    LightGBM/XGBoost multi-threaded prediction segfaults on macOS once torch's OpenMP runtime is loaded,
    so every estimator in the bundle is switched to a single thread (one post predicts instantly anyway).
    """
    import joblib

    bundle = joblib.load(path)
    model = bundle.get("model")
    if model is not None and hasattr(model, "get_params"):
        model.set_params(**{k: 1 for k in model.get_params(deep=True) if k.endswith("n_jobs")})
    return bundle
