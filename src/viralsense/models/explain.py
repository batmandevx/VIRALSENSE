"""SHAP explanations for the deployed XGBoost classifier, with embedding PCs grouped."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from viralsense.plots import SERIES, save

PRETTY = {
    "acct_past_median_er": "past engagement (account)", "acct_n_past_posts": "number of past posts",
    "follower_tier": "follower tier", "meta_hour": "hour (UTC)", "meta_hour_sin": "hour (UTC)",
    "meta_hour_cos": "hour (UTC)", "meta_weekday": "weekday", "img_brightness": "brightness",
    "img_contrast": "contrast", "img_colourfulness": "colourfulness", "img_face_count": "face count",
    "cap_len": "caption length", "cap_emoji_count": "emoji count", "cap_hashtag_count": "hashtag count",
    "cap_has_question": "question in caption", "cap_sentiment": "caption sentiment",
    "acct_recent_median_er": "recent engagement (last 10)", "acct_past_er_iqr": "engagement volatility",
    "acct_er_trend": "engagement trend", "acct_days_since_last": "days since last post",
    "acct_posts_last_30d": "posts in last 30 days", "meta_log_followers": "followers (log)",
}


def group_name(transformed: str) -> str:
    block, _, col = transformed.partition("__")
    if block == "clip":
        return "image content (CLIP)"
    if block == "txt":
        return "caption meaning (MiniLM)"
    col = col.removeprefix("missingindicator_")
    if col.startswith("img_concept_"):
        return "looks like: " + col.removeprefix("img_concept_").replace("_", " ")
    if col.startswith("jury_"):
        return "jury " + col.removeprefix("jury_").replace("_", " ")
    return PRETTY.get(col, col)


def viral_shap(bundle: dict, X: pd.DataFrame) -> pd.DataFrame:
    """SHAP values (log-odds of Viral) per original row, summed into readable feature groups."""
    pipe = bundle["model"]
    Z = pipe.named_steps["pre"].transform(X[bundle["columns"]])
    names = pipe.named_steps["pre"].get_feature_names_out()
    sv = shap.TreeExplainer(pipe.named_steps["clf"]).shap_values(Z)
    sv = sv[:, :, 2] if np.ndim(sv) == 3 else np.asarray(sv[2])
    return pd.DataFrame(sv, columns=[group_name(n) for n in names]).T.groupby(level=0).sum().T


def importance_report(bundle: dict, X: pd.DataFrame, tables: Path, figs: Path, top: int = 12) -> pd.DataFrame:
    s = viral_shap(bundle, X)
    imp = s.abs().mean().sort_values(ascending=False).rename("mean_abs_shap_viral").rename_axis("feature").reset_index()
    imp.to_csv(tables / "shap_importance_viral.csv", index=False)
    head = imp.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6, 0.32 * len(head) + 1))
    ax.barh(head["feature"], head["mean_abs_shap_viral"], color=SERIES[0], height=0.6)
    ax.set_xlabel("mean |SHAP| on Viral log-odds (test accounts)")
    ax.set_title("What drives the Viral prediction")
    ax.grid(axis="y", visible=False)
    save(fig, figs / "shap_importance_viral.png")
    return imp
