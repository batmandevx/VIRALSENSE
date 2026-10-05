"""Negative control on the team's Kaggle "Instagram Analytics" dataset (synthetic, 20 accounts).

    python -m viralsense.models.control

Same method as the main study: engagement rate = (likes + comments) / followers, 60/30/10 labels within
follower tiers fit on training accounts, account-disjoint test set + GroupKFold, pre-publication
features only. traffic_source is excluded: it describes how viewers found the post, so it only exists
after publishing. If our pipeline reports clearly-above-chance results from timing/content features on
data where they carry no signal, the pipeline is broken.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from viralsense.data.prepare import apply_labels, fit_labels, past_median_engagement
from viralsense.features.build import check_no_leakage
from viralsense.models.common import CLASSES, base_estimator, metrics
from viralsense.utils import ensure_dir, load_config, seed_everything

PRE = ["account_type", "media_type", "content_category", "has_call_to_action", "caption_length", "hashtags_count"]
POST_PUBLICATION = ["likes", "comments", "shares", "saves", "reach", "impressions", "engagement_rate",
                    "followers_gained", "performance_bucket_label", "traffic_source"]


def load_kaggle(path: str | Path) -> pd.DataFrame:
    raw = pd.read_csv(path, parse_dates=["post_datetime"])
    df = pd.DataFrame({
        "post_id": raw["post_id"], "account": raw["account_id"].astype(str), "followers": raw["follower_count"],
        "timestamp": raw["post_datetime"].dt.tz_localize("UTC"), "likes": raw["likes"], "comments": raw["comments"],
        **{c: raw[c] for c in PRE},
    })
    df["engagement_rate"] = (df["likes"] + df["comments"]) / df["followers"]
    return past_median_engagement(df)


def features(df: pd.DataFrame, with_history: bool) -> pd.DataFrame:
    h = df["timestamp"].dt.hour
    X = pd.DataFrame({
        "meta_hour_sin": np.sin(2 * np.pi * h / 24), "meta_hour_cos": np.cos(2 * np.pi * h / 24),
        "meta_weekday": df["timestamp"].dt.weekday, "follower_tier": df["follower_tier"],
        "cap_len": df["caption_length"], "cap_hashtag_count": df["hashtags_count"], "cap_has_cta": df["has_call_to_action"],
        "meta_is_brand": (df["account_type"] == "brand").astype(int),
    }, index=df.index)
    X = pd.concat([X, pd.get_dummies(df["media_type"], prefix="media", dtype=int),
                   pd.get_dummies(df["content_category"], prefix="topic", dtype=int)], axis=1)
    if with_history:
        X["acct_past_median_er"] = df["acct_past_median_er"].fillna(df["acct_past_median_er"].median())
        X["acct_n_past_posts"] = df["acct_n_past_posts"]
    return X


def _fit(name, X, y, cfg):
    m = make_pipeline(StandardScaler(), base_estimator(name, cfg))
    return m.fit(X, y, **{f"{m.steps[-1][0]}__sample_weight": compute_sample_weight("balanced", y)})


def main(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    seed = cfg["seed"]
    seed_everything(seed)
    tables = ensure_dir(Path(cfg["paths"]["reports"]) / "tables")
    df = load_kaggle(Path(cfg["control"]["kaggle_csv"]).expanduser())

    acc = df.groupby("account")["followers"].first().sort_values()
    rng = np.random.default_rng(seed)
    strata = np.array_split(acc.index.to_numpy(), 4)  # follower quartiles of the 20 accounts
    test_acc = {a for s in strata for a in rng.choice(s, 1, replace=False)}
    df["split"] = np.where(df["account"].isin(test_acc), "test", "train")
    spec = fit_labels(df[df.split == "train"], cfg["labels"]["n_tiers"], cfg["labels"]["quantiles"])
    df = apply_labels(df, spec, CLASSES)
    df["y"] = df["label"].map({c: i for i, c in enumerate(CLASSES)})
    tr, te = df[df.split == "train"], df[df.split == "test"]

    rows = []
    for with_hist in (False, True):
        Xtr, Xte = features(tr, with_hist), features(te, with_hist).reindex(columns=features(tr, with_hist).columns, fill_value=0)
        check_no_leakage(Xtr, cfg)
        assert not set(Xtr.columns) & set(POST_PUBLICATION)
        for name in ("dummy", "logreg", "rf", "xgb"):
            cv = []
            for f_tr, f_va in GroupKFold(cfg["split"]["n_folds"]).split(Xtr, groups=tr["account"]):
                m = _fit(name, Xtr.iloc[f_tr], tr["y"].iloc[f_tr].to_numpy(), cfg)
                cv.append(metrics(tr["y"].iloc[f_va].to_numpy(), m.predict_proba(Xtr.iloc[f_va]))["macro_f1"])
            m = _fit(name, Xtr, tr["y"].to_numpy(), cfg)
            test = metrics(te["y"].to_numpy(), m.predict_proba(Xte))
            rows.append({"features": "with account history" if with_hist else "post + timing only", "model": name,
                         "cv_macro_f1_mean": np.mean(cv), "cv_macro_f1_std": np.std(cv, ddof=1),
                         **{f"test_{k}": v for k, v in test.items()}})
            print(f"  {rows[-1]['features']:22s} {name:7s} CV {np.mean(cv):.3f} ± {np.std(cv, ddof=1):.3f} | "
                  f"test {test['macro_f1']:.3f}")
    out = pd.DataFrame(rows)
    out.attrs["n_accounts"] = df["account"].nunique()
    out.to_csv(tables / "kaggle_negative_control.csv", index=False)
    pd.DataFrame([{"posts": len(df), "accounts": df["account"].nunique(), "test_accounts": len(test_acc),
                   "train_posts": len(tr), "test_posts": len(te)}]).to_csv(tables / "kaggle_negative_control_data.csv", index=False)
    return out


if __name__ == "__main__":
    main()
