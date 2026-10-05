import numpy as np
import pandas as pd
import pytest

from viralsense.data.prepare import past_median_engagement
from viralsense.features.build import ID_COLS, LeakageError, check_no_leakage, feature_groups, make_preprocessor
from viralsense.features.metadata import metadata_features
from viralsense.features.text import caption_stats

TARGET_COLS = ["likes", "comments", "engagement_rate", "label"]


def _feature_frame(posts):
    rng = np.random.default_rng(0)
    X = metadata_features(posts).drop(columns="post_id")
    X = pd.concat([X, caption_stats(posts["caption"]).set_index(X.index)], axis=1)
    for p, d in (("clip", 8), ("txt", 6)):
        for i in range(d):
            X[f"{p}_{i:03d}"] = rng.normal(size=len(X))
    return X


def test_pipeline_features_pass_guard(posts, cfg):
    check_no_leakage(_feature_frame(posts), cfg)


@pytest.mark.parametrize("col", TARGET_COLS + ["log_engagement_rate", "viral_prob", "n_likes", "comment_count", "target"])
def test_guard_rejects_target_derived_column(posts, cfg, col):
    X = _feature_frame(posts)
    X[col] = 0.0
    with pytest.raises(LeakageError):
        check_no_leakage(X, cfg)


def test_model_inputs_never_include_targets_or_ids(posts, cfg):
    X = _feature_frame(posts)
    for c in ID_COLS + TARGET_COLS:
        X[c] = 0
    used = {c for cols in feature_groups(X.columns).values() for c in cols}
    assert not used & set(ID_COLS + TARGET_COLS)
    pre = make_preprocessor(X.columns, pca_dims=4, seed=0).fit(X.drop(columns=["split", "account", "post_id", "label"]))
    consumed = {c for _, _, cols in pre.transformers_ if isinstance(cols, list) for c in cols}
    assert not consumed & set(ID_COLS + TARGET_COLS)


def test_history_feature_ignores_current_and_future_posts(posts):
    base = past_median_engagement(posts)
    acct = posts["account"].iloc[0]
    rows = posts.index[posts["account"] == acct]
    order = posts.loc[rows, "timestamp"].sort_values().index
    pivot = order[len(order) // 2]

    later = posts.copy()
    later.loc[later["timestamp"] >= later.loc[pivot, "timestamp"], "engagement_rate"] *= 100
    changed = past_median_engagement(later)
    at_or_before = (posts["account"] == acct) & (posts["timestamp"] <= posts.loc[pivot, "timestamp"])
    np.testing.assert_array_equal(base.loc[at_or_before, "acct_past_median_er"],
                                  changed.loc[at_or_before, "acct_past_median_er"])

    earlier = posts.copy()
    earlier.loc[order[0], "engagement_rate"] *= 1000
    assert not np.allclose(base.loc[order[1:], "acct_past_median_er"].fillna(-1),
                           past_median_engagement(earlier).loc[order[1:], "acct_past_median_er"].fillna(-1))


def test_first_post_has_no_history(posts):
    out = past_median_engagement(posts)
    first = out.sort_values("timestamp").groupby("account").head(1)
    assert first["acct_past_median_er"].isna().all()
    assert (first["acct_n_past_posts"] == 0).all()


def test_history_feature_matches_strictly_earlier_reference(posts):
    out = past_median_engagement(posts)
    for _, r in out.sample(200, random_state=0).iterrows():
        prior = posts[(posts.account == r.account) & (posts.timestamp < r.timestamp)]["engagement_rate"].dropna()
        assert r.acct_n_past_posts == len(prior)
        if len(prior):
            assert np.isclose(r.acct_past_median_er, prior.median())
        else:
            assert np.isnan(r.acct_past_median_er)


def test_jury_factor_columns_allowed_but_not_lookalikes(posts, cfg):
    X = _feature_frame(posts)
    for k in cfg["jury"]["factors"]:
        X[f"jury_{k}_mean"] = 5.0
        X[f"jury_{k}_std"] = 1.0
    check_no_leakage(X, cfg)
    X["jury_comment_count"] = 3
    with pytest.raises(LeakageError):
        check_no_leakage(X, cfg)


def test_all_history_features_match_strictly_earlier_reference(posts):
    from viralsense.data.prepare import account_history

    out = account_history(posts)
    for _, r in out.sample(150, random_state=1).iterrows():
        acc = posts[posts.account == r.account]
        prior = acc[acc.timestamp < r.timestamp].sort_values("timestamp", kind="stable")
        er = prior["engagement_rate"].dropna().to_numpy()
        assert r.acct_posts_last_30d == ((prior.timestamp >= r.timestamp - pd.Timedelta(days=30))).sum()
        if len(prior):
            assert np.isclose(r.acct_days_since_last, (r.timestamp - prior.timestamp.max()) / pd.Timedelta(days=1))
        else:
            assert np.isnan(r.acct_days_since_last)
        if len(er):
            assert np.isclose(r.acct_recent_median_er, np.median(er[-10:]))
            q1, q3 = np.percentile(er, [25, 75])
            assert np.isclose(r.acct_past_er_iqr, q3 - q1)
            assert np.isclose(r.acct_er_trend, np.log((np.median(er[-10:]) + 1e-6) / (np.median(er) + 1e-6)))
        else:
            assert np.isnan(r.acct_recent_median_er) and np.isnan(r.acct_er_trend)


def test_history_features_ignore_current_post(posts):
    from viralsense.data.prepare import HISTORY_COLS, account_history

    base = account_history(posts)
    bumped = posts.copy()
    i = bumped.index[5]
    bumped.loc[i, "engagement_rate"] *= 1000  # changing a post's own outcome must not change its own features
    np.testing.assert_array_equal(base.loc[i, HISTORY_COLS].astype(float).fillna(-1),
                                  account_history(bumped).loc[i, HISTORY_COLS].astype(float).fillna(-1))
