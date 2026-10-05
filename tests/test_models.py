import copy

import numpy as np
import pandas as pd
import pytest

from viralsense.models.common import fit_model, metrics


@pytest.fixture(scope="module")
def small_cfg(cfg):
    c = copy.deepcopy(cfg)
    c["models"]["xgb"]["n_estimators"] = 20
    c["models"]["rf"]["n_estimators"] = 20
    c["models"]["lgbm"]["n_estimators"] = 20
    c["models"]["hgb"]["max_iter"] = 20
    return c


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    n = 300
    X = pd.DataFrame({"follower_tier": rng.integers(0, 4, n), "acct_past_median_er": rng.random(n),
                      "img_brightness": rng.random(n), "cap_len": rng.integers(0, 300, n)})
    for i in range(40):
        X[f"clip_{i:03d}"] = rng.normal(size=n)
        X[f"txt_{i:03d}"] = rng.normal(size=n)
    X.loc[::7, "acct_past_median_er"] = np.nan
    y = rng.choice(3, n, p=[0.6, 0.3, 0.1])
    groups = np.repeat(np.arange(30), 10)
    return X, y, groups


@pytest.mark.parametrize("name", ["logreg", "rf", "xgb", "lgbm", "hgb", "vote", "stack", "dummy"])
@pytest.mark.parametrize("imb", ["class_weight", "smote"])
def test_every_model_fits_and_predicts(small_cfg, data, name, imb):
    X, y, g = data
    m = fit_model(name, imb, X.iloc[:200], y[:200], g[:200], small_cfg)
    p = m.predict_proba(X.iloc[200:])
    assert p.shape == (100, 3) and np.allclose(p.sum(1), 1)


def test_pca_is_fit_on_training_rows_only(small_cfg, data):
    X, y, g = data
    m = fit_model("logreg", "smote", X.iloc[:200], y[:200], g[:200], small_cfg)
    pca = m.named_steps["pre"].named_transformers_["clip"]
    assert pca.n_components_ == 32 and pca.n_samples_ == 200  # SMOTE runs after PCA, never before


def test_smote_is_training_only(small_cfg, data):
    X, y, g = data
    m = fit_model("xgb", "smote", X.iloc[:200], y[:200], g[:200], small_cfg)
    assert "smote" in m.named_steps
    assert len(m.predict(X.iloc[200:])) == 100  # imblearn skips samplers at predict time


def test_metrics_have_no_plain_accuracy():
    y = np.array([0, 0, 1, 2])
    p = np.eye(3)[[0, 1, 1, 2]]
    out = metrics(y, p)
    assert "accuracy" not in out
    assert {"macro_f1", "balanced_acc", "recall_Viral", "pr_auc_viral"} <= set(out)


def test_calibration_and_decision_rule():
    from viralsense.models.calibrate import apply_calibrators, ece, fit_calibrators, fit_decision_weights

    rng = np.random.default_rng(0)
    n = 3000
    y = rng.choice(3, n, p=[0.6, 0.3, 0.1])
    logits = np.eye(3)[y] * 1.2 + rng.normal(0, 1, (n, 3))
    P = np.exp(logits * 3) / np.exp(logits * 3).sum(1, keepdims=True)  # deliberately over-confident
    cals = fit_calibrators(P[:2000], y[:2000])
    Q = apply_calibrators(cals, P[2000:])
    assert np.allclose(Q.sum(1), 1)
    assert ece(Q[:, 2], (y[2000:] == 2).astype(float)) < ece(P[2000:, 2], (y[2000:] == 2).astype(float))
    from sklearn.metrics import f1_score

    w = fit_decision_weights(apply_calibrators(cals, P[:2000]), y[:2000])
    assert w[0] == 1.0 and len(w) == 3
    base = f1_score(y[:2000], apply_calibrators(cals, P[:2000]).argmax(1), average="macro")
    tuned = f1_score(y[:2000], (apply_calibrators(cals, P[:2000]) * w).argmax(1), average="macro")
    assert tuned >= base


def test_new_post_frame_supplies_every_model_input(cfg):
    """Regression test: a feature the model was trained on must exist for a brand-new post."""
    from datetime import datetime, timezone

    from viralsense.data.prepare import HISTORY_COLS
    from viralsense.features.metadata import metadata_features
    from viralsense.inference import post_frame

    spec = {"tier_edges": [5000, 15000, 50000], "cuts": {}}
    pf = post_frame("x.jpg", ["hello"], 20000, datetime(2024, 1, 1, tzinfo=timezone.utc), spec, 0.03, 120)
    meta = metadata_features(pf)
    for c in HISTORY_COLS + ["follower_tier", "meta_log_followers", "meta_hour", "meta_weekday"]:
        assert c in meta.columns, c
    assert meta.acct_recent_median_er.iloc[0] == 0.03 and meta.acct_er_trend.iloc[0] == 0.0
    assert np.isnan(meta.acct_days_since_last.iloc[0])  # unknown -> imputed by the pipeline


def test_lightgbm_predicts_after_torch_loaded(tmp_path):
    """Regression test for the macOS segfault: LightGBM prediction in a process that has loaded torch."""
    import subprocess
    import sys

    script = f"""
import viralsense, joblib, numpy as np, torch
from lightgbm import LGBMClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from viralsense.inference import load_model
torch.ones(8).sum()
X = np.random.rand(400, 10); y = np.random.randint(0, 3, 400)
m = make_pipeline(StandardScaler(), LGBMClassifier(n_estimators=30, verbose=-1, n_jobs=1)).fit(X, y)
m.set_params(lgbmclassifier__n_jobs=-1)  # saved multi-threaded, like the deployed model
joblib.dump({{"model": m}}, r"{tmp_path}/m.joblib")
b = load_model(r"{tmp_path}/m.joblib")
print(b["model"].predict_proba(X[:5]).shape)
"""
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    assert "(5, 3)" in out.stdout
