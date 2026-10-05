"""Phase 3b: XGBoost regression on log10 engagement rate, scored by Spearman correlation.

    python -m viralsense.models.regress
"""
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from viralsense.features.build import make_preprocessor
from viralsense.models.common import cv_splits, feature_columns, load_table
from viralsense.plots import MUTED, SERIES, save
from viralsense.utils import ensure_dir, load_config, seed_everything

EPS = 1e-6


def log_er(er) -> np.ndarray:
    return np.log10(np.asarray(er, dtype=float) + EPS)


def make_regressor(cols, cfg) -> Pipeline:
    return Pipeline([
        ("pre", make_preprocessor(cols, cfg["features"]["pca_dims"], cfg["seed"])),
        ("reg", XGBRegressor(**cfg["models"]["xgb"], tree_method="hist", random_state=cfg["seed"],
                             n_jobs=cfg["models"]["n_jobs"])),
    ])


def _rho(a, b) -> float:
    ok = ~(np.isnan(a) | np.isnan(b))
    return float(spearmanr(a[ok], b[ok]).statistic)


def main(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    seed_everything(cfg["seed"])
    rep = Path(cfg["paths"]["reports"])
    tables, figs, art = ensure_dir(rep / "tables"), ensure_dir(rep / "figures"), ensure_dir(cfg["paths"]["artifacts"])
    df = load_table(cfg)
    cols = feature_columns(df, cfg)

    rows = []
    for f, tr, va in cv_splits(df):
        m = make_regressor(cols, cfg).fit(tr[cols], log_er(tr["engagement_rate"]))
        y = log_er(va["engagement_rate"])
        rows.append({"fold": f, "spearman_model": _rho(m.predict(va[cols]), y),
                     # Baseline: the account's own past median alone (NaN rows dropped).
                     "spearman_past_median_only": _rho(va["acct_past_median_er"].to_numpy(), y)})
    cv = pd.DataFrame(rows)
    cv.to_csv(tables / "regression_cv_folds.csv", index=False)

    tr, te = df[df.split == "train"], df[df.split == "test"]
    m = make_regressor(cols, cfg).fit(tr[cols], log_er(tr["engagement_rate"]))
    pred, y = m.predict(te[cols]), log_er(te["engagement_rate"])
    result = {
        "cv_spearman_mean": cv["spearman_model"].mean(), "cv_spearman_std": cv["spearman_model"].std(),
        "cv_past_median_only_mean": cv["spearman_past_median_only"].mean(),
        "test_spearman": _rho(pred, y), "test_past_median_only": _rho(te["acct_past_median_er"].to_numpy(), y),
        "test_n": len(te),
    }
    pd.DataFrame([result]).to_csv(tables / "regression_summary.csv", index=False)
    joblib.dump({"model": m, "columns": cols, "target": "log10(engagement_rate + 1e-6)"}, art / "regressor.joblib")

    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.scatter(y, pred, s=8, alpha=0.35, color=SERIES[0], edgecolors="none")
    lo, hi = np.percentile(np.r_[y, pred], [0.5, 99.5])
    ax.plot([lo, hi], [lo, hi], color=MUTED, linewidth=1, linestyle="--")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("actual log10 engagement rate")
    ax.set_ylabel("predicted")
    ax.set_title(f"Test accounts: Spearman ρ = {result['test_spearman']:.3f}")
    save(fig, figs / "regression_test_scatter.png")
    print(json.dumps(result, indent=2, default=float))
    return result


if __name__ == "__main__":
    main()
