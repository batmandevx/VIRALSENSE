"""Shared pieces for Phase 3: data loading, estimators, pipelines, metrics."""
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, recall_score
from sklearn.model_selection import GroupKFold
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from viralsense.features.build import check_no_leakage, make_preprocessor, select

CLASSES = ["Low", "Moderate", "Viral"]
MODELS = ["logreg", "rf", "xgb", "lgbm", "hgb", "vote", "stack"]
BASELINES = ["dummy"]
ENSEMBLE_BASES = ("logreg", "rf", "xgb", "lgbm")  # stratified random guess: the floor every model must beat
IMBALANCE = ["class_weight", "smote"]


def load_table(cfg: dict, with_jury: bool = False) -> pd.DataFrame:
    proc = Path(cfg["paths"]["processed"])
    df = pd.read_parquet(proc / "features.parquet")
    if with_jury:
        df = df.merge(pd.read_parquet(proc / "jury_features.parquet"), on="post_id", how="inner")
    df["y"] = df["label"].map({c: i for i, c in enumerate(CLASSES)}).astype(int)
    return df


def feature_columns(df: pd.DataFrame, cfg: dict, groups=("metadata", "content_scalar", "clip", "txt")) -> list[str]:
    cols = select(df.columns, *groups)
    check_no_leakage(df[cols], cfg)
    return cols


def tuned_params(cfg: dict) -> dict | None:
    import json

    p = Path(cfg["paths"]["artifacts"]) / "tuned_params.json"
    return json.loads(p.read_text()) if p.exists() else None


def available_models(cfg: dict) -> list[str]:
    tp = tuned_params(cfg) or {}
    return MODELS + [f"{k}_tuned" for k in ("xgb", "lgbm") if k in tp]


def base_estimator(name: str, cfg: dict):
    m, seed, nj = cfg["models"], cfg["seed"], cfg["models"]["n_jobs"]
    if name.endswith("_tuned"):  # Optuna-tuned on training accounts only (models/tune.py)
        from viralsense.models.tune import estimator

        return estimator(name.removesuffix("_tuned"), tuned_params(cfg)[name.removesuffix("_tuned")], seed)
    if name == "logreg":
        return LogisticRegression(**m["lr"], random_state=seed)
    if name == "rf":
        return RandomForestClassifier(**m["rf"], random_state=seed, n_jobs=nj)
    if name == "xgb":
        return XGBClassifier(**m["xgb"], objective="multi:softprob", eval_metric="mlogloss",
                             tree_method="hist", random_state=seed, n_jobs=nj)
    if name == "lgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(**m["lgbm"], random_state=seed, n_jobs=nj, verbose=-1)
    if name == "hgb":
        return HistGradientBoostingClassifier(**m["hgb"], random_state=seed)
    if name == "dummy":
        return DummyClassifier(strategy="stratified", random_state=seed)
    raise ValueError(name)


def _smote(cfg):
    return SMOTE(random_state=cfg["seed"], k_neighbors=5)


def make_model(name: str, imbalance: str, columns: list[str], cfg: dict, inner_cv=None):
    """Preprocess (impute, scale, PCA fit on the training data only) -> [SMOTE] -> classifier.

    For stacking, SMOTE sits inside each base learner so the meta-learner's
    out-of-fold predictions are made on real rows, split by account (``inner_cv``).
    """
    pre = make_preprocessor(columns, cfg["features"]["pca_dims"], cfg["seed"])
    if name in ("stack", "vote"):
        bases = [(n, ImbPipeline([("smote", _smote(cfg)), ("clf", base_estimator(n, cfg))]) if imbalance == "smote"
                  else base_estimator(n, cfg)) for n in ENSEMBLE_BASES]
        if name == "vote":  # soft voting: average the base learners' probabilities
            return ImbPipeline([("pre", pre), ("clf", VotingClassifier(bases, voting="soft", n_jobs=1))])
        # With SMOTE the meta-learner still sees the original class mix, so it is class-weighted;
        # in class_weight mode it receives balanced sample weights through fit instead.
        meta = LogisticRegression(max_iter=2000, class_weight="balanced" if imbalance == "smote" else None)
        clf = StackingClassifier(bases, final_estimator=meta, cv=inner_cv,
                                 stack_method="predict_proba", n_jobs=1)
        return ImbPipeline([("pre", pre), ("clf", clf)])
    steps = [("pre", pre)]
    if imbalance == "smote" and name != "dummy":
        steps.append(("smote", _smote(cfg)))
    steps.append(("clf", base_estimator(name, cfg)))
    return ImbPipeline(steps)


def fit_model(name: str, imbalance: str, X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, cfg: dict):
    inner = list(GroupKFold(n_splits=3).split(X, y, groups)) if name == "stack" else None
    model = make_model(name, imbalance, list(X.columns), cfg, inner_cv=inner)
    params = {}
    if imbalance == "class_weight":
        # StackingClassifier forwards sample_weight to its base learners and its meta-learner.
        params["clf__sample_weight"] = compute_sample_weight("balanced", y)
    return model.fit(X, y, **params)


def metrics(y: np.ndarray, proba: np.ndarray) -> dict:
    pred = proba.argmax(1)
    rec = recall_score(y, pred, labels=[0, 1, 2], average=None, zero_division=0)
    return {
        "macro_f1": f1_score(y, pred, average="macro", labels=[0, 1, 2], zero_division=0),
        "balanced_acc": balanced_accuracy_score(y, pred),
        **{f"recall_{c}": r for c, r in zip(CLASSES, rec)},
        "pr_auc_viral": average_precision_score(y == 2, proba[:, 2]),
        "viral_prevalence": float(np.mean(y == 2)),
    }


def cv_splits(df: pd.DataFrame):
    tr = df[df["split"] == "train"]
    for f in sorted(tr["fold"].unique()):
        yield int(f), tr[tr["fold"] != f], tr[tr["fold"] == f]


def summarize(rows: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    num = [c for c in rows.columns if c not in by + ["fold"]]
    agg = rows.groupby(by)[num].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    return agg.reset_index()
