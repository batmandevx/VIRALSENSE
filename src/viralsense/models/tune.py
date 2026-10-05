"""Hyperparameter tuning with Optuna (TPE), on training accounts only.

    python -m viralsense.models.tune [--trials 25]

Objective: mean macro-F1 over 3 account-grouped folds of the training accounts, class-weighted.
The held-out test accounts are never touched. Preprocessing (impute, scale, PCA) is fit once per fold
and reused across trials, so each trial only refits the classifier. Tuned parameters are then used by
classify.py as the 'xgb_tuned' and 'lgbm_tuned' models and evaluated with the usual 5-fold CV + test.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.utils.class_weight import compute_sample_weight

from viralsense.features.build import make_preprocessor
from viralsense.models.common import feature_columns, load_table
from viralsense.utils import ensure_dir, load_config, seed_everything


def space(trial: optuna.Trial, name: str) -> dict:
    if name == "xgb":
        return {"n_estimators": trial.suggest_int("n_estimators", 150, 700, step=50),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
                "min_child_weight": trial.suggest_float("min_child_weight", 1, 20, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True)}
    return {"n_estimators": trial.suggest_int("n_estimators", 150, 700, step=50),
            "num_leaves": trial.suggest_int("num_leaves", 8, 96, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0), "subsample_freq": 1,
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True)}


def estimator(name: str, params: dict, seed: int):
    if name == "xgb":
        from xgboost import XGBClassifier

        return XGBClassifier(**params, objective="multi:softprob", eval_metric="mlogloss", tree_method="hist",
                             random_state=seed, n_jobs=-1)
    from lightgbm import LGBMClassifier

    return LGBMClassifier(**params, random_state=seed, n_jobs=-1, verbose=-1)


def main(cfg: dict | None = None, trials: int = 25) -> dict:
    cfg = cfg or load_config()
    seed = cfg["seed"]
    seed_everything(seed)
    tables, art = ensure_dir(Path(cfg["paths"]["reports"]) / "tables"), ensure_dir(cfg["paths"]["artifacts"])
    df = load_table(cfg)
    tr = df[df.split == "train"].reset_index(drop=True)
    cols = feature_columns(df, cfg)
    folds = []
    for a, b in GroupKFold(3).split(tr, groups=tr["account"]):
        pre = make_preprocessor(cols, cfg["features"]["pca_dims"], seed).fit(tr.loc[a, cols])
        folds.append((pre.transform(tr.loc[a, cols]), tr.loc[a, "y"].to_numpy(),
                      pre.transform(tr.loc[b, cols]), tr.loc[b, "y"].to_numpy()))

    best, history = {}, []
    for name in ("xgb", "lgbm"):
        def objective(trial):
            params = space(trial, name)
            scores = []
            for Xa, ya, Xb, yb in folds:
                m = estimator(name, params, seed).fit(Xa, ya, sample_weight=compute_sample_weight("balanced", ya))
                scores.append(f1_score(yb, m.predict(Xb), average="macro"))
            trial.set_user_attr("std", float(np.std(scores)))
            return float(np.mean(scores))

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed),
                                    study_name=f"{name}_tuning")
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(objective, n_trials=trials, show_progress_bar=False,
                       callbacks=[lambda s, t: print(f"  {name} trial {t.number:2d}: macro-F1 {t.value:.4f} "
                                                     f"(best {s.best_value:.4f})", flush=True)])
        best[name] = study.best_params | ({"subsample_freq": 1} if name == "lgbm" else {})
        for t in study.trials:
            history.append({"model": name, "trial": t.number, "macro_f1": t.value, "std": t.user_attrs.get("std"),
                            **{f"param_{k}": v for k, v in t.params.items()}})
        imp = optuna.importance.get_param_importances(study)
        pd.Series(imp, name="importance").rename_axis("param").reset_index().assign(model=name).to_csv(
            tables / f"tuning_param_importance_{name}.csv", index=False)
    pd.DataFrame(history).to_csv(tables / "tuning_history.csv", index=False)
    (art / "tuned_params.json").write_text(json.dumps(best, indent=2))
    print(json.dumps(best, indent=2))
    return best


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=25)
    main(trials=ap.parse_args().trials)
