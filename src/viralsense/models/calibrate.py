"""Probability calibration + macro-F1 decision rule for the deployed classifier.

    python -m viralsense.models.calibrate

Class weights / SMOTE distort probabilities, and argmax of probabilities is not the macro-F1-optimal
decision. Both fixes are learned from out-of-fold (OOF) predictions on the training accounts only:
  1. per-class isotonic regression on OOF probabilities, rows renormalised to sum to 1;
  2. per-class decision weights w (Low fixed at 1): predict argmax(w * p), w chosen to maximise OOF macro-F1.
The held-out test accounts are used once, for the final report.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import f1_score

from viralsense.models.common import CLASSES, cv_splits, feature_columns, fit_model, load_table, metrics
from viralsense.utils import ensure_dir, load_config, seed_everything


def fit_calibrators(P: np.ndarray, y: np.ndarray) -> list[IsotonicRegression]:
    return [IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(P[:, k], (y == k).astype(float))
            for k in range(P.shape[1])]


def apply_calibrators(cals, P: np.ndarray) -> np.ndarray:
    Q = np.column_stack([c.predict(P[:, k]) for k, c in enumerate(cals)]) + 1e-6
    return Q / Q.sum(1, keepdims=True)


def fit_decision_weights(P: np.ndarray, y: np.ndarray, grid=np.round(np.arange(0.5, 3.01, 0.1), 2)) -> np.ndarray:
    best, best_w = -1.0, np.ones(P.shape[1])
    for wm in grid:
        for wv in grid:
            w = np.array([1.0, wm, wv])
            f = f1_score(y, (P * w).argmax(1), average="macro")
            if f > best + 1e-9:
                best, best_w = f, w
    return best_w


def ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error for one class: weighted |mean prediction - observed rate| over equal-width bins."""
    b = np.clip((p * bins).astype(int), 0, bins - 1)
    return float(sum((b == i).mean() * abs(p[b == i].mean() - y[b == i].mean()) for i in range(bins) if (b == i).any()))


def calibrated_proba(bundle: dict, X: pd.DataFrame) -> np.ndarray:
    P = bundle["model"].predict_proba(X[bundle["columns"]])
    return apply_calibrators(bundle["calibrators"], P) if bundle.get("calibrators") else P


def predict_label(bundle: dict, P: np.ndarray) -> np.ndarray:
    w = np.asarray(bundle.get("decision_weights", np.ones(P.shape[1])))
    return (P * w).argmax(1)


def main(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    seed_everything(cfg["seed"])
    tables, art = ensure_dir(Path(cfg["paths"]["reports"]) / "tables"), Path(cfg["paths"]["artifacts"])
    bundle = joblib.load(art / "classifier.joblib")
    name, imb = json.loads((tables / "classification_selected.json").read_text())["deployed"].replace(")", "").split(" (")
    df = load_table(cfg)
    cols = feature_columns(df, cfg)
    assert cols == bundle["columns"], "deployed model was trained on different columns"

    oof_idx, oof_P = [], []
    for f, tr, va in cv_splits(df):
        m = fit_model(name, imb, tr[cols], tr["y"].to_numpy(), tr["account"].to_numpy(), cfg)
        oof_idx.append(va.index.to_numpy())
        oof_P.append(m.predict_proba(va[cols]))
        print(f"  OOF fold {f} done")
    idx = np.concatenate(oof_idx)
    P_oof, y_oof = np.vstack(oof_P), df.loc[idx, "y"].to_numpy()
    cals = fit_calibrators(P_oof, y_oof)
    w = fit_decision_weights(apply_calibrators(cals, P_oof), y_oof)

    te = df[df.split == "test"]
    y = te["y"].to_numpy()
    P_raw = bundle["model"].predict_proba(te[cols])
    P_cal = apply_calibrators(cals, P_raw)
    rows = []
    for label, P, pred in [("raw probabilities, argmax", P_raw, P_raw.argmax(1)),
                           ("calibrated, argmax", P_cal, P_cal.argmax(1)),
                           ("calibrated + macro-F1 decision rule", P_cal, (P_cal * w).argmax(1))]:
        m = metrics(y, P)
        m["macro_f1"] = f1_score(y, pred, average="macro")
        from sklearn.metrics import balanced_accuracy_score, recall_score

        m["balanced_acc"] = balanced_accuracy_score(y, pred)
        for c, r in zip(CLASSES, recall_score(y, pred, labels=[0, 1, 2], average=None, zero_division=0)):
            m[f"recall_{c}"] = r
        m["brier_multiclass"] = float(np.mean(np.sum((P - np.eye(3)[y]) ** 2, axis=1)))
        m["ece_viral"] = ece(P[:, 2], (y == 2).astype(float))
        rows.append({"variant": label, **m})
    out = pd.DataFrame(rows)
    out.to_csv(tables / "calibration_summary.csv", index=False)

    rel = []
    for label, P in (("raw", P_raw), ("calibrated", P_cal)):
        b = np.clip((P[:, 2] * 10).astype(int), 0, 9)
        for i in range(10):
            if (b == i).any():
                rel.append({"variant": label, "bin": i, "mean_pred": P[b == i, 2].mean(), "observed": (y[b == i] == 2).mean(),
                            "n": int((b == i).sum())})
    pd.DataFrame(rel).to_csv(tables / "calibration_reliability.csv", index=False)
    (tables / "decision_rule.json").write_text(json.dumps({"model": f"{name} ({imb})", "weights": dict(zip(CLASSES, w.tolist())),
                                                           "oof_posts": int(len(idx))}, indent=2))
    bundle.update({"calibrators": cals, "decision_weights": w.tolist()})
    joblib.dump(bundle, art / "classifier.joblib")

    best = out.iloc[-1]
    m = cfg["models"]
    if best.macro_f1 > m["too_good_macro_f1"] or best.pr_auc_viral > m["too_good_viral_pr_auc"]:
        raise RuntimeError("calibrated model exceeds plausibility thresholds: check for leakage")
    print(out[["variant", "macro_f1", "balanced_acc", "recall_Viral", "pr_auc_viral", "brier_multiclass", "ece_viral"]]
          .round(4).to_string(index=False))
    print("decision weights:", dict(zip(CLASSES, w.round(2).tolist())))
    return out


if __name__ == "__main__":
    main()
