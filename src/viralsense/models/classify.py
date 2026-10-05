"""Phase 3a: Low / Moderate / Viral classification.

    python -m viralsense.models.classify

Grid: {logreg, rf, xgb, stack} x {class_weight, smote}, 5 account-grouped CV folds.
The configuration with the best mean CV macro-F1 is refit on all training accounts and
evaluated once on the held-out test accounts. The deployed model (dashboard) is XGBoost
with the better imbalance strategy, because TreeSHAP needs a tree model.
"""
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_curve

from viralsense.models.common import (CLASSES, IMBALANCE, MODELS, available_models, cv_splits, feature_columns, fit_model, load_table,
                                      metrics, summarize)
from viralsense.models.explain import importance_report
from viralsense.plots import MUTED, SERIES, save
from viralsense.utils import ensure_dir, load_config, seed_everything


def history_split(cols: list[str]) -> tuple[list[str], list[str]]:
    """History-only baseline = every account-history signal plus account size; no-history = everything else."""
    hist = [c for c in cols if c.startswith("acct_")] + [c for c in ("follower_tier", "meta_log_followers") if c in cols]
    return hist, [c for c in cols if not c.startswith("acct_")]


class TooGoodError(RuntimeError):
    pass


def cross_validate(df, cols, cfg, models=MODELS, imbalances=IMBALANCE) -> pd.DataFrame:
    rows = []
    for name in models:
        for imb in imbalances:
            for f, tr, va in cv_splits(df):
                m = fit_model(name, imb, tr[cols], tr["y"].to_numpy(), tr["account"].to_numpy(), cfg)
                rows.append({"model": name, "imbalance": imb, "fold": f, **metrics(va["y"].to_numpy(), m.predict_proba(va[cols]))})
                print(f"  {name:6s} {imb:12s} fold {f}: macro-F1 {rows[-1]['macro_f1']:.3f}")
    return pd.DataFrame(rows)


def check_plausibility(row: dict, cfg: dict, where: str) -> None:
    m = cfg["models"]
    if row["macro_f1"] > m["too_good_macro_f1"] or row["pr_auc_viral"] > m["too_good_viral_pr_auc"]:
        raise TooGoodError(
            f"{where}: macro-F1 {row['macro_f1']:.3f}, Viral PR-AUC {row['pr_auc_viral']:.3f} exceed the "
            "plausibility thresholds in config.models. Stop and look for leakage before trusting these numbers.")


def plot_cv(summary: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 3.6))
    labels = summary["model"] + "\n" + summary["imbalance"].str.replace("class_weight", "weights")
    x = np.arange(len(summary))
    colors = [SERIES[0] if i == "class_weight" else SERIES[1] for i in summary["imbalance"]]
    ax.bar(x, summary["macro_f1_mean"], yerr=summary["macro_f1_std"], color=colors, width=0.6,
           error_kw={"ecolor": MUTED, "elinewidth": 1, "capsize": 3})
    ax.set_xticks(x, labels, fontsize=8)
    ax.set_ylabel("macro-F1 (mean ± std, 5 folds)")
    ax.set_title("Cross-validated macro-F1 by model and imbalance strategy")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=SERIES[0]), plt.Rectangle((0, 0), 1, 1, color=SERIES[1])],
              labels=["class weights", "SMOTE"], loc="upper left", ncols=2)
    ax.set_ylim(0, max(0.5, (summary["macro_f1_mean"] + summary["macro_f1_std"]).max() * 1.15))
    save(fig, path)


def plot_confusion(y, proba, title, path: Path) -> None:
    cm = confusion_matrix(y, proba.argmax(1), labels=[0, 1, 2], normalize="true")
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.grid(False)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", color="white" if cm[i, j] > 0.55 else "#0b0b0b")
    ax.set_xticks(range(3), CLASSES)
    ax.set_yticks(range(3), CLASSES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true (row-normalised recall)")
    ax.set_title(title)
    save(fig, path)


def plot_pr_viral(curves: dict, prevalence: float, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    for i, (name, (y, p)) in enumerate(curves.items()):
        prec, rec, _ = precision_recall_curve(y, p)
        ax.plot(rec, prec, color=SERIES[i % len(SERIES)], label=name)
    ax.axhline(prevalence, color=MUTED, linestyle="--", linewidth=1, label=f"chance ({prevalence:.2f})")
    ax.set_xlabel("recall (Viral)")
    ax.set_ylabel("precision (Viral)")
    ax.set_title("Viral precision-recall on held-out test accounts")
    ax.legend(fontsize=8)
    save(fig, path)


def main(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    seed_everything(cfg["seed"])
    rep = Path(cfg["paths"]["reports"])
    tables, figs = ensure_dir(rep / "tables"), ensure_dir(rep / "figures")
    art = ensure_dir(cfg["paths"]["artifacts"])

    df = load_table(cfg)
    cols = feature_columns(df, cfg)
    print(f"{len(df)} posts, {len(cols)} raw feature columns")

    models = available_models(cfg)
    cv = cross_validate(df, cols, cfg, models=models)
    # Baselines: how much comes from the account's own history vs. the post itself.
    hist, no_hist = history_split(cols)
    for tag, bcols in (("history only", hist), ("no history", no_hist)):
        b = cross_validate(df, bcols, cfg, models=["xgb"], imbalances=["class_weight"])
        cv = pd.concat([cv, b.assign(model=f"xgb [{tag}]")], ignore_index=True)
    dummy = cross_validate(df, cols, cfg, models=["dummy"], imbalances=["class_weight"])
    cv = pd.concat([cv, dummy.assign(imbalance="none")], ignore_index=True)
    cv.to_csv(tables / "classification_cv_folds.csv", index=False)
    summary = summarize(cv, ["model", "imbalance"]).sort_values("macro_f1_mean", ascending=False)
    summary.to_csv(tables / "classification_cv_summary.csv", index=False)
    main_grid = summary[summary.model.isin(models)]
    plot_cv(main_grid.sort_values(["model", "imbalance"]), figs / "classification_cv_macro_f1.png")
    best = main_grid.iloc[0]
    check_plausibility({"macro_f1": best["macro_f1_mean"], "pr_auc_viral": best["pr_auc_viral_mean"]}, cfg, "CV")

    tr, te = df[df.split == "train"], df[df.split == "test"]
    test_rows, curves, fitted, preds = [], {}, {}, []
    for name in models:
        imb = main_grid[main_grid.model == name].iloc[0]["imbalance"]  # best imbalance per model, chosen on CV
        m = fit_model(name, imb, tr[cols], tr["y"].to_numpy(), tr["account"].to_numpy(), cfg)
        proba = m.predict_proba(te[cols])
        fitted[name] = (m, imb)
        test_rows.append({"model": name, "imbalance": imb,
                          "selected_by_cv": name == best["model"] and imb == best["imbalance"],
                          **metrics(te["y"].to_numpy(), proba)})
        curves[f"{name} ({imb})"] = (te["y"].to_numpy() == 2, proba[:, 2])
        preds.append(pd.DataFrame({"model": name, "imbalance": imb, "post_id": te["post_id"].to_numpy(),
                                   "y": te["y"].to_numpy(), **{f"p_{c}": proba[:, i] for i, c in enumerate(CLASSES)}}))
        if name == best["model"]:
            plot_confusion(te["y"].to_numpy(), proba, f"Test confusion: {name} ({imb})", figs / "confusion_test.png")
    m = fit_model("dummy", "class_weight", tr[cols], tr["y"].to_numpy(), tr["account"].to_numpy(), cfg)
    test_rows.append({"model": "dummy", "imbalance": "none", "selected_by_cv": False,
                      **metrics(te["y"].to_numpy(), m.predict_proba(te[cols]))})
    for tag, bcols in (("history only", hist), ("no history", no_hist)):
        m = fit_model("xgb", "class_weight", tr[bcols], tr["y"].to_numpy(), tr["account"].to_numpy(), cfg)
        test_rows.append({"model": f"xgb [{tag}]", "imbalance": "class_weight", "selected_by_cv": False,
                          **metrics(te["y"].to_numpy(), m.predict_proba(te[bcols]))})
    pd.concat(preds).to_csv(tables / "classification_test_predictions.csv", index=False)
    test = pd.DataFrame(test_rows)
    test.to_csv(tables / "classification_test.csv", index=False)
    plot_pr_viral(curves, float((te["y"] == 2).mean()), figs / "pr_curve_viral_test.png")
    check_plausibility(test[test.selected_by_cv].iloc[0].to_dict(), cfg, "test")

    # Deployed model: the best tree model by CV (TreeSHAP needs a single tree ensemble).
    tree = main_grid[main_grid.model.isin(["xgb", "xgb_tuned", "lgbm", "lgbm_tuned"])].iloc[0]["model"]
    deployed, imb = fitted[tree]
    bundle = {"model": deployed, "columns": cols, "imbalance": imb, "classes": CLASSES}
    joblib.dump(bundle, art / "classifier.joblib")
    importance_report(bundle, te, tables, figs)
    result = {"best_cv": best[["model", "imbalance", "macro_f1_mean", "macro_f1_std"]].to_dict(),
              "test_selected": test[test.selected_by_cv].iloc[0].to_dict(), "deployed": f"{tree} ({imb})"}
    (tables / "classification_selected.json").write_text(json.dumps(result, indent=2, default=float))
    print(summary.round(3).to_string(index=False))
    print(test.round(3).to_string(index=False))
    return result


if __name__ == "__main__":
    main()
