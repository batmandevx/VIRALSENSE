"""Phase 3d: does the persona jury add anything? Ablation on the jury subset.

    python -m viralsense.models.ablation

(a) metadata only, (b) metadata + content, (c) metadata + content + jury.
Same model, same folds (training accounts of the subset), mean and std across folds.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ttest_rel

from viralsense.models.common import CLASSES, cv_splits, feature_columns, fit_model, load_table, metrics, summarize
from viralsense.utils import ensure_dir, load_config, seed_everything

SETS = {
    "a_metadata": ("metadata",),
    "b_metadata+content": ("metadata", "content_scalar", "clip", "txt"),
    "c_metadata+content+jury": ("metadata", "content_scalar", "clip", "txt", "jury"),
}
ABLATION_MODELS = [("logreg", "class_weight"), ("xgb", "class_weight")]


def verdict(folds: pd.DataFrame, model: str, metric: str, min_gain: float = 0.01, alpha: float = 0.05) -> str:
    """'Helps' only if the fold-paired gain is both significant (one-sided t-test) and >= min_gain."""
    piv = folds[folds.model == model].pivot(index="fold", columns="feature_set", values=metric)
    d = piv["c_metadata+content+jury"] - piv["b_metadata+content"]
    p = ttest_rel(piv["c_metadata+content+jury"], piv["b_metadata+content"], alternative="greater").pvalue
    helps = d.mean() >= min_gain and p < alpha
    return (f"{model}, {metric}: jury minus no-jury = {d.mean():+.3f} ± {d.std():.3f} (std over {len(d)} folds), "
            f"better in {int((d > 0).sum())}/{len(d)} folds, one-sided paired t-test p = {p:.3f} -> "
            f"{'the jury HELPS' if helps else 'the jury does NOT reliably help'} "
            f"(criterion: gain >= {min_gain} and p < {alpha})")


def main(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    seed_everything(cfg["seed"])
    tables = ensure_dir(Path(cfg["paths"]["reports"]) / "tables")
    import copy

    cfg = copy.deepcopy(cfg)
    cfg["features"]["pca_dims"] = cfg["models"].get("ablation_pca_dims", cfg["features"]["pca_dims"])
    df = load_table(cfg, with_jury=True)
    print(f"embedding PCA dims for the ablation: {cfg['features']['pca_dims']}")
    print(f"jury subset: {len(df)} posts ({(df.split == 'train').sum()} train used for CV)")

    rows = []
    for set_name, groups in SETS.items():
        cols = feature_columns(df, cfg, groups)
        for name, imb in ABLATION_MODELS:
            for f, tr, va in cv_splits(df):
                m = fit_model(name, imb, tr[cols], tr["y"].to_numpy(), tr["account"].to_numpy(), cfg)
                rows.append({"feature_set": set_name, "model": name, "fold": f, "n_features": len(cols),
                             **metrics(va["y"].to_numpy(), m.predict_proba(va[cols]))})
    folds = pd.DataFrame(rows)
    folds.to_csv(tables / "ablation_folds.csv", index=False)
    summary = summarize(folds.drop(columns="n_features"), ["feature_set", "model"])
    summary.to_csv(tables / "ablation_summary.csv", index=False)

    jcols = [c for c in df.columns if c.startswith("jury_") and c.endswith("_mean")] + ["jury_disagreement"]
    trj = df[df.split == "train"]
    trj.groupby("label")[jcols].mean().reindex(CLASSES).to_csv(tables / "jury_scores_by_label.csv")
    pd.DataFrame({"feature": jcols, "spearman_vs_log_er": [
        spearmanr(trj[c], np.log10(trj["engagement_rate"] + 1e-6)).statistic for c in jcols]}).to_csv(
        tables / "jury_spearman_vs_engagement.csv", index=False)

    lines = [verdict(folds, m, metric) for m, _ in ABLATION_MODELS for metric in ("macro_f1", "pr_auc_viral")]
    (tables / "ablation_verdict.txt").write_text("\n".join(lines) + "\n")
    cols = ["feature_set", "model", "macro_f1_mean", "macro_f1_std", "balanced_acc_mean", "pr_auc_viral_mean", "pr_auc_viral_std"]
    print(summary[cols].round(3).to_string(index=False))
    print("\n".join(lines))
    return summary


if __name__ == "__main__":
    main()
