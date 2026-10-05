"""Significance checks on saved results (no refitting).

    python -m viralsense.models.compare

1. Does each classifier beat the account-history-only baseline? Fold-paired one-sided t-test.
2. Does LinUCB beat the random policy in replay? Share of random seeds that do at least as well.
"""
from pathlib import Path

import pandas as pd
from scipy.stats import ttest_rel

from viralsense.utils import ensure_dir, load_config

BASE = "xgb [history only]|class_weight"


def main(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    tables = ensure_dir(Path(cfg["paths"]["reports"]) / "tables")
    f = pd.read_csv(tables / "classification_cv_folds.csv")
    rows = []
    for metric in ("macro_f1", "pr_auc_viral", "balanced_acc"):
        piv = f.assign(k=f.model + "|" + f.imbalance).pivot(index="fold", columns="k", values=metric)
        for k in piv.columns:
            if k == BASE:
                continue
            d = piv[k] - piv[BASE]
            rows.append({"comparison": f"{k} vs history only", "metric": metric, "mean_gain": d.mean(), "std_gain": d.std(),
                         "folds_better": int((d > 0).sum()), "n_folds": len(d),
                         "p_one_sided": ttest_rel(piv[k], piv[BASE], alternative="greater").pvalue})
    sig = pd.DataFrame(rows)
    sig["significant_p05"] = sig.p_one_sided < 0.05
    sig.to_csv(tables / "classification_vs_history_baseline.csv", index=False)

    b = pd.read_csv(tables / "bandit_replay_runs.csv")
    lin = b.loc[b.policy == "linucb", "mean_reward"].mean()
    rnd = b.loc[b.policy == "random", "mean_reward"]
    bandit = pd.DataFrame([{"linucb_mean_reward": lin, "random_mean": rnd.mean(), "random_std": rnd.std(),
                            "random_seeds_at_least_linucb": float((rnd >= lin).mean()), "n_random_seeds": len(rnd)}])
    bandit.to_csv(tables / "bandit_vs_random.csv", index=False)
    print(sig[sig.metric == "macro_f1"].round(3).to_string(index=False))
    print(bandit.round(4).to_string(index=False))
    return sig


if __name__ == "__main__":
    main()
