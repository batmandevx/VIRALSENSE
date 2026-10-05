"""Jury analytics from the cached ratings: who is harsh, who agrees with whom, whose taste tracks real engagement.

    python -m viralsense.jury.report

Correlations with engagement use training accounts only.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from viralsense.jury.run import load_personas
from viralsense.utils import ensure_dir, load_config


def load_ratings(cfg: dict) -> pd.DataFrame:
    cache = Path(cfg["paths"]["cache"]) / "jury" / cfg["jury"]["backend"]
    rows = []
    for f in cache.glob("*/*.json"):
        d = json.loads(f.read_text())
        rows.append({"post_id": str(d["post_id"]), "persona_id": d["persona_id"], **d["scores"]})
    return pd.DataFrame(rows)


def main(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    tables = ensure_dir(Path(cfg["paths"]["reports"]) / "tables")
    F = cfg["jury"]["factors"]
    r = load_ratings(cfg)
    if r.empty:
        print("no cached jury ratings yet")
        return r
    posts = pd.read_parquet(Path(cfg["paths"]["processed"]) / "posts.parquet",
                            columns=["post_id", "split", "label", "engagement_rate"])
    r = r.merge(posts, on="post_id")
    r["overall"] = r[F].mean(axis=1)
    r["y"] = r["label"].map({"Low": 0, "Moderate": 1, "Viral": 2})
    r["log_er"] = np.log10(r["engagement_rate"] + 1e-6)
    names = {p["id"]: p["name"] for p in load_personas(cfg)}

    tr = r[r.split == "train"]
    stats = r.groupby("persona_id")[F + ["overall"]].mean()
    stats["n_ratings"] = r.groupby("persona_id").size()
    stats["spread"] = r.groupby("persona_id")["overall"].std()
    stats["spearman_vs_engagement"] = tr.groupby("persona_id").apply(
        lambda g: spearmanr(g["overall"], g["log_er"]).statistic, include_groups=False)
    stats["spearman_vs_label"] = tr.groupby("persona_id").apply(
        lambda g: spearmanr(g["overall"], g["y"]).statistic, include_groups=False)
    stats["name"] = stats.index.map(names)
    stats = stats.sort_values("overall")
    stats.to_csv(tables / "jury_persona_stats.csv")

    wide = r.pivot_table(index="post_id", columns="persona_id", values="overall")
    agree = wide.corr(method="spearman")
    agree.index, agree.columns = [names[i] for i in agree.index], [names[i] for i in agree.columns]
    agree.to_csv(tables / "jury_persona_agreement.csv")

    per_post = r.groupby("post_id").agg(**{f: (f, "mean") for f in F}, overall=("overall", "mean"),
                                        disagreement=("overall", "std"), label=("label", "first"), split=("split", "first"))
    per_post[F + ["overall"]].corr(method="spearman").to_csv(tables / "jury_factor_correlation.csv")
    per_post.groupby("label")["disagreement"].describe().to_csv(tables / "jury_disagreement_by_label.csv")
    pd.DataFrame([{"posts_rated": wide.shape[0], "complete_posts": int((wide.notna().sum(1) == len(names)).sum()),
                   "ratings": len(r)}]).to_csv(tables / "jury_progress.csv", index=False)
    print(stats[["name", "overall", "spread", "spearman_vs_engagement", "n_ratings"]].round(3).to_string())
    return stats


if __name__ == "__main__":
    main()
