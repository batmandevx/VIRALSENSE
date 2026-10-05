"""Phase 3c: K-Means on content features (image + caption), k chosen by silhouette.

    python -m viralsense.models.cluster
"""
import re
from collections import Counter
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from viralsense.features.build import make_preprocessor
from viralsense.models.common import feature_columns, load_table
from viralsense.plots import SERIES, save
from viralsense.utils import ensure_dir, load_config, seed_everything

CONTENT = ("content_scalar", "clip", "txt")
STOP = set("the a an and or to of in on for with is it this that my your i you me we our at be are was so just".split())


def top_terms(captions: pd.Series, n: int = 8) -> tuple[str, str]:
    tags, words = Counter(), Counter()
    for c in captions.fillna(""):
        low = c.lower()
        tags.update(re.findall(r"#\w+", low))
        words.update(w for w in re.findall(r"[a-z]{3,}", re.sub(r"#\w+", " ", low)) if w not in STOP)
    return ", ".join(t for t, _ in tags.most_common(n)), ", ".join(w for w, _ in words.most_common(n))


def main(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    seed = cfg["seed"]
    seed_everything(seed)
    rep = Path(cfg["paths"]["reports"])
    tables, figs = ensure_dir(rep / "tables"), ensure_dir(rep / "figures")
    df = load_table(cfg)
    posts = pd.read_parquet(Path(cfg["paths"]["processed"]) / "posts.parquet")[["post_id", "caption"]]
    df = df.merge(posts, on="post_id")
    cols = feature_columns(df, cfg, CONTENT)
    tr = df[df.split == "train"]

    pre = make_preprocessor(cols, cfg["features"]["pca_dims"], seed).fit(tr[cols])
    Z = pre.transform(tr[cols])
    rng = np.random.default_rng(seed)
    sample = rng.choice(len(Z), min(5000, len(Z)), replace=False)
    sil = []
    for k in cfg["models"]["kmeans_k"]:
        if k >= len(Z):
            continue
        km = KMeans(k, n_init=10, random_state=seed).fit(Z)
        sil.append({"k": k, "silhouette": silhouette_score(Z[sample], km.labels_[sample]), "inertia": km.inertia_})
    sil = pd.DataFrame(sil)
    sil.to_csv(tables / "kmeans_silhouette.csv", index=False)
    k = int(sil.loc[sil.silhouette.idxmax(), "k"])
    km = KMeans(k, n_init=10, random_state=seed).fit(Z)
    joblib.dump({"pre": pre, "kmeans": km, "columns": cols}, ensure_dir(cfg["paths"]["artifacts"]) / "clusters.joblib")

    fig, ax = plt.subplots(figsize=(5, 3.4))
    ax.plot(sil.k, sil.silhouette, marker="o", markersize=6, color=SERIES[0])
    ax.scatter([k], [sil.silhouette.max()], s=90, facecolors="none", edgecolors=SERIES[1], linewidths=2, zorder=3)
    ax.set_xlabel("k")
    ax.set_ylabel("silhouette (train, ≤5k sample)")
    ax.set_title(f"K-Means on content features: k = {k} chosen")
    save(fig, figs / "kmeans_silhouette.png")

    tr = tr.assign(cluster=km.labels_)
    from sklearn.decomposition import PCA
    xy = PCA(2, random_state=seed).fit_transform(Z)
    pts = rng.choice(len(Z), min(2500, len(Z)), replace=False)
    pd.DataFrame({"x": xy[pts, 0], "y": xy[pts, 1], "cluster": km.labels_[pts],
                  "label": tr["label"].to_numpy()[pts]}).to_csv(tables / "kmeans_points.csv", index=False)
    scal = [c for c in cols if c.startswith(("img_", "cap_"))]
    prof = []
    for c, g in tr.groupby("cluster"):
        tags, words = top_terms(g["caption"])
        prof.append({"cluster": c, "n_posts": len(g), "n_accounts": g["account"].nunique(),
                     **{f"share_{l}": (g["label"] == l).mean() for l in ("Low", "Moderate", "Viral")},
                     "median_engagement_rate": g["engagement_rate"].median(),
                     **{f"mean_{s}": g[s].mean() for s in scal}, "top_hashtags": tags, "top_words": words})
    prof = pd.DataFrame(prof)
    overall = {f"mean_{s}": tr[s].mean() for s in scal}
    prof["description"] = prof.apply(lambda r: describe(r, overall), axis=1)
    prof.to_csv(tables / "kmeans_cluster_profiles.csv", index=False)
    print(sil.round(3).to_string(index=False))
    print(prof[["cluster", "n_posts", "share_Viral", "description", "top_hashtags"]].to_string(index=False))
    return prof


def describe(r: pd.Series, overall: dict) -> str:
    """Name the scalar traits that differ most from the overall mean (relative difference > 20%)."""
    names = {"img_brightness": "brightness", "img_contrast": "contrast", "img_colourfulness": "colourfulness",
             "img_face_count": "faces", "cap_len": "caption length", "cap_emoji_count": "emoji",
             "cap_hashtag_count": "hashtags", "cap_has_question": "questions", "cap_sentiment": "sentiment"}
    diffs = []
    for k, label in names.items():
        o, v = overall.get(f"mean_{k}"), r.get(f"mean_{k}")
        if o is None or v is None or abs(o) < 1e-9:
            continue
        rel = (v - o) / abs(o)
        if abs(rel) > 0.2:
            diffs.append((abs(rel), f"{'higher' if rel > 0 else 'lower'} {label}"))
    traits = [d for _, d in sorted(diffs, reverse=True)[:3]] or ["close to average on all scalar traits"]
    return "; ".join(traits)


if __name__ == "__main__":
    main()
