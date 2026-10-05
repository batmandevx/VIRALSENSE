"""Page: browse real held-out posts with their true and predicted labels."""
import base64
import html
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from ui import charts
from ui.theme import CLASS_COLORS, demo_banner, empty, hero, section

CLASSES = ["Low", "Moderate", "Viral"]


@st.cache_data(show_spinner=False)
def _load(processed: str, tables: str):
    pp, tp = Path(processed) / "posts.parquet", Path(tables) / "classification_test_predictions.csv"
    if not pp.exists() or not tp.exists():
        return None
    posts = pd.read_parquet(pp, columns=["post_id", "account", "caption", "image_path", "followers", "timestamp",
                                         "engagement_rate", "follower_tier", "label", "split"])
    preds = pd.read_csv(tp, dtype={"post_id": str})
    test = pd.read_csv(Path(tables) / "classification_test.csv")
    sel = test[test.selected_by_cv].iloc[0].model
    preds = preds[preds.model == sel]
    df = posts.merge(preds[["post_id", "p_Low", "p_Moderate", "p_Viral"]], on="post_id")
    df["pred"] = np.array(CLASSES)[df[["p_Low", "p_Moderate", "p_Viral"]].to_numpy().argmax(1)]
    return df, sel


@st.cache_data(show_spinner=False, max_entries=512)
def _thumb(path: str) -> str:
    try:
        return "data:image/jpeg;base64," + base64.b64encode(Path(path).read_bytes()).decode()
    except OSError:
        return ""


def page(cfg: dict) -> None:
    hero("Explore predictions", "Real posts from accounts the model never saw in training. Each tile shows the true class "
         "(left) and the model's prediction (right).", "held-out test accounts")
    demo_banner(cfg)
    loaded = _load(cfg["paths"]["processed"], str(Path(cfg["paths"]["reports"]) / "tables"))
    if loaded is None:
        empty("test predictions", "python -m viralsense.pipeline --stages classify")
        return
    df, model = loaded

    f1, f2, f3, f4 = st.columns([1, 1, 1, 1.2])
    true = f1.multiselect("True class", CLASSES, default=CLASSES)
    pred = f2.multiselect("Predicted class", CLASSES, default=CLASSES)
    outcome = f3.segmented_control("Outcome", ["All", "Correct", "Wrong"], default="All") or "All"
    order = f4.selectbox("Sort by", ["Viral probability ↓", "Viral probability ↑", "Engagement rate ↓", "Random"])
    v = df[df.label.isin(true) & df.pred.isin(pred)]
    if outcome != "All":
        v = v[(v.label == v.pred) == (outcome == "Correct")]
    v = {"Viral probability ↓": v.sort_values("p_Viral", ascending=False),
         "Viral probability ↑": v.sort_values("p_Viral"),
         "Engagement rate ↓": v.sort_values("engagement_rate", ascending=False),
         "Random": v.sample(frac=1, random_state=0)}[order]
    st.caption(f"{len(v):,} posts match · model: {model} · accuracy on this selection {(v.label == v.pred).mean():.1%}"
               if len(v) else "No posts match.")

    page_size = 24
    n_pages = max(1, int(np.ceil(len(v) / page_size)))
    pg = st.number_input("Page", 1, n_pages, 1, key="explore_page") if n_pages > 1 else 1
    chunk = v.iloc[(pg - 1) * page_size: pg * page_size]
    tiles = []
    for i, r in enumerate(chunk.itertuples()):
        tiles.append(
            f'<div class="vs-tile" style="animation-delay:{i * 0.025:.3f}s">'
            f'<span class="chip" style="--c:{CLASS_COLORS[r.label]}">{r.label}</span>'
            f'<span class="chip pred" style="--c:{CLASS_COLORS[r.pred]}">→ {r.pred}</span>'
            f'<img src="{_thumb(r.image_path)}" alt="post by {html.escape(r.account)}" loading="lazy"/>'
            f'<div class="meta"><b style="color:#f2f1ee">@{html.escape(r.account)}</b><br/>'
            f'Viral {r.p_Viral:.0%} · ER {r.engagement_rate:.1%}</div></div>')
    st.markdown(f'<div class="vs-gallery">{"".join(tiles)}</div>', unsafe_allow_html=True)

    if len(chunk):
        section("Post detail")
        pick = st.selectbox("Choose a post", chunk.post_id, format_func=lambda p: f"@{chunk.set_index('post_id').loc[p, 'account']} · {p}")
        r = chunk.set_index("post_id").loc[pick]
        a, b, c = st.columns([1, 1, 1.3], gap="large")
        a.image(r.image_path, width="stretch")
        with b:
            charts.class_donut({k: float(r[f"p_{k}"]) for k in CLASSES}, height=260)
        with c:
            st.markdown(f"**True class:** {r.label} · **Predicted:** {r.pred}")
            st.markdown(f"**Followers:** {int(r.followers):,} (tier {int(r.follower_tier) + 1}) · "
                        f"**Engagement rate:** {r.engagement_rate:.2%}")
            st.markdown(f"**Posted:** {r.timestamp:%Y-%m-%d %H:%M} UTC")
            st.markdown(f"> {html.escape(str(r.caption))[:600]}")
