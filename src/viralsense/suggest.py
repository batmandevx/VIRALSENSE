"""Suggestion engine (idea from the team's notebook): search the changes a creator controls and rank
them by how much they raise the classifier's P(Viral). Every candidate is re-scored by the same model,
with everything else about the post held fixed. Gains are the model's opinion, not measured engagement.
"""
import re

import numpy as np
import pandas as pd

from viralsense.features.text import text_features

DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
HASHTAG = re.compile(r"#\w+")


def caption_edits(caption: str) -> dict[str, str]:
    """Deterministic, explainable rewrites of the creator's own caption."""
    c = caption.strip()
    tags = HASHTAG.findall(c)
    body = HASHTAG.sub("", c).strip()
    edits = {}
    if "?" not in c:
        edits["Ask the audience a question"] = f"{body} What do you think?" + (" " + " ".join(tags) if tags else "")
    if tags:
        edits["Remove all hashtags"] = body
    if len(tags) > 5:
        edits["Keep only the first 5 hashtags"] = body + " " + " ".join(tags[:5])
    sentences = re.split(r"(?<=[.!?])\s+", body)
    if len(sentences) > 1:
        edits["Shorten to the first sentence"] = sentences[0] + (" " + " ".join(tags) if tags else "")
    return {k: re.sub(r"\s+", " ", v).strip() for k, v in edits.items() if v.strip() and v.strip() != c}


def _with_time(X: pd.DataFrame, hours, weekdays) -> pd.DataFrame:
    Z = pd.concat([X] * len(hours), ignore_index=True)
    h = np.asarray(hours)
    Z["meta_hour"], Z["meta_hour_sin"], Z["meta_hour_cos"] = h, np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24)
    Z["meta_weekday"] = np.asarray(weekdays)
    return Z


def _with_caption(X: pd.DataFrame, captions: list[str], cfg: dict) -> pd.DataFrame:
    T = text_features(pd.DataFrame({"post_id": [f"c{i}" for i in range(len(captions))], "caption": captions}), cfg)
    Z = pd.concat([X] * len(captions), ignore_index=True)
    for c in T.columns.drop("post_id"):
        Z[c] = T[c].to_numpy()
    return Z


def suggest(X: pd.DataFrame, clf: dict, cfg: dict, caption: str, variants: list[str] | None = None,
            top: int = 6) -> pd.DataFrame:
    """X: the post's single feature row. Returns candidate changes ranked by gain in P(Viral)."""
    from viralsense.models.calibrate import calibrated_proba

    vi = clf["classes"].index("Viral")
    proba = lambda Z: calibrated_proba(clf, Z) if "calibrators" in clf else clf["model"].predict_proba(Z[clf["columns"]])
    base = float(proba(X)[0, vi])
    rows = []

    cur_h, cur_d = int(X["meta_hour"].iloc[0]), int(X["meta_weekday"].iloc[0])
    grid = [(h, d) for d in range(7) for h in range(24) if (h, d) != (cur_h, cur_d)]
    P = proba(_with_time(X, [h for h, _ in grid], [d for _, d in grid]))[:, vi]
    for (h, d), p in zip(grid, P):
        what = f"Post on {DAYS[d]} at {h:02d}:00 UTC" if d != cur_d else f"Post at {h:02d}:00 UTC instead"
        rows.append({"kind": "timing", "change": what, "p_viral": p})

    edits = caption_edits(caption)
    llm = {f"Use AI variant {i + 1}": v for i, v in enumerate(variants or [])}
    texts = {**edits, **llm}
    if texts:
        P = proba(_with_caption(X, list(texts.values()), cfg))[:, vi]
        for (what, text), p in zip(texts.items(), P):
            rows.append({"kind": "caption", "change": what, "p_viral": p, "new_caption": text})

    out = pd.DataFrame(rows)
    out["delta"] = out["p_viral"] - base
    out["base_p_viral"] = base
    # Keep only the 3 best timing options so the list is not dominated by near-duplicate hours.
    timing = out[out.kind == "timing"].sort_values("delta", ascending=False).head(3)
    caps = out[out.kind == "caption"]
    return pd.concat([timing, caps]).sort_values("delta", ascending=False).head(top).reset_index(drop=True)
