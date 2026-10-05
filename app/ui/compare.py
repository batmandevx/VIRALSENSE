"""Page: A/B test two drafts before publishing."""
import hashlib
import html
import tempfile
from datetime import datetime, time, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from ui import charts
from ui.analyze import load_artifacts
from ui.theme import CLASS_COLORS, SERIES, demo_banner, empty, hero, section
from viralsense.inference import featurize, post_frame
from viralsense.models.explain import viral_shap
from viralsense.suggest import suggest


def _draft(col, key: str, default_hour: int):
    with col:
        st.markdown(f"#### Draft {key}")
        up = st.file_uploader(f"Image {key}", type=["jpg", "jpeg", "png", "webp"], key=f"ab_img_{key}")
        cap = st.text_area(f"Caption {key}", height=110, key=f"ab_cap_{key}")
        c1, c2 = st.columns(2)
        day = c1.date_input(f"Date {key}", key=f"ab_day_{key}")
        hour = c2.time_input(f"Time {key} (UTC)", value=time(default_hour, 0), step=1800, key=f"ab_time_{key}")
    return up, cap, datetime.combine(day, hour).replace(tzinfo=timezone.utc)


def page(cfg: dict) -> None:
    hero("A/B compare", "Two drafts, one decision. ViralSense scores both under the same account settings, "
         "tells you which is likelier to land, and why.", "pre-publication split test")
    demo_banner(cfg)
    arts, missing = load_artifacts(cfg["paths"]["artifacts"], cfg["paths"]["processed"], cfg)
    if missing:
        empty("trained models", "python -m viralsense.pipeline")
        return
    with st.form("ab"):
        a_col, b_col = st.columns(2, gap="large")
        A = _draft(a_col, "A", 18)
        B = _draft(b_col, "B", 9)
        s1, s2, s3 = st.columns(3)
        followers = s1.number_input("Follower count", min_value=1, value=25_000, step=1_000)
        past = s2.number_input("Typical engagement rate, %", min_value=0.0, value=3.0, step=0.1)
        n_past = s3.number_input("Posts on your account so far", min_value=0, value=100, step=10)
        go = st.form_submit_button("Compare drafts", type="primary", width="stretch")
    if go:
        if A[0] is None or B[0] is None:
            st.warning("Upload an image for both drafts.")
            return
        st.session_state["ab"] = run(cfg, arts, [A, B], followers, past / 100 if past > 0 else None, int(n_past))
    res = st.session_state.get("ab")
    if res:
        show(cfg, res)


def run(cfg, arts, drafts, followers, past, n_past) -> dict:
    clf, spec = arts["classifier"], arts["spec"]
    out = {}
    with st.spinner("Scoring both drafts"):
        for key, (up, cap, when) in zip("AB", drafts):
            h = hashlib.sha256(up.getvalue()).hexdigest()[:12]
            path = Path(tempfile.gettempdir()) / f"viralsense_ab_{h}{Path(up.name).suffix}"
            path.write_bytes(up.getvalue())
            X = featurize(post_frame(str(path), [cap], followers, when, spec, past, n_past), cfg)
            from viralsense.models.calibrate import calibrated_proba

            proba = calibrated_proba(clf, X)[0]
            er = max(10 ** float(arts["regressor"]["model"].predict(X[arts["regressor"]["columns"]])[0]) - 1e-6, 0)
            out[key] = {"path": str(path), "caption": cap, "when": when, "X": X, "er": er,
                        "proba": dict(zip(clf["classes"], proba.tolist())), "shap": viral_shap(clf, X).iloc[0],
                        "suggest": suggest(X, clf, cfg, cap, top=3)}
    return out


def show(cfg, res: dict) -> None:
    a, b = res["A"], res["B"]
    win = "A" if a["proba"]["Viral"] >= b["proba"]["Viral"] else "B"
    lose = "B" if win == "A" else "A"
    gap = res[win]["proba"]["Viral"] - res[lose]["proba"]["Viral"]
    close = gap < 0.01
    st.markdown(f'<div class="vs-winner"><div class="vs-label">{"Too close to call" if close else "Likely winner"}</div>'
                f'<div class="big">{"Draft A ≈ Draft B" if close else f"Draft {win}"}</div>'
                f'<div class="vs-note">P(Viral) {res[win]["proba"]["Viral"]:.1%} vs {res[lose]["proba"]["Viral"]:.1%} '
                f'({gap * 100:+.1f} pts) · predicted engagement {res[win]["er"]:.2%} vs {res[lose]["er"]:.2%}</div></div>',
                unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap="large")
    for col, key in ((c1, "A"), (c2, "B")):
        with col:
            st.image(res[key]["path"], width="stretch")
            st.caption(f"{res[key]['when']:%a %d %b, %H:%M} UTC · “{res[key]['caption'][:120]}”")
    section("Class probabilities")
    classes = list(a["proba"])
    charts.grouped_bars_err(classes, {f"Draft {k}": ([res[k]["proba"][c] for c in classes], [np.nan] * 3) for k in "AB"},
                            height=300, ymax=1, colors=[SERIES[0], SERIES[1]])
    section(f"Why draft {win} scores higher", f"SHAP for draft {win} minus draft {lose}, per factor (Viral log-odds). "
            "Positive bars favour the winner.")
    d = (res[win]["shap"] - res[lose]["shap"]).sort_values(key=np.abs, ascending=False).head(10).iloc[::-1]
    charts.shap_bars(list(d.index), d.tolist(), height=360)
    section("How to improve each draft")
    c3, c4 = st.columns(2, gap="large")
    for col, key in ((c3, "A"), (c4, "B")):
        with col:
            st.markdown(f"**Draft {key}**")
            sg = res[key]["suggest"]
            for _, r in sg.iterrows():
                color = CLASS_COLORS["Viral"] if r.delta > 0.0005 else "#8d8c85"
                st.markdown(f'<div class="vs-variant" style="margin-bottom:8px"><div class="t"><b>{html.escape(r.change)}</b></div>'
                            f'<span style="color:{color};font-weight:800">{r.delta * 100:+.2f} pts</span> '
                            f'<span class="vs-note">→ P(Viral) {r.p_viral:.1%}</span></div>', unsafe_allow_html=True)
