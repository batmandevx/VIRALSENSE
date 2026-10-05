"""Page: analyse one unpublished post."""
import hashlib
import html
import tempfile
from datetime import datetime, time, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from ui import charts
from ui.theme import CLASS_COLORS, FAINT, MUTED, badge, demo_banner, empty, hero, section

from viralsense.bandit.posting_time import recommend
from viralsense.inference import featurize, load_label_spec, post_frame
from viralsense.jury.client import OllamaClient
from viralsense.jury.run import load_personas
from viralsense.models.explain import viral_shap


@st.cache_resource(show_spinner=False)
def bubble(name: str, avatar: str, reaction: str, scores: dict, delay: float = 0.0) -> str:
    pills = "".join(f'<span class="vs-pill">{k.replace("_", " ")} <b>{int(v)}</b></span>' for k, v in scores.items())
    avg = sum(scores.values()) / max(1, len(scores))
    return (f'<div class="vs-msg" style="animation-delay:{delay:.2f}s"><div class="av">{avatar}</div>'
            f'<div class="bubble"><div class="who">{html.escape(name)} · avg {avg:.1f}/10</div>'
            f'<div class="txt">{html.escape(str(reaction))}</div><div class="vs-pills">{pills}</div></div></div>')


def typing(name: str, avatar: str) -> str:
    return (f'<div class="vs-msg"><div class="av">{avatar}</div><div class="bubble"><div class="who">{html.escape(name)} '
            f'is looking at your post</div><div class="vs-typing"><i></i><i></i><i></i></div></div></div>')


@st.cache_resource(show_spinner=False)
def _placeholder():
    return None


def load_artifacts(art_dir: str, processed: str, _cfg: dict):
    art = Path(art_dir)
    need = ["classifier.joblib", "regressor.joblib", "bandit.joblib"]
    missing = [n for n in need if not (art / n).exists()]
    if missing:
        return None, missing
    from viralsense.inference import load_model

    a = {n.split(".")[0]: load_model(art / n) for n in need}
    a["spec"] = load_label_spec(_cfg)
    a["caption_style"] = joblib.load(art / "caption_style.joblib") if (art / "caption_style.joblib").exists() else None
    posts = pd.read_parquet(Path(processed) / "posts.parquet", columns=["split", "follower_tier", "engagement_rate"])
    a["train_er"] = posts[posts.split == "train"]
    return a, []


def _ollama_ok(cfg: dict) -> bool:
    import requests

    try:
        tags = requests.get(f"{cfg['jury']['host']}/api/tags", timeout=2).json()
        return any(m["name"].startswith(cfg["jury"]["model"]) for m in tags.get("models", []))
    except Exception:
        return False


def page(cfg: dict) -> None:
    hero("Analyse a post", "Upload the image and caption you plan to publish. ViralSense predicts how it will do "
         "relative to accounts of your size, explains why, asks a 12-persona jury, and suggests a time and caption.",
         "pre-publication check")
    demo_banner(cfg)
    arts, missing = load_artifacts(cfg["paths"]["artifacts"], cfg["paths"]["processed"], cfg)
    if missing:
        empty("trained models", "python -m viralsense.pipeline")
        return
    llm = _ollama_ok(cfg)

    with st.form("post", border=True):
        left, right = st.columns([1.05, 1], gap="large")
        with left:
            upload = st.file_uploader("Image", type=["jpg", "jpeg", "png", "webp"])
            caption = st.text_area("Caption", height=150, placeholder="Write the caption exactly as you'd post it, hashtags included.")
        with right:
            followers = st.number_input("Follower count", min_value=1, value=25_000, step=1_000)
            c1, c2 = st.columns(2)
            day = c1.date_input("Planned date")
            hour = c2.time_input("Planned time (UTC)", value=time(18, 0), step=1800)
            p1, p2 = st.columns(2)
            past = p1.number_input("Typical engagement rate, %", min_value=0.0, value=0.0, step=0.1,
                                   help="Median (likes + comments) / followers of your recent posts. 0 = unknown. "
                                        "This is the model's strongest input.")
            n_past = p2.number_input("Posts on your account so far", min_value=0, value=100, step=10,
                                     help="Used with your typical engagement rate; the training median is 137.")
            j1, j2 = st.columns(2)
            run_jury = j1.toggle("Persona jury", value=llm, disabled=not llm, help="12 local vision-LLM calls")
            run_caps = j2.toggle("Caption variants", value=llm, disabled=not llm, help="3 rewrites via the local LLM")
            if not llm:
                st.caption(f"Local LLM `{cfg['jury']['model']}` not reachable: start `ollama serve` to enable jury and captions.")
        go = st.form_submit_button("Analyse post", type="primary", width="stretch")

    if go:
        if upload is None:
            st.warning("Upload an image first.")
            return
        key = hashlib.sha256(upload.getvalue() + caption.encode()).hexdigest()[:16]
        tmp = Path(tempfile.gettempdir()) / f"viralsense_{key}{Path(upload.name).suffix}"
        tmp.write_bytes(upload.getvalue())
        st.session_state["req"] = dict(path=str(tmp), caption=caption, followers=followers,
                                       when=datetime.combine(day, hour).replace(tzinfo=timezone.utc),
                                       past=past / 100 if past > 0 else None, n_past=int(n_past),
                                       jury=run_jury, caps=run_caps, key=key)
        st.session_state.pop("res", None)

    req = st.session_state.get("req")
    if not req:
        return
    if "res" not in st.session_state:
        st.session_state["res"] = compute(req, arts, cfg)
    show(req, st.session_state["res"], arts, cfg)


def compute(req: dict, arts: dict, cfg: dict) -> dict:
    spec, clf = arts["spec"], arts["classifier"]
    res = {}
    with st.status("Analysing your post", expanded=True) as status:
        st.write("Embedding the image (CLIP) and caption (MiniLM)")
        posts = post_frame(req["path"], [req["caption"]], req["followers"], req["when"], spec, req["past"], req["n_past"])
        X = featurize(posts, cfg)
        res["X"] = X
        st.write("Scoring with the classifier and regressor")
        from viralsense.models.calibrate import calibrated_proba, predict_label

        P = calibrated_proba(clf, X)
        res["proba"] = dict(zip(clf["classes"], P[0].tolist()))
        res["pred"] = clf["classes"][int(predict_label(clf, P)[0])]
        log_er = float(arts["regressor"]["model"].predict(X[arts["regressor"]["columns"]])[0])
        res["er"] = max(10 ** log_er - 1e-6, 0.0)
        res["tier"] = int(posts["follower_tier"].iloc[0])
        tier_er = arts["train_er"].loc[arts["train_er"].follower_tier == res["tier"], "engagement_rate"]
        res["score"] = float((tier_er < res["er"]).mean() * 100) if len(tier_er) else float("nan")
        st.write("Explaining the prediction (SHAP)")
        res["shap"] = viral_shap(clf, X).iloc[0]
        st.write("Ranking posting slots (LinUCB)")
        res["slots"] = recommend(arts["bandit"], X)

        if req["jury"]:
            from viralsense.jury.client import focus_group, react

            client, personas, rows = OllamaClient(cfg), load_personas(cfg), []
            st.write("The jury is reacting. Bubbles appear as each agent answers.")
            bar = st.progress(0.0, text="Asking the jury")
            live, shown = st.empty(), []
            for i, p in enumerate(personas):
                bar.progress(i / len(personas), text=f"{p.get('avatar', '')} {p['name']} ({i + 1}/12)")
                live.markdown('<div class="vs-chat">' + "".join(shown) + typing(p["name"], p.get("avatar", "")) + "</div>",
                              unsafe_allow_html=True)
                try:
                    r = react(client, req["path"], req["caption"], p)
                    rows.append({"persona": p["name"], "avatar": p.get("avatar", ""), "reaction": r["reaction"], **r["scores"]})
                    shown.append(bubble(p["name"], p.get("avatar", ""), r["reaction"], r["scores"]))
                except Exception as ex:
                    shown.append(bubble(p["name"], p.get("avatar", ""), f"(no answer: {type(ex).__name__})", {}))
            live.markdown('<div class="vs-chat">' + "".join(shown) + typing("Moderator", "🎙️") + "</div>", unsafe_allow_html=True)
            bar.progress(1.0, text="The moderator is summarising the discussion")
            res["jury"] = pd.DataFrame(rows).set_index("persona") if rows else None
            if rows:
                try:
                    F = cfg["jury"]["factors"]
                    res["focus"] = focus_group(client, req["caption"], [
                        {"name": r["persona"], "scores": {f: r[f] for f in F}, "reaction": r["reaction"]} for r in rows])
                except Exception as ex:
                    res["focus_error"] = f"{type(ex).__name__}: {ex}"
            bar.empty()
            live.empty()
        if req["caps"]:
            st.write("Writing and rescoring caption variants")
            try:
                from viralsense.bandit.captions import best_caption

                client = OllamaClient({**cfg, "jury": {**cfg["jury"], "model": cfg["captions"]["model"]}})
                res["captions"] = best_caption(clf, client, req["path"], req["caption"], req["followers"], req["when"],
                                               cfg, spec, req["past"], req["n_past"])
            except Exception as ex:
                res["captions_error"] = f"{type(ex).__name__}: {ex}"
        st.write("Searching for changes that raise P(Viral)")
        from viralsense.suggest import suggest

        variants = list(res["captions"].query("kind != 'original'").caption) if "captions" in res else None
        res["suggest"] = suggest(X, clf, cfg, req["caption"], variants)
        status.update(label="Analysis complete", state="complete", expanded=False)
    return res


def suggestions(req: dict, res: dict, arts: dict) -> None:
    from viralsense.bandit.caption_style import style_arm

    section("Suggestions", "Changes you control, ranked by how much they raise the model's P(Viral). Each one is "
            "re-scored with everything else held fixed; gains are the model's estimate, not a guarantee.")
    sug = res.get("suggest")
    if sug is not None and len(sug):
        cards = []
        for i, r in sug.iterrows():
            up = r.delta > 0.0005
            color = CLASS_COLORS["Viral"] if up else FAINT
            extra = (f'<div class="t" style="font-size:.82rem;color:{MUTED}">“{html.escape(str(r.new_caption))[:160]}”</div>'
                     if isinstance(r.get("new_caption"), str) else "")
            cards.append(
                f'<div class="vs-variant" style="animation-delay:{i * 0.07:.2f}s">'
                f'<div class="k">{"⏱ timing" if r.kind == "timing" else "✎ caption"}</div>'
                f'<div class="t"><b>{html.escape(r.change)}</b></div>{extra}'
                f'<div style="display:flex;justify-content:space-between;align-items:baseline">'
                f'<span style="font-size:1.35rem;font-weight:800;color:{color}">{r.delta * 100:+.2f} pts</span>'
                f'<span style="font-size:.78rem;color:{MUTED}">P(Viral) {r.p_viral:.1%}</span></div></div>')
        st.markdown('<div class="vs-gallery" style="grid-template-columns:repeat(auto-fill,minmax(230px,1fr))">'
                    + "".join(cards) + "</div>", unsafe_allow_html=True)
        if sug.delta.max() <= 0.0005:
            st.caption("None of the searched changes beats the post as planned.")

    cs = arts.get("caption_style")
    if cs is not None:
        X = res["X"]
        mine = style_arm(X["cap_len"], X["cap_hashtag_count"], X["cap_has_question"], cs["len_edges"])[0]
        post = cs["posterior"]
        tier = post[(post.follower_tier == res["tier"]) & (post.n >= 20)].head(5)
        if len(tier):
            with st.expander(f"Caption styles that work for accounts of your size (yours: {mine})"):
                st.caption("From the caption-style bandit: share of Moderate-or-Viral posts per style in your follower tier, "
                           "with a 95% Beta credible interval. Styles seen fewer than 20 times are hidden.")
                show = tier.assign(rate=tier.posterior_mean.map("{:.1%}".format),
                                   interval=[f"{a:.0%} to {b:.0%}" for a, b in zip(tier.ci_low, tier.ci_high)])
                st.dataframe(show[["arm", "n", "rate", "interval"]].rename(columns={"arm": "style", "n": "posts"}),
                             hide_index=True, width="stretch")


def _with_time(X: pd.DataFrame, hours, weekday: int | None = None) -> pd.DataFrame:
    Z = pd.concat([X] * len(hours), ignore_index=True)
    h = np.asarray(hours)
    Z["meta_hour"], Z["meta_hour_sin"], Z["meta_hour_cos"] = h, np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24)
    if weekday is not None:
        Z["meta_weekday"] = weekday
    return Z


def whatif(req: dict, res: dict, arts: dict, cfg: dict) -> None:
    from viralsense.features.text import text_features

    clf = arts["classifier"]
    X = res["X"]
    section("What if…", "Change one thing and re-score instantly. Everything else about the post stays fixed.")
    t1, t2 = st.tabs(["Posting time", "Caption"])
    with t1:
        days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        wd = st.segmented_control("Weekday", days, default=days[req["when"].weekday()], key="wi_day") or days[req["when"].weekday()]
        from viralsense.models.calibrate import calibrated_proba

        Z = _with_time(X, range(24), days.index(wd))
        P = calibrated_proba(clf, Z)
        charts.lines(np.arange(24), {"P(Viral)": P[:, 2], "P(Moderate or Viral)": P[:, 1] + P[:, 2]}, height=300,
                     xname="hour (UTC)", yname="probability", ref=(f"planned {req['when'].hour:02d}:00", float(P[req["when"].hour, 2])))
        best = int(P[:, 2].argmax())
        spread = P[:, 2].max() - P[:, 2].min()
        note = " Timing matters little for this post next to its content and account history." if spread < 0.02 else ""
        st.caption(f"Best hour on {wd}: {best:02d}:00 UTC (P(Viral) {P[best, 2]:.1%}). Across the day the model's "
                   f"Viral probability ranges over {spread:.1%}.{note}")
    with t2:
        new = st.text_area("Edit the caption", value=req["caption"], height=110, key="wi_cap")
        if st.button("Re-score caption", key="wi_go"):
            T = text_features(pd.DataFrame({"post_id": ["x"], "caption": [new]}), cfg).drop(columns="post_id")
            Z = X.copy()
            for c in T.columns:
                Z[c] = T[c].to_numpy()
            from viralsense.models.calibrate import calibrated_proba

            p_new = calibrated_proba(clf, Z)[0]
            p_old = np.array([res["proba"][k] for k in clf["classes"]])
            cols = st.columns(3)
            for col, k, a, b in zip(cols, clf["classes"], p_old, p_new):
                col.metric(f"P({k})", f"{b:.1%}", f"{(b - a) * 100:+.1f} pts", delta_color="normal" if k != "Low" else "inverse")


def show(req: dict, res: dict, arts: dict, cfg: dict) -> None:
    proba = res["proba"]
    pred = res.get("pred") or max(proba, key=proba.get)
    n_tiers = len(arts["spec"]["tier_edges"]) + 1

    if req["past"] is None:
        st.info("You left your typical engagement rate blank. It is the model's strongest input, so without it the "
                "prediction relies on the post alone and tends to lean toward Low. Add it for a sharper estimate.",
                icon=":material/info:")
    section("Verdict", f"Relative to follower tier {res['tier'] + 1} of {n_tiers}. Labels are the bottom 60% / next 30% / "
            "top 10% of engagement within a tier.")
    a, b, c, d = st.columns([1.05, 1, 1, 1], gap="medium")
    with a:
        st.image(req["path"], width="stretch")
    with b:
        st.markdown(f'<div style="color:{FAINT};font-size:.8rem;font-weight:600;margin-bottom:.4rem">PREDICTED CLASS</div>',
                    unsafe_allow_html=True)
        badge(pred)
        charts.class_donut(proba, height=240)
    with c:
        charts.gauge(proba["Viral"] * 100, "chance of Viral", CLASS_COLORS["Viral"])
        st.caption(f"Base rate is 10% by construction; {proba['Viral'] / 0.10:.1f}× the base rate.")
    with d:
        if np.isnan(res["score"]):
            st.metric("Engagement score", "n/a")
        else:
            charts.gauge(res["score"], "engagement score", "#9085e9", suffix="")
        st.caption(f"Predicted engagement rate {res['er']:.2%}; score = its percentile among training posts in your tier.")

    section("Why", "SHAP contributions of each factor to the Viral prediction. Embedding components are grouped.")
    s = res["shap"]
    top = s.reindex(s.abs().sort_values(ascending=False).index).head(10).iloc[::-1]
    charts.shap_bars(list(top.index), top.tolist(), height=360)

    suggestions(req, res, arts)

    left, right = st.columns(2, gap="large")
    with left:
        section("Persona jury", "Twelve simulated viewers rate the post from 1 to 10. They never see follower or engagement numbers.")
        jury = res.get("jury")
        if jury is None:
            st.info("Jury not run for this analysis.")
        else:
            f = cfg["jury"]["factors"]
            mean = jury[f].mean()
            spread = jury[f].mean(axis=1)
            fans, critics = spread.idxmax(), spread.idxmin()
            charts.radar(f, {"jury mean": mean.tolist(), f"most positive: {fans}": jury.loc[fans, f].tolist(),
                             f"least positive: {critics}": jury.loc[critics, f].tolist()}, height=380)
            st.caption(f"Disagreement (std of persona means): {spread.std(ddof=0):.2f}. The deployed classifier does not "
                       "use jury scores; Model insights → Persona jury shows whether they help.")
            with st.expander("All 12 personas as a heatmap"):
                charts.heatmap(list(jury.index), [x.replace("_", " ") for x in f], jury[f].to_numpy(), 1, 10, height=440, decimals=0)
    with right:
        section("When to post", "LinUCB expected reward (chance of Moderate or Viral) per 3-hour UTC slot.")
        slots = res["slots"].sort_values("slot")
        best = int(res["slots"]["slot"].iloc[0])
        charts.clock(slots["slot_label"].str.replace(" UTC", "").tolist(), slots["expected_reward"].tolist(),
                     slots["slot"].tolist().index(best), height=360)
        planned = req["when"].hour // (24 // arts["bandit"]["n_slots"])
        msg = "matches your planned time" if planned == best else f"your planned slot ranks #{int(res['slots'].index[res['slots'].slot == planned][0]) + 1}" \
            if (res["slots"].slot == planned).any() else "your planned slot has no training data"
        st.markdown(f"**Best slot: {res['slots']['slot_label'].iloc[0]}** ({msg}).")
        st.caption("Learned from logged posts. Creators chose their own times, so this is a correlation, not a tested causal effect.")

    jury = res.get("jury")
    if jury is not None and "reaction" in jury:
        section("What the jury said", "Each agent's one-line reaction, in its own voice, followed by the moderator's summary.")
        f = cfg["jury"]["factors"]
        left, right = st.columns([1.4, 1], gap="large")
        with left:
            msgs = [bubble(name, r.avatar, r.reaction, {k: r[k] for k in f}, i * 0.08)
                    for i, (name, r) in enumerate(jury.assign(_avg=jury[f].mean(axis=1)).sort_values("_avg", ascending=False).iterrows())]
            st.markdown('<div class="vs-chat">' + "".join(msgs) + "</div>", unsafe_allow_html=True)
        with right:
            fg = res.get("focus")
            if fg:
                tips = "".join(f"<li>{html.escape(t)}</li>" for t in fg.get("tips", []))
                st.markdown(f'<div class="vs-glass"><div class="vs-label">Focus-group moderator</div>'
                            f'<div class="vs-stat" style="font-size:1.25rem;margin:.3rem 0">{html.escape(fg.get("verdict", ""))}</div>'
                            f'<div class="vs-note"><b>Consensus.</b> {html.escape(fg.get("consensus", ""))}</div>'
                            f'<div class="vs-note"><b>Disagreement.</b> {html.escape(fg.get("disagreement", ""))}</div>'
                            f'<div class="vs-note"><b>Try this:</b><ul style="margin:.3rem 0 0 1rem;padding:0">{tips}</ul></div></div>',
                            unsafe_allow_html=True)
            elif "focus_error" in res:
                st.warning(f"Moderator summary failed: {res['focus_error']}")

    section("Caption variants", "Three rewrites from the local LLM, each rescored by the classifier with everything else unchanged.")
    if "captions_error" in res:
        st.error(f"Caption generation failed: {res['captions_error']}")
    elif "captions" not in res:
        st.info("Caption variants not generated for this analysis.")
    else:
        ranked = res["captions"]
        orig = ranked[ranked.kind == "original"].iloc[0]
        variants = ranked[ranked.kind != "original"]
        cols = st.columns(len(variants) + 1, gap="small")
        for i, (col, (_, r)) in enumerate(zip(cols, pd.concat([orig.to_frame().T, variants]).iterrows())):
            best_v = r.kind != "original" and r.caption == variants.iloc[0].caption
            label = "original" if r.kind == "original" else ("best variant ✦" if best_v else r.kind)
            col.markdown(
                f'<div class="vs-variant{" best" if best_v else ""}" style="animation-delay:{i * 0.1:.1f}s">'
                f'<div class="k">{html.escape(label)}</div><div class="t">{html.escape(str(r.caption))}</div>'
                f'<div style="display:flex;justify-content:space-between;font-size:.78rem;color:{MUTED}">'
                f'<span>Viral probability</span><b>{float(r.p_Viral):.1%}</b></div>'
                f'<div class="vs-bar"><span style="width:{min(100, float(r.p_Viral) * 100 / max(ranked.p_Viral.max(), 1e-9)):.0f}%"></span></div></div>',
                unsafe_allow_html=True)
        delta = variants.iloc[0].p_Viral - orig.p_Viral
        st.caption(f"Best variant changes the model's Viral probability by {delta:+.1%}. This is the model's opinion, not measured engagement.")

    whatif(req, res, arts, cfg)
