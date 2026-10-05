"""Page: landing overview. Every number is read from reports/; missing results show as empty states."""
import html
from pathlib import Path

import pandas as pd
import streamlit as st

from ui import charts
from ui.theme import CLASS_COLORS, MODEL_NAMES, SERIES, demo_banner, empty, hero, kpis, section


def _csv(cfg, name):
    p = Path(cfg["paths"]["reports"]) / "tables" / name
    return pd.read_csv(p) if p.exists() else None


def _glass(label: str, body: str, note: str = "") -> str:
    n = f'<div class="vs-note">{html.escape(note)}</div>' if note else ""
    return f'<div class="vs-glass"><div class="vs-label">{html.escape(label)}</div>{body}{n}</div>'


def page(cfg: dict) -> None:
    hero("ViralSense", "Know how a post will land before you publish it. A multimodal model reads the image, caption, "
         "timing and account history, and a 12-persona AI jury and a posting-time bandit add their own signal.",
         "pre-publication virality prediction · Instagram")
    demo_banner(cfg)

    test, reg = _csv(cfg, "classification_test.csv"), _csv(cfg, "regression_summary.csv")
    comp = _csv(cfg, "phase1_sample_composition.csv")
    if test is None:
        empty("results", "python -m viralsense.pipeline")
        return
    sel = test[test.selected_by_cv].iloc[0]
    n_posts = int(comp.posts.sum()) if comp is not None else None
    n_acc = int(comp.accounts.sum()) if comp is not None else None
    kpis([
        {"label": "Posts analysed", "value": n_posts, "fmt": "int", "note": f"{n_acc} influencer accounts" if n_acc else ""},
        {"label": "Macro-F1 on unseen accounts", "value": sel.macro_f1, "fmt": "num3", "note": f"{MODEL_NAMES.get(sel.model, sel.model)} · chance ≈ 0.33"},
        {"label": "Viral recall", "value": sel.recall_Viral, "fmt": "pct", "note": "share of truly Viral posts caught"},
        {"label": "Viral PR-AUC", "value": sel.pr_auc_viral, "fmt": "num3", "note": f"vs {sel.viral_prevalence:.3f} for a random ranking"},
        {"label": "Engagement ranking ρ", "value": None if reg is None else reg.test_spearman.iloc[0], "fmt": "num3",
         "note": "Spearman, regression on test"},
    ], height=140)

    c1, c2 = st.columns([1.15, 1], gap="medium")
    with c1:
        section("Where the accuracy comes from", "Macro-F1 on held-out accounts. The baselines separate account history from the post itself.")
        t = test.drop_duplicates("model").set_index("model").sort_values("macro_f1")
        names = [MODEL_NAMES.get(n, n) for n in t.index]
        charts.hbars(names, t.macro_f1.tolist(), color=SERIES[0], height=40 + 34 * len(names), xname="macro-F1 (test)")
    with c2:
        imp = _csv(cfg, "shap_importance_viral.csv")
        section("What drives a Viral prediction", "Mean |SHAP| on test accounts; embedding components grouped.")
        if imp is not None:
            top = imp.head(7).iloc[::-1]
            charts.hbars(top.feature.tolist(), top.mean_abs_shap_viral.tolist(), color=SERIES[1], height=300)

    c3, c4, c5 = st.columns(3, gap="medium")
    with c3:
        v = Path(cfg["paths"]["reports"]) / "tables" / "ablation_verdict.txt"
        if v.exists():
            lines = v.read_text().strip().splitlines()
            helps = sum("HELPS" in l for l in lines)
            body = f'<div class="vs-stat">{helps}<small>of {len(lines)} checks</small></div>'
            st.markdown(_glass("Persona jury adds signal in", body, "Paired t-test across folds, minimum gain 0.01. "
                               "See Model insights → Persona jury."), unsafe_allow_html=True)
        else:
            st.markdown(_glass("Persona jury", '<div class="vs-stat">pending</div>', "Ablation runs after the jury finishes."),
                        unsafe_allow_html=True)
    with c4:
        b = _csv(cfg, "bandit_replay_summary.csv")
        if b is not None and "lift_ci_low" in b:
            pol = b[~b.policy.str.startswith("logged")].sort_values("mean_reward", ascending=False).iloc[0]
            sig = bool(pol.ci_excludes_zero)
            body = f'<div class="vs-stat">{pol.lift_vs_logged * 100:+.1f}<small>pts · best policy: {pol.policy}</small></div>'
            note = (f"Posting-time bandit vs what creators did (replay, held-out accounts). 95% CI "
                    f"{pol.lift_ci_low * 100:+.1f} to {pol.lift_ci_high * 100:+.1f} pts: "
                    f"{'a reliable gain' if sig else 'not distinguishable from zero'}.")
            st.markdown(_glass("Posting-time recommender", body, note), unsafe_allow_html=True)
    with c5:
        dist = _csv(cfg, "label_distribution.csv")
        if dist is not None:
            tot = dist[["Low", "Moderate", "Viral"]].sum()
            body = "".join(f'<span style="display:inline-block;margin-right:.9rem"><span style="color:{CLASS_COLORS[k]};font-weight:800;'
                           f'font-size:1.6rem">{tot[k] / tot.sum():.0%}</span> <span class="vs-note">{k}</span></span>'
                           for k in ["Low", "Moderate", "Viral"])
            st.markdown(_glass("Label definition", body, "Within each follower tier: bottom 60% Low, next 30% Moderate, top 10% Viral."),
                        unsafe_allow_html=True)

    from viralsense.jury.run import load_personas

    section("The AI jury", "Twelve persona agents rate every post on scroll-stop, emotional pull, shareability, save intent "
            "and comment trigger. Hover an avatar to see who it is.")
    av = "".join(f'<span title="{html.escape(p["name"])}, {p["age"]}" style="animation-delay:{i * 0.05:.2f}s">'
                 f'{p.get("avatar", "🙂")}</span>' for i, p in enumerate(load_personas(cfg)))
    st.markdown(f'<div class="vs-avatars">{av}</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:.8rem'></div>", unsafe_allow_html=True)
    pages = st.session_state.get("pages", {})
    links = [("analyse", "Analyse your post →"), ("compare", "A/B test two drafts →"), ("jury", "Meet the jury →"),
             ("explore", "Browse real predictions →"), ("insights", "Model insights →"), ("method", "How it works →")]
    cols = st.columns(3)
    for i, (key, label) in enumerate(links):
        if key in pages:
            cols[i % 3].page_link(pages[key], label=label)
