"""Page: the 12 persona agents: who they are, how they score, whose taste matches real engagement."""
import html
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from ui import charts
from ui.theme import SERIES, demo_banner, empty, hero, kpis, section
from viralsense.jury.client import BIG5
from viralsense.jury.run import load_personas


def _csv(cfg, name, **kw):
    p = Path(cfg["paths"]["reports"]) / "tables" / name
    return pd.read_csv(p, **kw) if p.exists() else None


def page(cfg: dict) -> None:
    hero("Meet the jury", "Twelve AI audience agents, each with its own age, interests, scrolling habits and Big Five "
         "personality. Every agent looks at the image and caption, never the numbers, and rates the post on five factors.",
         "multi-agent persona simulation")
    demo_banner(cfg)
    personas = load_personas(cfg)
    stats = _csv(cfg, "jury_persona_stats.csv", index_col=0)
    prog = _csv(cfg, "jury_progress.csv")
    usage = _csv(cfg, "jury_usage_summary.csv")
    if prog is not None:
        kpis([
            {"label": "Posts rated", "value": prog.complete_posts.iloc[0], "fmt": "int", "note": f"of {cfg['jury']['n_posts']:,} in the jury subset"},
            {"label": "Agent ratings", "value": prog.ratings.iloc[0], "fmt": "int", "note": "12 agents × 5 factors each"},
            {"label": "Agents", "value": len(personas), "fmt": "int", "note": f"{cfg['jury']['model']} · local · $0"},
            {"label": "Seconds per rating", "value": None if usage is None else usage.sec_per_call.iloc[0], "fmt": "num3",
             "note": "local vision LLM on Apple GPU"},
        ])

    section("The agents", "Hover a card. The bars are Big Five traits (1–5); the numbers below come from their real ratings.")
    cards = []
    for i, p in enumerate(personas):
        b5 = p.get("big_five", {})
        meters = "".join(f'<div style="display:flex;justify-content:space-between;font-size:.68rem;color:#8d8c85;margin-top:.3rem">'
                         f'<span>{k}</span><span>{b5.get(k, 0)}/5</span></div>'
                         f'<div class="vs-meter"><span style="width:{b5.get(k, 0) * 20}%"></span></div>' for k in BIG5)
        s = stats.loc[p["id"]] if stats is not None and p["id"] in stats.index else None
        nums = (f'<div class="vs-pills" style="margin-top:.6rem"><span class="vs-pill">avg <b>{s.overall:.1f}</b>/10</span>'
                f'<span class="vs-pill">spread <b>{s.spread:.2f}</b></span>'
                f'<span class="vs-pill">ρ with engagement <b>{s.spearman_vs_engagement:+.2f}</b></span></div>') if s is not None else ""
        chips = "".join(f'<span class="vs-chip">{html.escape(t)}</span>' for t in p["interests"])
        cards.append(f'<div class="vs-persona" style="animation-delay:{i * 0.04:.2f}s"><div class="ava">{p.get("avatar", "🙂")}</div>'
                     f'<h4>{html.escape(p["name"])}</h4><div class="meta">age {p["age"]} · {html.escape(p.get("voice", ""))}</div>'
                     f'<div class="bio">{html.escape(p.get("bio", ""))}</div>{chips}{meters}{nums}</div>')
    st.markdown('<div class="vs-gallery" style="grid-template-columns:repeat(auto-fill,minmax(250px,1fr))">' + "".join(cards)
                + "</div>", unsafe_allow_html=True)

    if stats is None:
        empty("jury analytics", "python -m viralsense.jury.report")
        return
    F = cfg["jury"]["factors"]
    c1, c2 = st.columns(2, gap="large")
    with c1:
        section("Harsh or generous?", "Average overall score each agent gives (1–10).")
        s = stats.sort_values("overall")
        charts.hbars(s.name.tolist(), s.overall.tolist(), color=SERIES[6], height=420, xname="mean score")
    with c2:
        section("Whose taste matches real engagement?", "Spearman ρ between each agent's score and the post's real "
                "engagement rate (training accounts). Near zero means that agent's taste does not predict virality.")
        s = stats.sort_values("spearman_vs_engagement")
        charts.hbars(s.name.tolist(), s.spearman_vs_engagement.tolist(), color=SERIES[4], height=420, xname="Spearman ρ")
    section("How each agent scores the five factors")
    hm = stats.set_index("name")[F]
    charts.heatmap(list(hm.index), [f.replace("_", " ") for f in F], hm.to_numpy(), float(hm.values.min()),
                   float(hm.values.max()), height=480, colors=("#1a1622", "#8134af", "#f58529"))
    c3, c4 = st.columns([1.3, 1], gap="large")
    agree = _csv(cfg, "jury_persona_agreement.csv", index_col=0)
    fac = _csv(cfg, "jury_factor_correlation.csv", index_col=0)
    with c3:
        if agree is not None:
            section("Who agrees with whom", "Spearman correlation of overall scores across the same posts.")
            charts.heatmap(list(agree.index), list(agree.columns), agree.to_numpy(), float(np.nanmin(agree.values)), 1.0,
                           height=520, decimals=2)
    with c4:
        if fac is not None:
            section("Are the five factors distinct?", "Correlation between the jury's mean factor scores across posts.")
            charts.heatmap([f.replace("_", " ") for f in fac.index], [f.replace("_", " ") for f in fac.columns],
                           fac.to_numpy(), float(np.nanmin(fac.values)), 1.0, height=420, decimals=2)
