"""Visual language: tokens, global CSS with motion, hero, KPI tiles, cards, empty states."""
import html
import json

import streamlit as st

# Data colours: categorical slots 1-3 (dark-mode steps), validated as a set.
CLASS_COLORS = {"Low": "#3987e5", "Moderate": "#d95926", "Viral": "#199e70"}
SERIES = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]
TOWARD, AWAY = "#d95926", "#3987e5"  # diverging poles for SHAP
TEXT, MUTED, FAINT, GRID, SURFACE = "#f2f1ee", "#c3c2b7", "#8d8c85", "#2a2931", "#17161d"
MODEL_NAMES = {"dummy": "Dummy (stratified)", "logreg": "Logistic regression", "rf": "Random forest", "xgb": "XGBoost",
               "lgbm": "LightGBM", "hgb": "HistGradientBoosting", "vote": "Soft voting", "stack": "Stacking ensemble",
               "xgb_tuned": "XGBoost (Optuna-tuned)", "lgbm_tuned": "LightGBM (Optuna-tuned)",
               "xgb [history only]": "XGBoost · account history only", "xgb [no history]": "XGBoost · post only (no history)"}
BRAND = "linear-gradient(90deg,#f58529 0%,#dd2a7b 45%,#8134af 75%,#515bd4 100%)"

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stMarkdown, .stText {{ font-family: 'Inter', system-ui, sans-serif; }}
.block-container {{ padding-top: 1.6rem; max-width: 1280px; }}
/* Ambient gradient mesh: slow drift, sits behind everything */
[data-testid="stAppViewContainer"]::before {{ content: ""; position: fixed; inset: -20%; z-index: 0; pointer-events: none;
  background: radial-gradient(40% 35% at 15% 10%, rgba(221,42,123,.16), transparent 60%),
              radial-gradient(35% 30% at 85% 15%, rgba(81,91,212,.16), transparent 60%),
              radial-gradient(45% 40% at 60% 95%, rgba(245,133,41,.10), transparent 60%);
  filter: blur(20px); animation: vsMesh 28s ease-in-out infinite alternate; }}
@keyframes vsMesh {{ 0% {{ transform: translate3d(0,0,0) scale(1); }} 100% {{ transform: translate3d(-3%,2%,0) scale(1.06); }} }}
[data-testid="stMain"], [data-testid="stSidebar"] {{ position: relative; z-index: 1; }}
[data-testid="stHeader"] {{ background: rgba(14,13,18,.6); backdrop-filter: blur(14px); }}
.vs-glass {{ background: rgba(23,22,29,.62); backdrop-filter: blur(14px) saturate(140%); -webkit-backdrop-filter: blur(14px) saturate(140%);
  border: 1px solid rgba(255,255,255,.07); border-radius: 18px; padding: 1.1rem 1.2rem; animation: vsFadeUp .55s ease both; }}
.vs-bento {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; }}
.vs-bento .span2 {{ grid-column: span 2; }}
@media (max-width: 900px) {{ .vs-bento {{ grid-template-columns: 1fr; }} .vs-bento .span2 {{ grid-column: auto; }} }}
.vs-stat {{ font-size: 2.4rem; font-weight: 800; letter-spacing: -.03em; color: {TEXT}; line-height: 1.05; }}
.vs-stat small {{ font-size: .95rem; color: {MUTED}; font-weight: 600; margin-left: .25rem; }}
.vs-label {{ font-size: .74rem; text-transform: uppercase; letter-spacing: .12em; color: {FAINT}; font-weight: 700; }}
.vs-note {{ font-size: .84rem; color: {MUTED}; margin-top: .35rem; line-height: 1.45; }}
.vs-gallery {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(170px, 1fr)); gap: 12px; }}
.vs-tile {{ position: relative; border-radius: 14px; overflow: hidden; border: 1px solid {GRID}; background: {SURFACE};
  animation: vsFadeUp .5s ease both; transition: transform .25s ease, box-shadow .25s ease; }}
.vs-tile:hover {{ transform: translateY(-3px) scale(1.015); box-shadow: 0 14px 30px rgba(0,0,0,.45); }}
.vs-tile img {{ width: 100%; aspect-ratio: 1/1; object-fit: cover; display: block; }}
.vs-tile .meta {{ padding: .5rem .6rem .6rem; font-size: .76rem; color: {MUTED}; }}
.vs-tile .chip {{ position: absolute; top: 8px; left: 8px; padding: .15rem .5rem; border-radius: 999px; font-size: .7rem;
  font-weight: 700; color: #fff; background: var(--c); box-shadow: 0 2px 8px rgba(0,0,0,.4); }}
.vs-tile .chip.pred {{ left: auto; right: 8px; background: rgba(14,13,18,.75); border: 1.5px solid var(--c); }}
[data-testid="stSidebar"] {{ background: #121117; border-right: 1px solid {GRID}; }}

@keyframes vsFadeUp {{ from {{ opacity: 0; transform: translateY(14px); }} to {{ opacity: 1; transform: none; }} }}
@keyframes vsShimmer {{ 0% {{ background-position: 0% 50%; }} 100% {{ background-position: 200% 50%; }} }}
@keyframes vsPulse {{ 0%,100% {{ box-shadow: 0 0 0 0 var(--glow); }} 50% {{ box-shadow: 0 0 0 10px transparent; }} }}
@keyframes vsFlow {{ from {{ background-position: 0 0; }} to {{ background-position: 40px 0; }} }}

.vs-hero {{ animation: vsFadeUp .6s ease both; margin-bottom: .6rem; }}
.vs-hero h1 {{ font-size: 2.7rem; font-weight: 800; letter-spacing: -.03em; margin: 0;
  background: {BRAND}; background-size: 200% auto; -webkit-background-clip: text; background-clip: text;
  color: transparent; animation: vsShimmer 6s linear infinite; }}
.vs-hero p {{ color: {MUTED}; font-size: 1.05rem; margin: .3rem 0 0; max-width: 760px; }}
.vs-eyebrow {{ text-transform: uppercase; letter-spacing: .14em; font-size: .72rem; font-weight: 700; color: {FAINT}; }}

.vs-card {{ background: {SURFACE}; border: 1px solid {GRID}; border-radius: 16px; padding: 1.1rem 1.2rem;
  animation: vsFadeUp .55s ease both; transition: transform .2s ease, border-color .2s ease; }}
.vs-card:hover {{ transform: translateY(-2px); border-color: #3a3942; }}
.vs-card h3 {{ margin: 0 0 .25rem; font-size: 1rem; font-weight: 700; color: {TEXT}; }}
.vs-card .vs-sub {{ color: {FAINT}; font-size: .82rem; margin-bottom: .4rem; }}

.vs-badge {{ display: inline-flex; align-items: center; gap: .5rem; padding: .5rem 1.1rem; border-radius: 999px;
  font-weight: 800; font-size: 1.6rem; letter-spacing: -.01em; color: {TEXT};
  border: 2px solid var(--c); background: color-mix(in srgb, var(--c) 18%, transparent);
  --glow: color-mix(in srgb, var(--c) 55%, transparent); animation: vsPulse 2.2s ease-in-out 3, vsFadeUp .5s ease both; }}
.vs-badge .dot {{ width: .7rem; height: .7rem; border-radius: 50%; background: var(--c); }}

.vs-banner {{ border-radius: 14px; padding: .8rem 1rem; margin: .2rem 0 1rem; font-weight: 600;
  background: repeating-linear-gradient(135deg, #3a2a05 0 14px, #2e2205 14px 28px); border: 1px solid #c98500;
  color: #ffd27a; animation: vsFadeUp .4s ease both; }}
.vs-empty {{ border: 1px dashed #3a3942; border-radius: 14px; padding: 1.2rem; color: {MUTED}; text-align: center; }}
.vs-empty code {{ color: {TEXT}; }}
.vs-verdict {{ border-radius: 12px; padding: .7rem .9rem; margin: .35rem 0; border-left: 4px solid var(--c);
  background: color-mix(in srgb, var(--c) 10%, {SURFACE}); color: {TEXT}; font-size: .9rem; animation: vsFadeUp .5s ease both; }}
.vs-variant {{ background: {SURFACE}; border: 1px solid {GRID}; border-radius: 14px; padding: .9rem 1rem; height: 100%;
  animation: vsFadeUp .5s ease both; }}
.vs-variant.best {{ border-color: {CLASS_COLORS['Viral']}; box-shadow: 0 0 0 1px {CLASS_COLORS['Viral']} inset; }}
.vs-variant .k {{ font-size: .72rem; text-transform: uppercase; letter-spacing: .1em; color: {FAINT}; font-weight: 700; }}
.vs-variant .t {{ color: {TEXT}; margin: .35rem 0 .6rem; font-size: .93rem; line-height: 1.45; }}
.vs-bar {{ height: 6px; border-radius: 4px; background: {GRID}; overflow: hidden; }}
.vs-bar > span {{ display: block; height: 100%; border-radius: 4px; background: {CLASS_COLORS['Viral']};
  transform-origin: left; animation: vsGrow 1s cubic-bezier(.2,.8,.2,1) both; }}
@keyframes vsGrow {{ from {{ transform: scaleX(0); }} to {{ transform: scaleX(1); }} }}

.vs-flow {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: .7rem; }}
.vs-step {{ background: {SURFACE}; border: 1px solid {GRID}; border-radius: 14px; padding: .9rem; position: relative;
  animation: vsFadeUp .5s ease both; }}
.vs-step .n {{ font-weight: 800; font-size: .75rem; color: transparent; background: {BRAND}; -webkit-background-clip: text;
  background-clip: text; letter-spacing: .12em; }}
.vs-step h4 {{ margin: .2rem 0 .3rem; font-size: .95rem; color: {TEXT}; }}
.vs-step p {{ margin: 0; font-size: .8rem; color: {MUTED}; line-height: 1.4; }}
.vs-step::after {{ content: ""; position: absolute; left: 0; right: 0; bottom: 0; height: 3px; border-radius: 0 0 14px 14px;
  background: repeating-linear-gradient(90deg, #dd2a7b 0 10px, transparent 10px 20px); animation: vsFlow 1.2s linear infinite; opacity: .7; }}
.vs-flow .vs-step:nth-child(2) {{ animation-delay: .08s; }} .vs-flow .vs-step:nth-child(3) {{ animation-delay: .16s; }}
.vs-flow .vs-step:nth-child(4) {{ animation-delay: .24s; }} .vs-flow .vs-step:nth-child(5) {{ animation-delay: .32s; }}
.vs-flow .vs-step:nth-child(6) {{ animation-delay: .40s; }}

/* Jury chat */
.vs-chat {{ display: flex; flex-direction: column; gap: 10px; }}
.vs-msg {{ display: flex; gap: 10px; align-items: flex-start; animation: vsFadeUp .45s ease both; }}
.vs-msg .av {{ flex: 0 0 38px; height: 38px; border-radius: 50%; display: grid; place-items: center; font-size: 1.2rem;
  background: linear-gradient(135deg, rgba(221,42,123,.35), rgba(81,91,212,.35)); border: 1px solid rgba(255,255,255,.12); }}
.vs-msg .bubble {{ background: rgba(23,22,29,.75); border: 1px solid rgba(255,255,255,.08); border-radius: 4px 16px 16px 16px;
  padding: .55rem .8rem; max-width: 100%; }}
.vs-msg .who {{ font-size: .74rem; color: {FAINT}; font-weight: 700; letter-spacing: .02em; }}
.vs-msg .txt {{ color: {TEXT}; font-size: .9rem; margin: .15rem 0 .35rem; line-height: 1.4; }}
.vs-typing {{ display: inline-flex; gap: 4px; padding: .35rem 0 .2rem; }}
.vs-typing i {{ width: 7px; height: 7px; border-radius: 50%; background: {MUTED}; opacity: .35; animation: vsDot 1.2s ease-in-out infinite; }}
.vs-typing i:nth-child(2) {{ animation-delay: .15s; }} .vs-typing i:nth-child(3) {{ animation-delay: .3s; }}
@keyframes vsDot {{ 0%, 80%, 100% {{ opacity: .25; transform: translateY(0); }} 40% {{ opacity: 1; transform: translateY(-4px); }} }}
.vs-pills {{ display: flex; flex-wrap: wrap; gap: 4px; }}
.vs-pill {{ font-size: .68rem; padding: .1rem .45rem; border-radius: 999px; background: rgba(255,255,255,.06); color: {MUTED}; }}
.vs-pill b {{ color: {TEXT}; }}
/* Persona cards */
.vs-persona {{ background: rgba(23,22,29,.7); border: 1px solid rgba(255,255,255,.07); border-radius: 18px; padding: 1rem;
  animation: vsFadeUp .5s ease both; transition: transform .25s ease, border-color .25s ease; height: 100%; }}
.vs-persona:hover {{ transform: translateY(-4px); border-color: rgba(221,42,123,.5); }}
.vs-persona .ava {{ font-size: 2rem; width: 56px; height: 56px; border-radius: 16px; display: grid; place-items: center;
  background: linear-gradient(135deg, rgba(245,133,41,.3), rgba(129,52,175,.35)); margin-bottom: .5rem; }}
.vs-persona h4 {{ margin: 0; color: {TEXT}; font-size: 1rem; }}
.vs-persona .meta {{ color: {FAINT}; font-size: .78rem; margin: .1rem 0 .45rem; }}
.vs-persona .bio {{ color: {MUTED}; font-size: .84rem; line-height: 1.4; }}
.vs-chip {{ display: inline-block; font-size: .7rem; padding: .12rem .5rem; margin: .35rem .25rem 0 0; border-radius: 999px;
  border: 1px solid rgba(255,255,255,.12); color: {MUTED}; }}
.vs-meter {{ height: 5px; border-radius: 4px; background: {GRID}; margin-top: .25rem; overflow: hidden; }}
.vs-meter > span {{ display: block; height: 100%; background: linear-gradient(90deg,#f58529,#dd2a7b,#8134af);
  transform-origin: left; animation: vsGrow 1.1s cubic-bezier(.2,.8,.2,1) both; }}
/* Avatar strip */
.vs-avatars {{ display: flex; flex-wrap: wrap; gap: 10px; align-items: center; }}
.vs-avatars span {{ width: 46px; height: 46px; border-radius: 14px; display: grid; place-items: center; font-size: 1.4rem;
  background: linear-gradient(135deg, rgba(245,133,41,.22), rgba(129,52,175,.3)); border: 1px solid rgba(255,255,255,.1);
  animation: vsFadeUp .5s ease both; transition: transform .2s ease; cursor: default; }}
.vs-avatars span:hover {{ transform: translateY(-5px) scale(1.12); }}
/* A/B winner */
.vs-winner {{ text-align: center; padding: 1.1rem; border-radius: 18px; background: linear-gradient(135deg, rgba(25,158,112,.18), rgba(81,91,212,.14));
  border: 1px solid rgba(25,158,112,.5); animation: vsFadeUp .5s ease both, vsPulse 2.4s ease-in-out 2; --glow: rgba(25,158,112,.4); }}
.vs-winner .big {{ font-size: 1.7rem; font-weight: 800; color: {TEXT}; }}
[data-testid="stTabs"] button[role="tab"] {{ font-weight: 600; }}
/* Page entry */
[data-testid="stMainBlockContainer"] {{ animation: vsFadeUp .45s ease both; }}
/* Live dot */
.vs-live {{ width: 8px; height: 8px; border-radius: 50%; background: #199e70; display: inline-block;
  box-shadow: 0 0 0 0 rgba(25,158,112,.6); animation: vsLive 2s ease-out infinite; }}
@keyframes vsLive {{ 0% {{ box-shadow: 0 0 0 0 rgba(25,158,112,.6); }} 100% {{ box-shadow: 0 0 0 9px rgba(25,158,112,0); }} }}
/* Onboarding steps */
.vs-steps {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 12px; margin-top: .4rem; }}
.vs-stepcard {{ background: rgba(23,22,29,.6); border: 1px dashed rgba(255,255,255,.12); border-radius: 16px; padding: 1rem;
  animation: vsFadeUp .5s ease both; }}
.vs-stepcard .num {{ width: 28px; height: 28px; border-radius: 9px; display: grid; place-items: center; font-weight: 800;
  font-size: .85rem; color: #fff; background: {BRAND}; margin-bottom: .5rem; }}
.vs-stepcard h4 {{ margin: 0 0 .25rem; color: {TEXT}; font-size: .95rem; }}
.vs-stepcard p {{ margin: 0; color: {MUTED}; font-size: .82rem; line-height: 1.45; }}
.vs-chiprow {{ display: flex; flex-wrap: wrap; gap: 6px; margin-top: .7rem; }}
.vs-chiprow span {{ font-size: .74rem; padding: .25rem .6rem; border-radius: 999px; color: {MUTED};
  background: rgba(255,255,255,.05); border: 1px solid rgba(255,255,255,.08); }}
[data-testid="stFileUploaderDropzone"] {{ border-radius: 14px; }}
div[data-testid="stForm"] {{ border-radius: 16px; border-color: {GRID}; background: {SURFACE}; }}
@media (prefers-reduced-motion: reduce) {{ *, *::before, *::after {{ animation: none !important; transition: none !important; }} }}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def hero(title: str, subtitle: str, eyebrow: str = "") -> None:
    eb = f'<div class="vs-eyebrow">{html.escape(eyebrow)}</div>' if eyebrow else ""
    st.markdown(f'<div class="vs-hero">{eb}<h1>{html.escape(title)}</h1><p>{html.escape(subtitle)}</p></div>',
                unsafe_allow_html=True)


def demo_banner(cfg: dict) -> None:
    if cfg.get("demo"):
        st.markdown('<div class="vs-banner">⚠ SYNTHETIC DATA. These charts come from a pipeline smoke test on '
                    'generated posts. They show that the code runs, not how well ViralSense works.</div>',
                    unsafe_allow_html=True)


def card(title: str, sub: str = "") -> None:
    s = f'<div class="vs-sub">{html.escape(sub)}</div>' if sub else ""
    st.markdown(f'<div class="vs-card"><h3>{html.escape(title)}</h3>{s}</div>', unsafe_allow_html=True)


def section(title: str, sub: str = "") -> None:
    s = f'<div style="color:{MUTED};font-size:.88rem;margin:-.2rem 0 .5rem">{html.escape(sub)}</div>' if sub else ""
    st.markdown(f'<h3 style="margin:1.2rem 0 .3rem;font-weight:700">{html.escape(title)}</h3>{s}', unsafe_allow_html=True)


def empty(what: str, command: str) -> None:
    st.markdown(f'<div class="vs-empty">No {html.escape(what)} yet. Run <code>{html.escape(command)}</code> '
                'to generate it.</div>', unsafe_allow_html=True)


def badge(label: str) -> None:
    c = CLASS_COLORS.get(label, SERIES[0])
    st.markdown(f'<span class="vs-badge" style="--c:{c}"><span class="dot"></span>{html.escape(label)}</span>',
                unsafe_allow_html=True)


def verdict(text: str, good: bool) -> None:
    c = CLASS_COLORS["Viral"] if good else "#c98500"
    icon = "✓" if good else "!"
    st.markdown(f'<div class="vs-verdict" style="--c:{c}"><b>{icon}</b> {html.escape(text)}</div>', unsafe_allow_html=True)


def kpis(items: list[dict], height: int = 138) -> None:
    """Count-up stat tiles. items: {label, value (float), fmt: 'pct'|'num3'|'int', note}."""
    def _num(v):
        try:
            v = float(v)
        except (TypeError, ValueError):
            return None
        return None if v != v else v

    payload = json.dumps([{**it, "value": _num(it.get("value")), "note": str(it.get("note", ""))} for it in items])
    st.iframe(f"""<!doctype html><html><head><meta charset="utf-8"><style>html,body{{margin:0;background:transparent}}</style></head><body>
<div id="k" style="display:grid;grid-template-columns:repeat({len(items)},1fr);gap:12px;font-family:Inter,system-ui,sans-serif"></div>
<script>
const items = {payload};
const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
const fmt = (v, f) => f === 'pct' ? (v*100).toFixed(1) + '%' : f === 'int' ? Math.round(v).toLocaleString() : v.toFixed(3);
const root = document.getElementById('k');
items.forEach((it, i) => {{
  const d = document.createElement('div');
  d.style.cssText = `background:{SURFACE};border:1px solid {GRID};border-radius:16px;padding:14px 16px;opacity:0;
    transform:translateY(10px);transition:all .5s ease ${{i*90}}ms`;
  d.innerHTML = `<div style="font-size:12px;color:{FAINT};font-weight:600;letter-spacing:.04em">${{it.label}}</div>
    <div class="v" style="font-size:30px;font-weight:800;color:{TEXT};letter-spacing:-.02em;margin-top:4px">–</div>
    <div style="font-size:11.5px;color:{MUTED};margin-top:2px">${{it.note || ''}}</div>`;
  root.appendChild(d);
  requestAnimationFrame(() => {{ d.style.opacity = 1; d.style.transform = 'none'; }});
  const el = d.querySelector('.v');
  if (it.value === null || isNaN(it.value)) {{ el.textContent = 'n/a'; return; }}
  if (reduce) {{ el.textContent = fmt(it.value, it.fmt); return; }}
  const t0 = performance.now(), dur = 1100 + i*120;
  const tick = now => {{ const p = Math.min(1, (now - t0)/dur), e = 1 - Math.pow(1-p, 3);
    el.textContent = fmt(it.value * e, it.fmt); if (p < 1) requestAnimationFrame(tick); }};
  requestAnimationFrame(tick);
}});
</script></body></html>""", height=height)


def status_card(cfg: dict) -> None:
    """Sidebar card: which model is live and how good it is (read from reports/)."""
    from pathlib import Path

    t = Path(cfg["paths"]["reports"]) / "tables"
    rows = []
    try:
        sel = json.loads((t / "classification_selected.json").read_text())
        ts = sel["test_selected"]
        rows += [("Model", MODEL_NAMES.get(sel["best_cv"]["model"], sel["best_cv"]["model"])),
                 ("Macro-F1 (unseen accounts)", f"{ts['macro_f1']:.3f}"), ("Viral PR-AUC", f"{ts['pr_auc_viral']:.3f}")]
    except (OSError, KeyError, ValueError):
        rows.append(("Model", "not trained yet"))
    try:
        import pandas as pd

        jp = pd.read_csv(t / "jury_progress.csv").iloc[0]
        rows.append(("Jury", f"12 agents · {int(jp.complete_posts):,} posts"))
    except (OSError, KeyError, ValueError, IndexError):
        pass
    items = "".join(f'<div style="display:flex;justify-content:space-between;gap:.5rem;font-size:.78rem;padding:.18rem 0">'
                    f'<span style="color:{FAINT}">{html.escape(k)}</span><b style="color:{TEXT};text-align:right">{html.escape(v)}</b></div>'
                    for k, v in rows)
    st.markdown(f'<div class="vs-glass" style="padding:.75rem .85rem;margin-top:.4rem">'
                f'<div class="vs-label" style="display:flex;align-items:center;gap:.4rem"><span class="vs-live"></span>Live model</div>'
                f'{items}</div><div style="font-size:.72rem;color:{FAINT};margin-top:.7rem;line-height:1.5">'
                f'Ayush Upadhyay · R Rishita · Avantika Gupta<br/>Local AI only · no external APIs</div>', unsafe_allow_html=True)


def onboarding(steps: list[tuple[str, str]], chips: list[str]) -> None:
    """Empty-state guide shown before the user runs anything."""
    cards = "".join(f'<div class="vs-stepcard" style="animation-delay:{i * 0.08:.2f}s"><div class="num">{i + 1}</div>'
                    f'<h4>{html.escape(t)}</h4><p>{html.escape(d)}</p></div>' for i, (t, d) in enumerate(steps))
    chip = "".join(f"<span>{html.escape(c)}</span>" for c in chips)
    st.markdown(f'<div class="vs-steps">{cards}</div><div class="vs-chiprow">{chip}</div>', unsafe_allow_html=True)
