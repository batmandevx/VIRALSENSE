"""ViralSense dashboard.

    .venv/bin/streamlit run app/streamlit_app.py
    VIRALSENSE_CONFIG=path/to/config.yaml .venv/bin/streamlit run app/streamlit_app.py   # alternate run
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import viralsense  # noqa: E402,F401  (sets OMP_NUM_THREADS before torch/xgboost load)
import streamlit as st  # noqa: E402

from ui import analyze, compare, explore, home, insights, jury_page, method  # noqa: E402
from ui.theme import inject_css  # noqa: E402
from viralsense.utils import load_config  # noqa: E402

st.set_page_config(page_title="ViralSense", page_icon="✦", layout="wide")
inject_css()
cfg = load_config(os.environ.get("VIRALSENSE_CONFIG"))

pages = {
    "home": st.Page(lambda: home.page(cfg), title="Overview", icon=":material/space_dashboard:", url_path="home", default=True),
    "analyse": st.Page(lambda: analyze.page(cfg), title="Analyse a post", icon=":material/auto_awesome:", url_path="analyse"),
    "compare": st.Page(lambda: compare.page(cfg), title="A/B compare", icon=":material/compare:", url_path="compare"),
    "jury": st.Page(lambda: jury_page.page(cfg), title="Meet the jury", icon=":material/groups:", url_path="jury"),
    "explore": st.Page(lambda: explore.page(cfg), title="Explore predictions", icon=":material/grid_view:", url_path="explore"),
    "insights": st.Page(lambda: insights.page(cfg), title="Model insights", icon=":material/insights:", url_path="insights"),
    "method": st.Page(lambda: method.page(cfg), title="How it works", icon=":material/account_tree:", url_path="method"),
}
st.session_state["pages"] = pages
nav = st.navigation({
    "": [pages["home"]],
    "Create": [pages["analyse"], pages["compare"]],
    "Understand": [pages["jury"], pages["explore"], pages["insights"], pages["method"]],
})
with st.sidebar:
    st.markdown("### ✦ ViralSense")
    st.caption("Pre-publication virality prediction for Instagram posts.")
    st.caption("Ayush Upadhyay · R Rishita · Avantika Gupta")
nav.run()
