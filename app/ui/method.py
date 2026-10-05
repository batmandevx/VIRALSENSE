"""Page: how ViralSense works, what it guarantees, and its limits."""
import streamlit as st

from ui import charts
from ui.theme import demo_banner, hero, section

STEPS = [
    ("01", "Sample", "~15k single-image posts from 500+ influencer accounts, stratified by follower count."),
    ("02", "Label", "Engagement rate = (likes + comments) / followers. Within each follower tier: bottom 60% Low, next 30% Moderate, top 10% Viral."),
    ("03", "Features", "CLIP image embedding + zero-shot concepts, MiniLM caption embedding (PCA 32 on train folds), image/caption stats, time, tier, 7 strictly-past account-history signals."),
    ("04", "Persona jury", "12 simulated viewers score 1,500 posts on 5 factors via a local vision LLM. They never see any numbers."),
    ("05", "Models", "LogReg, RF, XGBoost, LightGBM, HistGB, voting, stacking (class weights vs SMOTE) + Optuna tuning; XGBoost regression; K-Means."),
    ("06", "Recommend", "LinUCB + Thompson-sampling bandits for time and caption style, LLM caption variants, a suggestion engine."),
]


CATS = ["Data", "Features", "AI jury", "Models", "Explain & evaluate", "Recommend", "App"]
NODES = [
    {"id": "Instagram Influencer Dataset", "x": 9, "y": 55, "cat": "Data", "size": 26, "label_pos": "top",
     "desc": "Kim et al., WWW 2020: 33,935 influencers, 10.1M posts. Post JSON read in place from the split zip; only sampled images range-fetched."},
    {"id": "Sample + labels", "x": 17, "y": 55, "cat": "Data", "size": 22,
     "desc": "15,000 single-image posts from 633 accounts, stratified by followers. Within each follower tier: bottom 60% Low, next 30% Moderate, top 10% Viral."},
    {"id": "Account split", "x": 17, "y": 22, "cat": "Data",
     "desc": "Held-out test accounts + GroupKFold(5). No account in both train and test (tested)."},
    {"id": "Image", "x": 32, "y": 88, "cat": "Features",
     "desc": "CLIP ViT-B/32 embedding → PCA 32 (fit on train folds), zero-shot concepts (selfie, food, pet…), brightness, contrast, colourfulness, faces."},
    {"id": "Caption", "x": 32, "y": 68, "cat": "Features",
     "desc": "MiniLM sentence embedding → PCA 32, length, emoji, hashtags, question mark, VADER sentiment."},
    {"id": "Timing", "x": 32, "y": 48, "cat": "Features", "desc": "Hour (cyclic), weekday (UTC)."},
    {"id": "Account history", "x": 32, "y": 28, "cat": "Features",
     "desc": "From strictly earlier posts only: median, recent median, volatility, trend, days since last post, posts in 30 days."},
    {"id": "12 persona agents", "x": 46, "y": 95, "cat": "AI jury", "size": 22,
     "desc": "Local vision LLM (qwen2.5-VL 3B) role-plays 12 audience personas. Each rates scroll-stop, emotional pull, shareability, save intent and comment trigger from 1 to 10, never seeing numbers."},
    {"id": "Feature matrix", "x": 48, "y": 55, "cat": "Features", "size": 24,
     "desc": "~110 inputs per post. A leakage guard rejects any likes/comments-derived column except strictly-past history."},
    {"id": "Classifiers", "x": 63, "y": 68, "cat": "Models", "size": 24,
     "desc": "LogReg, RF, XGBoost, LightGBM, HistGB, soft voting and stacking × class weights vs SMOTE (inside the pipeline), plus Optuna-tuned XGB/LGBM."},
    {"id": "Regressor", "x": 63, "y": 42, "cat": "Models", "desc": "XGBoost on log engagement rate → engagement score (percentile within tier)."},
    {"id": "K-Means", "x": 63, "y": 20, "cat": "Models", "desc": "Content clusters, k chosen by silhouette."},
    {"id": "Ablation", "x": 63, "y": 92, "cat": "Explain & evaluate",
     "desc": "Metadata vs + content vs + jury on the 1,500-post jury subset, paired t-test across folds."},
    {"id": "SHAP", "x": 78, "y": 80, "cat": "Explain & evaluate", "desc": "TreeSHAP on the deployed model, with embedding components grouped."},
    {"id": "Controls", "x": 78, "y": 60, "cat": "Explain & evaluate",
     "desc": "Dummy floor, history-only baseline, time-ordered test, Kaggle negative control, leakage demo, plausibility stop at 0.85."},
    {"id": "Posting-time bandit", "x": 78, "y": 38, "cat": "Recommend",
     "desc": "LinUCB, linear and Beta Thompson sampling over 8 slots; replay evaluation with bootstrap CIs."},
    {"id": "Caption tools", "x": 78, "y": 16, "cat": "Recommend",
     "desc": "Caption-style bandit, 3 LLM rewrites rescored by the classifier, suggestion engine over times and edits."},
    {"id": "ViralSense app", "x": 95, "y": 50, "cat": "App", "size": 28,
     "desc": "Overview · Analyse · A/B compare · Meet the jury · Explore · Model insights · How it works."},
]
EDGES = [("Instagram Influencer Dataset", "Sample + labels"), ("Sample + labels", "Account split"),
         ("Sample + labels", "Image"), ("Sample + labels", "Caption"), ("Sample + labels", "Timing"),
         ("Sample + labels", "Account history"), ("Sample + labels", "12 persona agents"),
         ("Image", "Feature matrix"), ("Caption", "Feature matrix"), ("Timing", "Feature matrix"),
         ("Account history", "Feature matrix"), ("Account split", "Feature matrix"),
         ("Feature matrix", "Classifiers"), ("Feature matrix", "Regressor"), ("Feature matrix", "K-Means"),
         ("12 persona agents", "Ablation"), ("Feature matrix", "Ablation"),
         ("Classifiers", "SHAP"), ("Classifiers", "Controls"), ("Feature matrix", "Posting-time bandit"),
         ("Classifiers", "Caption tools"), ("SHAP", "ViralSense app"), ("Controls", "ViralSense app"),
         ("Regressor", "ViralSense app"), ("Posting-time bandit", "ViralSense app"), ("Caption tools", "ViralSense app"),
         ("Ablation", "ViralSense app"), ("K-Means", "ViralSense app"), ("12 persona agents", "ViralSense app")]


def page(cfg: dict) -> None:
    hero("How it works", "A pre-publication pipeline built so that nothing the model sees could only be known after posting.",
         "method")
    demo_banner(cfg)
    section("System architecture", "Hover any node for details; click a legend item to hide or show that stage. "
            "The moving dots show data flowing through the pipeline.")
    charts.flow(NODES, EDGES, CATS, height=640)
    section("Pipeline in six steps")
    st.markdown('<div class="vs-flow">' + "".join(
        f'<div class="vs-step"><div class="n">STEP {n}</div><h4>{t}</h4><p>{d}</p></div>' for n, t, d in STEPS) + "</div>",
        unsafe_allow_html=True)

    section("Built by the team", "Ayush Upadhyay (23BAI1231) · R Rishita (24BAI1632) · Avantika Gupta (24BAI1633)")
    st.markdown("""
These ideas come from the team's Colab notebook (Kaggle *Instagram Analytics*) and were rebuilt on the real dataset:
- **Suggestion engine:** searches posting times and caption edits, re-scores each, and ranks them by gain in P(Viral).
- **Thompson sampling and bootstrap CIs:** a Bayesian bandit next to LinUCB, with every policy judged by a 95% CI on lift.
- **Caption-style bandit:** arms are caption length × hashtags × question.
- **More models:** LightGBM, HistGradientBoosting, soft voting and a dummy floor, alongside LogReg, RF, XGBoost and stacking.
- **Time-ordered test:** older posts train, newer posts test.
- **Negative control:** the notebook's Kaggle dataset, where the pipeline correctly lands at chance.
- **Leakage demonstration:** what happens when likes and comments are allowed in.
""")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        section("Guarantees, enforced by tests")
        st.markdown("""
- **No account appears in both train and test**, or in two CV folds (GroupKFold plus a held-out set of unseen accounts).
- **No likes, comments or labels reach the features.** A name-based guard plus a check of which columns the pipeline actually reads.
- **Account history uses strictly earlier posts only.** It is checked against a brute-force reference.
- **PCA, scaling, imputation and SMOTE are fit on training folds only.** They live inside the model pipeline.
- **The jury prompt contains no follower or engagement numbers.** Every response is cached per (post, persona).
- **Plausibility stop:** results above macro-F1 0.85 or Viral PR-AUC 0.85 halt the run for a leakage check.
""")
    with c2:
        section("Limitations")
        st.markdown("""
- **Follower counts are a crawl-time snapshot**, not the count when each post went live.
- **Times are UTC.** The creator's local time is unknown.
- **"Viral" means top 10% of a tier in this dataset**, not reach in today's Instagram feed.
- **The posting-time replay is biased**, because creators chose when to post. It compares policies; it does not prove uplift.
- **Caption uplift is the classifier's opinion**, not measured engagement.
- **The persona jury is simulated** and is only kept if the ablation shows it helps.
- **Account history dominates.** On training accounts, the account's past median engagement has Spearman ρ 0.67
  with the label, while each content feature alone is below 0.06. The baselines separate the two.
- **Older posts skew Low.** 2012–2016 posts are measured against 2019 follower counts (8% of the sample).
""")
