"""Page: every result the pipeline produced, read from reports/tables (nothing is computed from thin air)."""
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import confusion_matrix, precision_recall_curve

from ui import charts
from ui.theme import CLASS_COLORS, MODEL_NAMES, SERIES, demo_banner, empty, hero, kpis, section, verdict

CLASSES = ["Low", "Moderate", "Viral"]


def _csv(cfg, name):
    p = Path(cfg["paths"]["reports"]) / "tables" / name
    return pd.read_csv(p) if p.exists() else None


def page(cfg: dict) -> None:
    hero("Model insights", "How well ViralSense predicts on accounts it has never seen, what drives it, and whether the "
         "persona jury earns its cost. Every number here is read from reports/.", "evaluation")
    demo_banner(cfg)
    tabs = st.tabs(["Overview", "Per class", "Explainability", "Persona jury", "Clusters", "Posting time",
                    "Caption style", "Robustness & controls", "Data"])
    with tabs[0]:
        overview(cfg)
    with tabs[1]:
        per_class(cfg)
    with tabs[2]:
        explain(cfg)
    with tabs[3]:
        jury(cfg)
    with tabs[4]:
        clusters(cfg)
    with tabs[5]:
        posting(cfg)
    with tabs[6]:
        caption_style(cfg)
    with tabs[7]:
        robustness(cfg)
    with tabs[8]:
        data(cfg)


def overview(cfg):
    test, cv, reg = _csv(cfg, "classification_test.csv"), _csv(cfg, "classification_cv_summary.csv"), _csv(cfg, "regression_summary.csv")
    if test is None or cv is None:
        empty("classification results", "python -m viralsense.pipeline --stages classify")
        return
    sel = test[test.selected_by_cv].iloc[0]
    kpis([
        {"label": "Macro-F1 (test)", "value": sel.macro_f1, "fmt": "num3", "note": f"{sel.model} + {sel.imbalance}; chance ≈ 0.33"},
        {"label": "Balanced accuracy", "value": sel.balanced_acc, "fmt": "pct", "note": "mean per-class recall"},
        {"label": "Viral recall", "value": sel.recall_Viral, "fmt": "pct", "note": "share of Viral posts caught"},
        {"label": "Viral PR-AUC", "value": sel.pr_auc_viral, "fmt": "num3", "note": f"baseline = prevalence {sel.viral_prevalence:.3f}"},
        {"label": "Spearman ρ (regression)", "value": None if reg is None else reg.test_spearman.iloc[0], "fmt": "num3",
         "note": "log engagement rate, test"},
    ])
    section("Cross-validated macro-F1", "Five folds split by account. Bars show the mean; whiskers show ± 1 std. Class weights vs SMOTE.")
    cv = cv.copy()
    models = [m for m in ["logreg", "rf", "xgb", "lgbm", "hgb", "vote", "stack", "xgb_tuned", "lgbm_tuned"] if m in set(cv.model)]
    groups = {}
    for imb, label in (("class_weight", "class weights"), ("smote", "SMOTE")):
        g = cv[cv.imbalance == imb].set_index("model").reindex(models)
        groups[label] = (g.macro_f1_mean.tolist(), g.macro_f1_std.tolist())
    charts.grouped_bars_err([MODEL_NAMES[m] for m in models], groups, height=360, yname="macro-F1")
    base = cv[cv.model.isin(["dummy", "xgb [history only]", "xgb [no history]"])].set_index("model")
    if len(base):
        st.caption(" · ".join(f"{MODEL_NAMES.get(k, k)}: {r.macro_f1_mean:.3f} ± {r.macro_f1_std:.3f}" for k, r in base.iterrows()))
    hist = _csv(cfg, "tuning_history.csv")
    if hist is not None:
        section("Hyperparameter search (Optuna, TPE)", "Each trial = mean macro-F1 over 3 account-grouped folds of the "
                "training accounts. Lines show the best score found so far; the test set is never used.")
        t1, t2 = st.columns([1.4, 1], gap="large")
        with t1:
            xs, ys = {}, {}
            for mdl, g in hist.groupby("model"):
                g = g.sort_values("trial")
                xs[f"{MODEL_NAMES.get(mdl, mdl)} best so far"] = g.trial.to_numpy()
                ys[f"{MODEL_NAMES.get(mdl, mdl)} best so far"] = g.macro_f1.cummax().to_numpy()
                xs[f"{MODEL_NAMES.get(mdl, mdl)} trial"] = g.trial.to_numpy()
                ys[f"{MODEL_NAMES.get(mdl, mdl)} trial"] = g.macro_f1.to_numpy()
            lo = float(hist.macro_f1.min())
            charts.lines(xs, ys, height=320, xname="trial", yname="CV macro-F1", ymin=round(lo - 0.005, 3))
        with t2:
            imps = [x for x in (_csv(cfg, f"tuning_param_importance_{k}.csv") for k in ("xgb", "lgbm")) if x is not None]
            if imps:
                imp = pd.concat(imps)
                imp["label"] = imp.model.map(lambda m: "XGB" if m == "xgb" else "LGBM") + " · " + imp.param
                imp = imp.sort_values("importance").tail(10)
                charts.hbars(imp.label.tolist(), imp.importance.tolist(), color=SERIES[6], height=320,
                             xname="fANOVA importance")
    section("Held-out test accounts", "Each model uses the imbalance strategy that won in CV. The selected row was picked on CV, not on test.")
    show = test[["model", "imbalance", "selected_by_cv", "macro_f1", "balanced_acc", "recall_Low", "recall_Moderate",
                 "recall_Viral", "pr_auc_viral"]].copy()
    st.dataframe(show.style.format({c: "{:.3f}" for c in show.columns if show[c].dtype.kind == "f"})
                 .highlight_max(subset=["macro_f1", "balanced_acc", "pr_auc_viral"], color="#1f3b2f"),
                 hide_index=True, width="stretch")


def per_class(cfg):
    preds, test = _csv(cfg, "classification_test_predictions.csv"), _csv(cfg, "classification_test.csv")
    if preds is None:
        empty("test predictions", "python -m viralsense.pipeline --stages classify")
        return
    models = list(dict.fromkeys(preds.model))
    default = test[test.selected_by_cv].model.iloc[0] if test is not None else models[0]
    m = st.segmented_control("Model", models, default=default, key="pc_model") or default
    p = preds[preds.model == m]
    proba = p[[f"p_{c}" for c in CLASSES]].to_numpy()
    c1, c2 = st.columns(2, gap="large")
    with c1:
        section("Confusion matrix", "Row-normalised: each row shows where that true class ends up.")
        cm = confusion_matrix(p.y, proba.argmax(1), labels=[0, 1, 2], normalize="true")
        charts.heatmap([f"true {c}" for c in CLASSES], [f"pred {c}" for c in CLASSES], cm, 0, 1, height=340)
    with c2:
        section("Recall per class", "All four models, held-out test accounts.")
        if test is not None:
            groups = {c: (test[f"recall_{c}"].tolist(), [np.nan] * len(test)) for c in CLASSES}
            charts.grouped_bars_err(test.model.tolist(), groups, height=340, ymax=1, colors=[CLASS_COLORS[c] for c in CLASSES])
    # At most 5 curves (the categorical palette holds 8): the strongest models by test macro-F1, always incl. the selected one.
    if test is not None:
        ranked = test[test.model.isin(models)].sort_values("macro_f1", ascending=False).model.drop_duplicates().tolist()
        top = list(dict.fromkeys([default] + [r for r in ranked if r != "dummy"]))[:5]
    else:
        top = models[:5]
    section("Viral precision-recall", "The 5 strongest models on test. The dashed line is the Viral prevalence, i.e. a random ranking.")
    xs, ys = {}, {}
    for name in top:
        q = preds[preds.model == name]
        prec, rec, _ = precision_recall_curve(q.y == 2, q.p_Viral)
        idx = np.linspace(0, len(rec) - 1, min(200, len(rec))).astype(int)
        xs[name], ys[name] = rec[idx], prec[idx]
    charts.lines(xs, ys, height=360, xname="recall (Viral)", yname="precision", ref=("chance", (preds[preds.model == models[0]].y == 2).mean()), ymax=1)

    from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, roc_curve

    section("Threshold explorer", f"Flag a post as Viral when P(Viral) ≥ threshold. Model: {MODEL_NAMES.get(m, m)}.")
    yv = (p.y == 2).to_numpy()
    t = st.slider("Threshold on P(Viral)", 0.05, 0.95, 0.5, 0.01, key="thr")
    flag = p.p_Viral.to_numpy() >= t
    kpis([
        {"label": "Precision", "value": precision_score(yv, flag, zero_division=0), "fmt": "pct", "note": "flagged posts that are Viral"},
        {"label": "Recall", "value": recall_score(yv, flag, zero_division=0), "fmt": "pct", "note": "Viral posts caught"},
        {"label": "F1 (Viral)", "value": f1_score(yv, flag, zero_division=0), "fmt": "num3", "note": "balance of the two"},
        {"label": "Posts flagged", "value": flag.mean(), "fmt": "pct", "note": f"{int(flag.sum()):,} of {len(flag):,}"},
    ], height=130)
    grid = np.linspace(0.05, 0.95, 91)
    curves = {"precision": [], "recall": [], "F1": []}
    for g in grid:
        fl = p.p_Viral.to_numpy() >= g
        curves["precision"].append(precision_score(yv, fl, zero_division=0))
        curves["recall"].append(recall_score(yv, fl, zero_division=0))
        curves["F1"].append(f1_score(yv, fl, zero_division=0))
    charts.lines(grid, curves, height=280, xname="threshold", yname="score", ymax=1)

    c3, c4 = st.columns(2, gap="large")
    with c3:
        section("ROC: Viral vs rest", "Same 5 models; area under the curve in the legend. The diagonal is a random ranking.")
        xs, ys = {}, {}
        for name in top:
            q = preds[preds.model == name]
            fpr, tpr, _ = roc_curve(q.y == 2, q.p_Viral)
            idx = np.linspace(0, len(fpr) - 1, min(200, len(fpr))).astype(int)
            label = f"{MODEL_NAMES.get(name, name)} ({roc_auc_score(q.y == 2, q.p_Viral):.3f})"
            xs[label], ys[label] = fpr[idx], tpr[idx]
        xs["random"], ys["random"] = np.array([0, 1]), np.array([0, 1])
        charts.lines(xs, ys, height=360, xname="false positive rate", yname="true positive rate", ymax=1)
    with c4:
        section("Calibration", "Mean predicted P(Viral) vs the real Viral share, in 10 bins. On the diagonal = honest probabilities.")
        bins = np.clip((p.p_Viral.to_numpy() * 10).astype(int), 0, 9)
        cal = pd.DataFrame({"b": bins, "p": p.p_Viral, "y": yv}).groupby("b").agg(p=("p", "mean"), y=("y", "mean"), n=("y", "size"))
        cal = cal[cal.n >= 10]
        charts.lines({"model": cal.p.to_numpy(), "perfect": np.array([0, 1])},
                     {"model": cal.y.to_numpy(), "perfect": np.array([0, 1])}, height=360,
                     xname="mean predicted P(Viral)", yname="observed Viral share", ymax=1)

    calsum, rel = _csv(cfg, "calibration_summary.csv"), _csv(cfg, "calibration_reliability.csv")
    if calsum is not None and rel is not None:
        import json as _json

        section("Calibration fix and decision rule (deployed model)", "Isotonic calibration and per-class decision weights "
                "learned from out-of-fold predictions on training accounts; the test accounts are used only for this report.")
        k1, k2 = st.columns([1.1, 1], gap="large")
        with k1:
            xs, ys = {}, {}
            for v, g in rel.groupby("variant"):
                xs[v], ys[v] = g.mean_pred.to_numpy(), g.observed.to_numpy()
            xs["perfect"], ys["perfect"] = np.array([0, 1]), np.array([0, 1])
            charts.lines(xs, ys, height=320, xname="predicted P(Viral)", yname="observed Viral share", ymax=1)
        with k2:
            show = calsum[["variant", "macro_f1", "balanced_acc", "recall_Viral", "brier_multiclass", "ece_viral"]]
            st.dataframe(show.round(4), hide_index=True, width="stretch")
            dr = Path(cfg["paths"]["reports"]) / "tables" / "decision_rule.json"
            if dr.exists():
                w = _json.loads(dr.read_text())["weights"]
                st.caption("Decision rule: predict argmax(w × p) with w = " + ", ".join(f"{k} {v:.1f}" for k, v in w.items())
                           + ". Lower Brier score and ECE mean more honest probabilities.")

    c5, c6 = st.columns(2, gap="large")
    with c5:
        section("Cumulative gains", "Review the posts with the highest P(Viral) first: what share of all Viral posts do you catch?")
        order = np.argsort(-p.p_Viral.to_numpy())
        caught = np.cumsum(yv[order]) / max(1, yv.sum())
        frac = np.arange(1, len(order) + 1) / len(order)
        idx = np.linspace(0, len(order) - 1, 200).astype(int)
        charts.lines({"model": frac[idx], "random": np.array([0, 1])}, {"model": caught[idx], "random": np.array([0, 1])},
                     height=320, xname="share of posts reviewed", yname="share of Viral posts caught", ymax=1)
    with c6:
        tiers = _posts(cfg["paths"]["processed"])
        if tiers is not None:
            section("Macro-F1 by follower tier", "Does the model work equally well for small and large accounts?")
            q = p.astype({"post_id": str}).merge(tiers[["post_id", "follower_tier"]].astype({"post_id": str}), on="post_id")
            q["pred"] = q[[f"p_{c}" for c in CLASSES]].to_numpy().argmax(1)
            per = q.groupby("follower_tier").apply(lambda g: f1_score(g.y, g.pred, average="macro"), include_groups=False)
            charts.hbars([f"tier {int(t) + 1}" for t in per.index], per.tolist(), color=SERIES[2], height=260, xname="macro-F1")


def explain(cfg):
    imp = _csv(cfg, "shap_importance_viral.csv")
    if imp is None:
        empty("SHAP importances", "python -m viralsense.pipeline --stages classify")
        return
    section("What drives the Viral prediction", "Mean |SHAP| on the deployed XGBoost over test accounts. Embedding PCs are summed into one group each.")
    top = imp.head(15).iloc[::-1]
    charts.hbars(top.feature.tolist(), top.mean_abs_shap_viral.tolist(), color=SERIES[1], height=460, xname="mean |SHAP|")
    st.caption("SHAP shows what the model relies on, not what causes engagement.")


def jury(cfg):
    summ, folds = _csv(cfg, "ablation_summary.csv"), _csv(cfg, "ablation_folds.csv")
    usage = _csv(cfg, "jury_usage_summary.csv")
    if usage is not None:
        u = usage.iloc[0]
        kpis([
            {"label": "LLM calls", "value": u.calls, "fmt": "int", "note": f"{u.failed_calls} failed after retries"},
            {"label": "Input tokens", "value": u.input_tokens, "fmt": "int", "note": u.model},
            {"label": "Output tokens", "value": u.output_tokens, "fmt": "int", "note": f"cost ${u.usd:.2f} (local model)"},
            {"label": "Wall time, hours", "value": u.wall_hours, "fmt": "num3", "note": f"{u.sec_per_call:.1f} s per call"},
        ])
    if summ is None:
        empty("ablation results", "python -m viralsense.pipeline --stages jury ablation")
        return
    section("Does the jury help?", "Same model and folds on the 1,500-post jury subset. Bars show the mean; whiskers show ± 1 std across 5 folds.")
    sets = ["a_metadata", "b_metadata+content", "c_metadata+content+jury"]
    names = ["(a) metadata", "(b) + content", "(c) + jury"]
    metric = st.segmented_control("Metric", ["macro_f1", "pr_auc_viral", "balanced_acc"], default="macro_f1", key="abl_metric") or "macro_f1"
    groups = {}
    for model in summ.model.unique():
        g = summ[summ.model == model].set_index("feature_set").reindex(sets)
        groups[model] = (g[f"{metric}_mean"].tolist(), g[f"{metric}_std"].tolist())
    charts.grouped_bars_err(names, groups, height=340, yname=metric)
    vpath = Path(cfg["paths"]["reports"]) / "tables" / "ablation_verdict.txt"
    if vpath.exists():
        for line in vpath.read_text().strip().splitlines():
            verdict(line, "HELPS" in line)
    if folds is not None:
        with st.expander("Fold-by-fold gain from adding the jury"):
            xs, ys = {}, {}
            for model in folds.model.unique():
                piv = folds[folds.model == model].pivot(index="fold", columns="feature_set", values=metric)
                xs[model], ys[model] = piv.index.to_numpy(), (piv[sets[2]] - piv[sets[1]]).to_numpy()
            charts.lines(xs, ys, height=280, xname="fold", yname=f"Δ {metric} (c − b)", ref=("no change", 0))
    by_label, rho = _csv(cfg, "jury_scores_by_label.csv"), _csv(cfg, "jury_spearman_vs_engagement.csv")
    c1, c2 = st.columns(2, gap="large")
    if by_label is not None:
        with c1:
            section("Jury scores by true class", "Mean persona score (1–10), training accounts.")
            b = by_label.set_index(by_label.columns[0])
            cols = [c for c in b.columns if c.endswith("_mean") and c != "jury_overall_mean"]
            charts.radar([c.removeprefix("jury_").removesuffix("_mean") for c in cols],
                         {k: b.loc[k, cols].tolist() for k in CLASSES if k in b.index}, height=360)
    if rho is not None:
        with c2:
            section("Correlation with real engagement", "Spearman ρ between each jury score and log engagement rate.")
            r = rho.sort_values("spearman_vs_log_er")
            charts.hbars([x.removeprefix("jury_").replace("_", " ") for x in r.feature], r.spearman_vs_log_er.tolist(),
                         color=SERIES[4], height=340, xname="Spearman ρ")


def clusters(cfg):
    sil, prof, pts = _csv(cfg, "kmeans_silhouette.csv"), _csv(cfg, "kmeans_cluster_profiles.csv"), _csv(cfg, "kmeans_points.csv")
    if sil is None or prof is None:
        empty("clusters", "python -m viralsense.pipeline --stages cluster")
        return
    c1, c2 = st.columns([1, 1.4], gap="large")
    with c1:
        section("Choosing k", "Silhouette on training posts; the highest wins.")
        charts.lines(sil.k.to_numpy(), {"silhouette": sil.silhouette.to_numpy()}, height=320, xname="k", yname="silhouette")
    with c2:
        if pts is not None:
            section("Content map", "Training posts projected to 2-D, coloured by cluster. Scroll to zoom.")
            groups = {f"cluster {c}": g[["x", "y"]].to_numpy() for c, g in pts.groupby("cluster")}
            charts.scatter_groups(groups, height=360, xname="PC1", yname="PC2")
    section("Engagement mix per cluster", "Share of Low, Moderate and Viral posts in each content cluster.")
    cats = [f"cluster {c} · {n} posts" for c, n in zip(prof.cluster, prof.n_posts)]
    charts.stacked_share(cats, {c: prof[f"share_{c}"].tolist() for c in CLASSES}, height=60 + 44 * len(prof))
    section("Cluster profiles")
    st.dataframe(prof[["cluster", "n_posts", "n_accounts", "share_Viral", "median_engagement_rate", "description",
                       "top_hashtags", "top_words"]].style.format({"share_Viral": "{:.1%}", "median_engagement_rate": "{:.4f}"}),
                 hide_index=True, width="stretch")


def posting(cfg):
    summ, curves, logged = _csv(cfg, "bandit_replay_summary.csv"), _csv(cfg, "bandit_replay_curves.csv"), _csv(cfg, "bandit_logged_slots.csv")
    if summ is None:
        empty("bandit results", "python -m viralsense.pipeline --stages bandit")
        return
    st.markdown("Replay keeps only test posts published in the slot the policy picks. Creators chose their own times, "
                "so these numbers compare policies under that bias. They do not estimate a causal uplift.")
    c1, c2 = st.columns([1.3, 1], gap="large")
    with c1:
        if curves is not None:
            section("Replay: running mean reward", "Seed 0, test accounts in time order.")
            ref = summ[summ.policy.str.startswith("logged")].mean_reward.iloc[0]
            cols = [c for c in curves.columns if c != "event"]
            charts.lines(curves.event.to_numpy(), {POLICY_NAMES.get(c, c): curves[c].to_numpy() for c in cols}, height=340,
                         xname="test events streamed", yname="mean reward", ref=("logged", ref))
    with c2:
        _policy_forest(summ, "Lift over what creators did")
    if logged is not None:
        section("What creators actually did", "Posts and reward rate per UTC slot in the logs.")
        tr = logged[logged.split == "train"].sort_values("slot")
        c3, c4 = st.columns(2, gap="large")
        with c3:
            charts.hbars(tr.slot_label.tolist(), tr.n_posts.tolist(), color=SERIES[0], height=300, xname="posts", decimals=0)
        with c4:
            charts.hbars(tr.slot_label.tolist(), tr.mean_reward.tolist(), color=CLASS_COLORS["Viral"], height=300, pct=True,
                         xname="share Moderate or Viral")


POLICY_NAMES = {"linucb": "LinUCB (contextual)", "lin_ts": "Linear Thompson (contextual)", "thompson": "Thompson sampling",
                "eps_greedy": "ε-greedy", "best_fixed": "Best fixed arm", "random": "Random"}


def _policy_forest(summ: pd.DataFrame, title: str) -> None:
    section(title, "Mean reward on matched replay events minus the logged average, with a bootstrap 95% CI. "
            "Green dots: the interval excludes zero.")
    p = summ[summ.policy.isin(POLICY_NAMES)].copy()
    none = p[p.matched_events.fillna(0) == 0].policy.map(POLICY_NAMES).tolist()
    if none:
        st.caption(f"No estimate for {', '.join(none)}: during replay it never picked the arm a test post actually used.")
    p = p[p.matched_events.fillna(0) > 0]
    if "lift_ci_low" not in p:
        st.info("Re-run the bandit stage to compute confidence intervals.")
        return
    p = p.iloc[::-1]
    charts.forest([POLICY_NAMES[k] for k in p.policy], p.lift_vs_logged.tolist(), p.lift_ci_low.tolist(),
                  p.lift_ci_high.tolist(), height=320, xname="lift in reward rate")
    tbl = p.iloc[::-1][["policy", "mean_reward", "std_reward", "matched_events", "lift_vs_logged", "lift_ci_low", "lift_ci_high"]]
    st.dataframe(tbl.assign(policy=tbl.policy.map(POLICY_NAMES)).round(4), hide_index=True, width="stretch")


def caption_style(cfg):
    summ, curves, post = (_csv(cfg, n) for n in ("caption_bandit_summary.csv", "caption_bandit_curves.csv",
                                                  "caption_style_by_tier.csv"))
    if summ is None:
        empty("caption-style bandit results", "python -m viralsense.pipeline --stages caption_bandit")
        return
    st.markdown("Arms are caption styles: **length** (short / medium / long) × **hashtags** (0 / 1–5 / 6+) × "
                "**asks a question**. The context is what is fixed before writing the caption: the image, follower tier "
                "and account history. Reward: the post was Moderate or Viral.")
    verdict("Read lifts here as correlation, not cause. Accounts that write a given caption style also differ in other "
            "ways, and replay cannot separate the two. A policy 'winning' means that style co-occurs with engagement "
            "in this data, not that switching style will raise yours.", False)
    c1, c2 = st.columns([1.2, 1], gap="large")
    with c1:
        if curves is not None:
            section("Replay: running mean reward", "Seed 0, held-out accounts in time order.")
            ref = summ[summ.policy.str.startswith("logged")].mean_reward.iloc[0]
            cols = [c for c in curves.columns if c != "event"]
            charts.lines(curves.event.to_numpy(), {POLICY_NAMES.get(c, c): curves[c].to_numpy() for c in cols},
                         height=340, xname="test events streamed", yname="mean reward", ref=("logged", ref))
    with c2:
        _policy_forest(summ, "Lift over what creators did")
    if post is not None:
        section("Which styles do well, by follower tier", "Beta-posterior share of Moderate-or-Viral posts (training "
                "accounts). Cells with fewer than 20 posts are blank.")
        p = post[post.n >= 20]
        piv = p.pivot(index="arm", columns="follower_tier", values="posterior_mean")
        piv = piv.loc[piv.mean(axis=1).sort_values(ascending=False).index]
        charts.heatmap(list(piv.index), [f"tier {int(t) + 1}" for t in piv.columns], piv.to_numpy(),
                       float(np.nanmin(piv.values)), float(np.nanmax(piv.values)), height=60 + 28 * len(piv),
                       colors=("#1a1622", "#8134af", "#f58529"))


def robustness(cfg):
    temp, ctrl, leak = _csv(cfg, "temporal_robustness.csv"), _csv(cfg, "kaggle_negative_control.csv"), _csv(cfg, "leakage_demo.csv")
    section("Does it hold up on newer posts?", "Train on the oldest 80% of posts by date, test on the newest 20% "
            "(the team notebook's split). 'time + accounts' also keeps test accounts unseen.")
    if temp is None:
        empty("temporal check", "python -m viralsense.pipeline --stages temporal")
    else:
        designs = list(dict.fromkeys(temp.design))
        models = list(dict.fromkeys(temp.model))
        groups = {d: (temp[temp.design == d].set_index("model").reindex(models).macro_f1.tolist(), [np.nan] * len(models))
                  for d in designs}
        charts.grouped_bars_err(models, groups, height=300, yname="macro-F1 (newest 20%)")
    section("Negative control: the team's Kaggle dataset", "Same method on synthetic data where pre-publication features "
            "carry no signal. A working pipeline should land at the dummy baseline here, and it does.")
    if ctrl is None:
        empty("negative control", "python -m viralsense.pipeline --stages control")
    else:
        models = list(dict.fromkeys(ctrl.model))
        groups = {f: (ctrl[ctrl.features == f].set_index("model").reindex(models).cv_macro_f1_mean.tolist(),
                      ctrl[ctrl.features == f].set_index("model").reindex(models).cv_macro_f1_std.tolist())
                  for f in dict.fromkeys(ctrl.features)}
        charts.grouped_bars_err([MODEL_NAMES.get(m, m) for m in models], groups, height=300, yname="CV macro-F1", ymax=0.6)
    section("What leakage looks like", "A model that may see likes and comments. Not a valid pre-publication result.")
    if leak is not None:
        r = leak.iloc[0]
        verdict(f"Leaky model: macro-F1 {r.macro_f1:.3f}, Viral PR-AUC {r.pr_auc_viral:.3f}. Scores like this are why "
                "the pipeline stops for a leakage check above 0.85.", False)


@st.cache_data(show_spinner=False)
def _posts(processed: str):
    p = Path(processed) / "posts.parquet"
    if not p.exists():
        return None
    return pd.read_parquet(p, columns=["post_id", "split", "follower_tier", "label", "engagement_rate", "timestamp", "account"])


def data(cfg):
    posts = _posts(cfg["paths"]["processed"])
    if posts is None:
        empty("prepared data", "python -m viralsense.pipeline --stages prepare")
        return
    kpis([
        {"label": "Posts", "value": len(posts), "fmt": "int", "note": "single-image"},
        {"label": "Accounts", "value": posts.account.nunique(), "fmt": "int", "note": f"{posts[posts.split == 'test'].account.nunique()} held out for test"},
        {"label": "Viral share", "value": (posts.label == "Viral").mean(), "fmt": "pct", "note": "top 10% within tier"},
        {"label": "Median engagement", "value": posts.engagement_rate.median(), "fmt": "pct", "note": "(likes + comments) / followers"},
    ])
    c1, c2 = st.columns(2, gap="large")
    with c1:
        section("Split → tier → class", "Click a ring to zoom in.")
        tree = []
        for s, gs in posts.groupby("split"):
            kids = []
            for t, gt in gs.groupby("follower_tier"):
                kids.append({"name": f"tier {t + 1}", "children": [
                    {"name": c, "value": int((gt.label == c).sum()), "itemStyle": {"color": CLASS_COLORS[c]}} for c in CLASSES]})
            tree.append({"name": s, "children": kids, "itemStyle": {"color": "#3a3942" if s == "train" else "#5a4a2a"}})
        charts.sunburst(tree, height=420)
    with c2:
        section("Engagement rate by follower tier", "log10 scale. Larger accounts get lower rates, which is why labels are set per tier.")
        le = np.log10(posts.engagement_rate + 1e-6)
        bins = np.linspace(np.percentile(le, 0.5), np.percentile(le, 99.5), 50)
        charts.histogram({f"tier {t + 1}": le[posts.follower_tier == t] for t in sorted(posts.follower_tier.unique())}, bins,
                         height=420, xname="log10 engagement rate")
    section("When posts were published", "Share of posts by weekday and UTC hour.")
    ts = posts.timestamp.dt
    grid = pd.crosstab(ts.weekday, ts.hour, normalize="all").reindex(index=range(7), columns=range(24), fill_value=0) * 100
    charts.calendar_heat(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], list(range(24)), grid.to_numpy(), height=320)
