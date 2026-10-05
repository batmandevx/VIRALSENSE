"""ViralSense inference helper. Usage:
    from viralsense_infer import ViralSense
    vs = ViralSense("best_path.pth")
    vs.predict({"media_type": "reel", "content_category": "Fitness", "post_hour": 7})
"""
import math
import numpy as np
import pandas as pd


def _load(path):
    try:
        import torch
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        import joblib
        return joblib.load(path)


class ViralSense:
    def __init__(self, path="best_path.pth"):
        b = _load(path); self.b = b
        self.model, self.reg = b["final_model"], b["regressor"]
        self.cols, self.classes = b["feature_cols"], b["class_names"]
        self.defaults, self.ref = b["defaults"], b["ref_defaults"]
        self.dims, self.cats, self.media = b["dims"], b["cats"], b["media"]
        self.days, self.sources, self.acct = b["days"], b["sources"], b["acct"]
        self.personas, self.seed = b["personas"], b["seed"]
        self.er_sorted = b["er_train_sorted"]
        self.t_arms, self.c_arms = b["t_arms"], b["c_arms"]

    # ---------------- persona jury ----------------
    @staticmethod
    def _circ_fit(hours, active):
        d = np.min([np.minimum(np.abs(hours - a), 24 - np.abs(hours - a)) for a in active], axis=0)
        return np.exp(-(d / 3.0) ** 2)

    def run_jury(self, df, noise=0.4, seed=None):
        rng = np.random.default_rng(self.seed if seed is None else seed)
        h = df.post_hour.values.astype(float); cl = df.caption_length.values.astype(float)
        ht = df.hashtags_count.values.astype(float); cta = df.has_call_to_action.values.astype(float)
        mi = df.media_type.map({m: i for i, m in enumerate(self.media)}).values
        ex = df.traffic_source.isin(["Explore", "Reels Feed", "Hashtags"]).values.astype(float)
        hashtag_fit = np.exp(-((ht - 8) / 6.0) ** 2)
        out = np.zeros((len(df), len(self.personas), len(self.dims)))
        for p, (name, (cats, med, hours, cap, cta_s, com)) in enumerate(self.personas.items()):
            cat = np.where(df.content_category.isin(cats), 1.0, 0.25)
            media = np.array(med)[mi]; time = self._circ_fit(h, hours)
            capf = np.exp(-((cl - cap) / 60.0) ** 2); ctaf = cta * cta_s
            sig = {
              "scroll_stop":     0.40*media + 0.30*cat + 0.20*time + 0.10*hashtag_fit,
              "emotional_pull":  0.40*cat   + 0.30*capf + 0.20*media + 0.10*time,
              "shareability":    0.30*media + 0.30*cat + 0.20*ctaf*2 + 0.10*hashtag_fit + 0.10*ex,
              "save_intent":     0.35*cat   + 0.35*capf + 0.20*(mi >= 1) + 0.10*hashtag_fit,
              "comment_trigger": 0.40*ctaf*2 + 0.25*capf + 0.15*time + 0.20*cat*com,
            }
            for d, dim in enumerate(self.dims):
                out[:, p, d] = np.clip(1 + 9 * sig[dim] + rng.normal(0, noise, len(df)), 1, 10)
        return out

    def jury_features(self, df, **kw):
        J = self.run_jury(df, **kw); f = {}
        for d, dim in enumerate(self.dims):
            f[f"jury_{dim}_mean"] = J[:, :, d].mean(1); f[f"jury_{dim}_std"] = J[:, :, d].std(1)
            f[f"jury_{dim}_min"] = J[:, :, d].min(1);  f[f"jury_{dim}_max"] = J[:, :, d].max(1)
        f["jury_overall"] = J.mean((1, 2))
        return pd.DataFrame(f, index=df.index), J

    # ---------------- features ----------------
    def build_features(self, df):
        X = pd.DataFrame(index=df.index)
        X["log_followers"] = np.log1p(df.follower_count)
        X["caption_length"] = df.caption_length; X["hashtags_count"] = df.hashtags_count
        X["has_cta"] = df.has_call_to_action; X["post_hour"] = df.post_hour
        X["hour_sin"] = np.sin(2*np.pi*df.post_hour/24); X["hour_cos"] = np.cos(2*np.pi*df.post_hour/24)
        X["is_weekend"] = df.day_of_week.isin(["Saturday", "Sunday"]).astype(int)
        X["is_prime_time"] = df.post_hour.between(18, 22).astype(int)
        X["caption_per_tag"] = df.caption_length / (df.hashtags_count + 1)
        for col, levels in [("account_type", self.acct), ("media_type", self.media), ("content_category", self.cats),
                            ("traffic_source", self.sources), ("day_of_week", self.days)]:
            for l in levels: X[f"{col}={l}"] = (df[col] == l).astype(int)
        jf, _ = self.jury_features(df)
        return pd.concat([X, jf], axis=1)[self.cols]

    def _row(self, d): return pd.DataFrame([{**self.defaults, **d}])
    def er_to_score(self, er): return float(100 * np.searchsorted(self.er_sorted, er) / len(self.er_sorted))

    # ---------------- prediction ----------------
    def predict(self, d):
        r = self._row(d); X = self.build_features(r)
        proba = self.model.predict_proba(X)[0]; er = float(self.reg.predict(X)[0])
        _, J = self.jury_features(r)
        jury = {name: {dim: round(float(J[0, i, k]), 2) for k, dim in enumerate(self.dims)}
                for i, name in enumerate(self.personas)}
        return {"label": self.classes[int(np.argmax(proba))],
                "probabilities": {c: float(p) for c, p in zip(self.classes, proba)},
                "engagement_score": self.er_to_score(er), "engagement_rate_pred": er,
                "jury": jury,
                "jury_mean": {dim: round(float(J[0, :, k].mean()), 2) for k, dim in enumerate(self.dims)}}

    def factor_effects(self, d):
        """Positive = this input raises P(Viral) vs. the training default."""
        base = self.predict(d)["probabilities"]["Viral"]; out = {}
        for k in self.b["controllable"]:
            out[k] = base - self.predict({**d, k: self.ref[k]})["probabilities"]["Viral"]
        return dict(sorted(out.items(), key=lambda kv: kv[1]))

    def suggest_changes(self, d, top=5):
        base = self.predict(d)["probabilities"]["Viral"]; full = {**self.defaults, **d}
        grid = {"post_hour": range(24), "day_of_week": self.days, "has_call_to_action": [0, 1],
                "hashtags_count": [3, 5, 8, 10, 15], "caption_length": [60, 100, 140, 180],
                "media_type": self.media, "traffic_source": self.sources}
        cands = [(k, v, {**full, k: v}) for k, vals in grid.items() for v in vals if v != full[k]]
        pv = self.model.predict_proba(self.build_features(pd.DataFrame([c[2] for c in cands])))[:, self.classes.index("Viral")]
        res = pd.DataFrame({"change": [f"{k} -> {v}" for k, v, _ in cands], "p_viral": pv, "delta": pv - base})
        return res.sort_values("delta", ascending=False).head(top).round(4).to_dict("records")

    # ---------------- bandit recommender ----------------
    @staticmethod
    def _post(st, ctx, arm):
        n, s = st["n"].get((ctx, arm), 0), st["s"].get((ctx, arm), 0.0)
        prec = 1/st["pv"] + n/st["nv"]
        return (st["pm"]/st["pv"] + s/st["nv"]) / prec, 1/prec, n

    def recommend(self, account_type="creator", media_type="reel", k=3):
        ctx = f"{account_type} / {media_type}"
        def rank(st, arms):
            rows = [(a, *self._post(st, ctx, a)) for a in arms]
            rows.sort(key=lambda r: r[1], reverse=True)
            return [{"arm": a, "posterior_mean": round(m, 3), "n_obs": n} for a, m, v, n in rows[:k]]
        return {"best_times": rank(self.b["bandit_time"], self.t_arms),
                "best_captions": rank(self.b["bandit_caption"], self.c_arms)}
