import numpy as np
import pandas as pd

from viralsense.bandit.caption_style import style_arm
from viralsense.bandit.replay import bootstrap_lift
from viralsense.suggest import caption_edits, suggest


def test_caption_edits_are_meaningful():
    e = caption_edits("Sunset hike. Best view ever! #travel #hike #nature #sun #sky #mountain #wow")
    assert e["Remove all hashtags"] == "Sunset hike. Best view ever!"
    assert e["Keep only the first 5 hashtags"].endswith("#travel #hike #nature #sun #sky")
    assert e["Ask the audience a question"].startswith("Sunset hike. Best view ever! What do you think?")
    assert e["Shorten to the first sentence"].startswith("Sunset hike.")
    assert caption_edits("Nice?") == {}


class HourModel:
    """P(Viral) peaks at 20:00 on Saturdays."""
    def predict_proba(self, X):
        v = 0.1 + 0.3 * np.exp(-((X["meta_hour"].to_numpy() - 20) / 2.0) ** 2) * (X["meta_weekday"].to_numpy() == 5)
        return np.c_[1 - v - 0.2, np.full(len(v), 0.2), v]


def test_suggest_ranks_by_gain_and_excludes_current_time(cfg):
    X = pd.DataFrame({"meta_hour": [9], "meta_hour_sin": [0.0], "meta_hour_cos": [1.0], "meta_weekday": [1]})
    clf = {"model": HourModel(), "columns": list(X.columns), "classes": ["Low", "Moderate", "Viral"]}
    s = suggest(X, clf, cfg, caption="Nice?")
    assert s.iloc[0]["change"] == "Post on Sat at 20:00 UTC"
    assert s["delta"].is_monotonic_decreasing and (s["kind"] == "timing").all()
    assert not s["change"].str.contains("Tue at 09:00").any()


def test_style_arm_buckets():
    a = style_arm([10, 150, 900], [0, 3, 12], [1, 0, 1], [80, 250])
    assert list(a) == ["short · 0 tags · question", "medium · 1-5 tags · no question", "long · 6+ tags · question"]


def test_bootstrap_lift_ci_brackets_true_gain():
    rng = np.random.default_rng(0)
    lo, hi = bootstrap_lift(rng.binomial(1, 0.5, 400).astype(float), baseline=0.4)
    assert lo < 0.1 < hi and lo > 0
    lo, hi = bootstrap_lift(rng.binomial(1, 0.4, 400).astype(float), baseline=0.4)
    assert lo < 0 < hi
