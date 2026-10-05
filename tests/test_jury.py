import json

import numpy as np
import pytest

from viralsense.jury.client import StubClient, build_prompt, validate_scores
from viralsense.jury.run import jury_features, load_personas, run_jury

FACTORS = ["scroll_stop", "emotional_pull", "shareability", "save_intent", "comment_trigger"]


def test_twelve_unique_personas(cfg):
    ps = load_personas(cfg)
    assert len(ps) == 12 and len({p["id"] for p in ps}) == 12
    for p in ps:
        assert {"age", "interests", "scrolling", "saves_or_shares"} <= set(p)


def test_prompt_contains_no_engagement_or_follower_numbers(cfg, posts):
    r = posts.iloc[0]
    for p in load_personas(cfg):
        prompt = build_prompt(p, r.caption, FACTORS).lower()
        for word in ("follower", "likes", "engagement"):
            assert word not in prompt
        for n in (r.followers, r.likes, r.comments):
            assert str(int(n)) not in prompt


@pytest.mark.parametrize("bad", [
    {"scroll_stop": 0}, {"scroll_stop": 11}, {"scroll_stop": 5.5}, {"scroll_stop": "7"}, {"scroll_stop": True}, {}])
def test_validate_rejects_bad_scores(bad):
    obj = {f: 5 for f in FACTORS}
    obj.pop("scroll_stop")
    obj.update(bad)
    with pytest.raises((ValueError, KeyError)):
        validate_scores(json.dumps(obj), FACTORS)


def test_validate_accepts_integral_floats():
    assert validate_scores({f: 7.0 for f in FACTORS}, FACTORS) == {f: 7 for f in FACTORS}


class CountingStub(StubClient):
    calls = 0

    def score(self, *a, **k):
        CountingStub.calls += 1
        return super().score(*a, **k)


def test_cache_prevents_repeat_calls_and_features_are_correct(cfg, posts):
    sub = posts.head(3)
    client = CountingStub(cfg)
    log = run_jury(sub, cfg, client)
    assert CountingStub.calls == 36 and log["ok"].all()
    assert len(run_jury(sub, cfg, client)) == 0 and CountingStub.calls == 36

    feats = jury_features(sub["post_id"], cfg)
    assert len(feats) == 3 and (feats["jury_n_personas"] == 12).all()
    assert len([c for c in feats if c.endswith(("_mean", "_std")) and c != "jury_overall_mean"]) == 10
    pid = sub["post_id"].iloc[0]
    S = np.array([[client.score(sub.iloc[0].image_path, sub.iloc[0].caption, p)["scores"][f] for f in FACTORS]
                  for p in sorted(load_personas(cfg), key=lambda p: p["id"])])
    row = feats.set_index("post_id").loc[pid]
    assert np.isclose(row["jury_overall_mean"], S.mean())
    assert np.isclose(row["jury_disagreement"], S.mean(1).std())
    assert np.isclose(row["jury_shareability_std"], S[:, 2].std())


class PanelFake:
    """Answers a panel request with fixed scores for every persona id found in the schema."""
    factors = FACTORS
    max_side = 64
    calls = 0

    def chat(self, prompt, image_b64=None, fmt=None, **kw):
        PanelFake.calls += 1
        ids = fmt["properties"]["ratings"]["required"]
        return {"text": json.dumps({"ratings": {i: {f: 1 + (k % 10) for f in FACTORS} for k, i in enumerate(ids)}}),
                "input_tokens": 10, "output_tokens": 10, "seconds": 0.0, "usd": 0.0}


def test_panel_mode_one_call_per_post_fills_all_personas(cfg, posts, tmp_path):
    import copy

    c = copy.deepcopy(cfg)
    c["jury"]["mode"] = "panel"
    c["paths"]["cache"] = str(tmp_path)
    sub = posts.head(4)
    run_jury(sub, c, PanelFake())
    assert PanelFake.calls == 4  # 4 posts -> 4 requests, not 48
    feats = jury_features(sub["post_id"], c)
    assert len(feats) == 4 and (feats["jury_n_personas"] == 12).all()
    run_jury(sub, c, PanelFake())
    assert PanelFake.calls == 4  # cached: no new requests
