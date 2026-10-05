"""Phase 4b: generate caption variants with the local LLM, rescore with the trained classifier.

    python -m viralsense.bandit.captions [--n-posts 30]

The offline evaluation reports how much the *classifier's* Viral probability rises when the
best variant replaces the original. That is the model's opinion, not measured engagement.
"""
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from viralsense.data.prepare import HISTORY_COLS
from viralsense.inference import featurize, load_label_spec
from viralsense.jury.client import OllamaClient, encode_image
from viralsense.utils import ensure_dir, load_config, seed_everything

PROMPT = (
    "You write Instagram captions. Rewrite the caption below for the attached image in {n} different ways. "
    "Keep the meaning and any brand or product names, keep a similar length, and keep it natural; "
    "vary tone, hook and hashtags.\n\nOriginal caption:\n\"\"\"{caption}\"\"\"\n\n"
    "Return JSON: {{\"variants\": [{n} strings]}}."
)


def generate_variants(client: OllamaClient, image_path: str, caption: str, n: int) -> list[str]:
    schema = {"type": "object", "required": ["variants"],
              "properties": {"variants": {"type": "array", "items": {"type": "string"}, "minItems": n, "maxItems": n}}}
    out = client.chat(PROMPT.format(n=n, caption=caption.strip() or "(no caption)"),
                      encode_image(image_path, client.max_side), fmt=schema, temperature=0.8)
    variants = [v.strip() for v in json.loads(out["text"])["variants"] if isinstance(v, str) and v.strip()]
    if len(variants) < n:
        raise ValueError(f"expected {n} variants, got {len(variants)}")
    return variants[:n]


def rescore(bundle: dict, posts: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    from viralsense.models.calibrate import calibrated_proba

    X = featurize(posts, cfg)
    proba = calibrated_proba(bundle, X)
    out = posts[["caption"]].reset_index(drop=True).copy()
    for i, c in enumerate(bundle["classes"]):
        out[f"p_{c}"] = proba[:, i]
    return out


def best_caption(bundle: dict, client: OllamaClient, image_path: str, caption: str, followers: float, when,
                 cfg: dict, spec: dict, past_median_er=None, n_past_posts: int = 0, history: dict | None = None) -> pd.DataFrame:
    """Original + n variants, ranked by Viral probability. The recommendation is the top variant."""
    from viralsense.inference import post_frame

    variants = generate_variants(client, image_path, caption, cfg["captions"]["n_variants"])
    posts = post_frame(image_path, [caption] + variants, followers, when, spec, past_median_er, n_past_posts, history)
    scored = rescore(bundle, posts, cfg)
    scored.insert(0, "kind", ["original"] + [f"variant {i + 1}" for i in range(len(variants))])
    return scored.sort_values("p_Viral", ascending=False).reset_index(drop=True)


def main(cfg: dict | None = None, n_posts: int = 30) -> pd.DataFrame:
    cfg = cfg or load_config()
    seed_everything(cfg["seed"])
    proc = Path(cfg["paths"]["processed"])
    tables = ensure_dir(Path(cfg["paths"]["reports"]) / "tables")
    from viralsense.inference import load_model

    bundle = load_model(Path(cfg["paths"]["artifacts"]) / "classifier.joblib")
    spec = load_label_spec(cfg)
    client = OllamaClient({**cfg, "jury": {**cfg["jury"], "model": cfg["captions"]["model"]}})
    posts = pd.read_parquet(proc / "posts.parquet")
    sample = posts[posts.split == "test"].sample(min(n_posts, (posts.split == "test").sum()), random_state=cfg["seed"])

    rows = []
    for r in sample.itertuples():
        try:
            ranked = best_caption(bundle, client, r.image_path, r.caption, r.followers, r.timestamp.to_pydatetime(),
                                  cfg, spec, r.acct_past_median_er, int(r.acct_n_past_posts),
                                  {c: getattr(r, c) for c in HISTORY_COLS})
        except Exception as e:
            rows.append({"post_id": r.post_id, "error": f"{type(e).__name__}: {e}"[:200]})
            continue
        orig = ranked.loc[ranked.kind == "original", "p_Viral"].iloc[0]
        best = ranked[ranked.kind != "original"].iloc[0]
        rows.append({"post_id": r.post_id, "true_label": r.label, "p_viral_original": orig,
                     "p_viral_best_variant": best["p_Viral"], "uplift": best["p_Viral"] - orig,
                     "original_caption": r.caption, "best_variant": best["caption"], "error": ""})
    res = pd.DataFrame(rows)
    res.to_csv(tables / "caption_variants_eval.csv", index=False)
    ok = res[res["error"] == ""]
    print(f"{len(ok)}/{len(res)} posts rewritten; best variant beats the original in {(ok.uplift > 0).mean():.0%}; "
          f"mean model-estimated Viral-probability change {ok['uplift'].mean():+.4f}")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-posts", type=int, default=30)
    main(n_posts=ap.parse_args().n_posts)
