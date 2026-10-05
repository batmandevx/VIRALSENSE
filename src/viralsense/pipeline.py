"""Run the whole pipeline in order.

    python -m viralsense.pipeline                      # everything
    python -m viralsense.pipeline --stages classify regress
    python -m viralsense.pipeline --jury-limit 20      # jury pilot only scores the first 20 subset posts
"""
import argparse

from viralsense.utils import load_config

STAGES = ["prepare", "features", "jury", "classify", "calibrate", "regress", "cluster", "ablation", "temporal", "control", "leakage",
          "bandit", "caption_bandit", "captions", "compare"]


def run(stages: list[str], cfg: dict | None = None, jury_limit: int | None = None, caption_posts: int = 30) -> None:
    cfg = cfg or load_config()
    for s in stages:
        print(f"\n=== {s} ===")
        if s == "prepare":
            from viralsense.data.prepare import main
            main(cfg)
        elif s == "features":
            from viralsense.features.build import build
            build(cfg)
        elif s == "jury":
            from viralsense.jury.run import main
            main(cfg, limit=jury_limit)
        elif s == "classify":
            from viralsense.models.classify import main
            main(cfg)
        elif s == "regress":
            from viralsense.models.regress import main
            main(cfg)
        elif s == "cluster":
            from viralsense.models.cluster import main
            main(cfg)
        elif s == "ablation":
            from viralsense.models.ablation import main
            main(cfg)
        elif s == "bandit":
            from viralsense.bandit.posting_time import main
            main(cfg)
        elif s == "calibrate":
            from viralsense.models.calibrate import main
            main(cfg)
        elif s == "temporal":
            from viralsense.models.temporal import main
            main(cfg)
        elif s == "control":
            from viralsense.models.control import main
            main(cfg)
        elif s == "leakage":
            from viralsense.models.leakage_demo import main
            main(cfg)
        elif s == "caption_bandit":
            from viralsense.bandit.caption_style import main
            main(cfg)
        elif s == "compare":
            from viralsense.models.compare import main
            main(cfg)
        elif s == "captions":
            from viralsense.bandit.captions import main
            main(cfg, n_posts=caption_posts)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="+", default=STAGES, choices=STAGES)
    ap.add_argument("--jury-limit", type=int, default=None)
    ap.add_argument("--caption-posts", type=int, default=30)
    a = ap.parse_args()
    run(a.stages, jury_limit=a.jury_limit, caption_posts=a.caption_posts)
