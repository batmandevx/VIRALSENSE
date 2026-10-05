"""What leakage looks like (idea from the team's notebook): a model that is allowed to see likes and
comments. NOT a pre-publication result; it exists only to show why the leakage guard matters.

    python -m viralsense.models.leakage_demo
"""
from pathlib import Path

import numpy as np
import pandas as pd

from viralsense.models.common import CLASSES, fit_model, metrics
from viralsense.utils import ensure_dir, load_config, seed_everything


def main(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    seed_everything(cfg["seed"])
    tables = ensure_dir(Path(cfg["paths"]["reports"]) / "tables")
    p = pd.read_parquet(Path(cfg["paths"]["processed"]) / "posts.parquet")
    # Deliberately leaky columns, renamed so the generic preprocessor treats them as numeric metadata.
    X = pd.DataFrame({"meta_log_l": np.log1p(p["likes"]), "meta_log_c": np.log1p(p["comments"]),
                      "meta_log_f": np.log1p(p["followers"]), "follower_tier": p["follower_tier"]})
    y = p["label"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    tr, te = (p.split == "train").to_numpy(), (p.split == "test").to_numpy()
    m = fit_model("logreg", "class_weight", X[tr], y[tr], p.account[tr].to_numpy(), cfg)
    out = pd.DataFrame([{"model": "logreg on log likes, log comments, log followers (LEAKY)",
                         **metrics(y[te], m.predict_proba(X[te]))}])
    out.to_csv(tables / "leakage_demo.csv", index=False)
    print(out.round(3).to_string(index=False))
    return out


if __name__ == "__main__":
    main()
