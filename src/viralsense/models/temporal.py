"""Robustness: does the model hold up on newer posts? (idea from the team's Colab notebook)

    python -m viralsense.models.temporal

Two designs, both training on the oldest 80% of posts by timestamp and testing on the newest 20%:
  time only          any account may appear on both sides (the notebook's split)
  time + accounts    train = older posts of training accounts, test = newer posts of held-out accounts
"""
import json
from pathlib import Path

import pandas as pd

from viralsense.models.classify import history_split
from viralsense.models.common import feature_columns, fit_model, load_table, metrics
from viralsense.utils import ensure_dir, load_config, seed_everything


def main(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    seed_everything(cfg["seed"])
    tables = ensure_dir(Path(cfg["paths"]["reports"]) / "tables")
    df = load_table(cfg)
    ts = pd.read_parquet(Path(cfg["paths"]["processed"]) / "posts.parquet", columns=["post_id", "timestamp"])
    df = df.merge(ts, on="post_id")
    cut = df["timestamp"].quantile(0.8)
    sel = json.loads((tables / "classification_selected.json").read_text())["best_cv"]
    cols = feature_columns(df, cfg)
    hist, _ = history_split(cols)
    old, new = df["timestamp"] < cut, df["timestamp"] >= cut

    designs = {
        "time only": (df[old], df[new]),
        "time + accounts": (df[old & (df.split == "train")], df[new & (df.split == "test")]),
    }
    rows = []
    for design, (tr, te) in designs.items():
        for label, name, imb, c in [(f"{sel['model']} ({sel['imbalance']})", sel["model"], sel["imbalance"], cols),
                                    ("xgb [history only]", "xgb", "class_weight", hist),
                                    ("dummy", "dummy", "class_weight", cols)]:
            m = fit_model(name, imb, tr[c], tr["y"].to_numpy(), tr["account"].to_numpy(), cfg)
            rows.append({"design": design, "model": label, "n_train": len(tr), "n_test": len(te),
                         "test_from": te["timestamp"].min().date().isoformat(),
                         **metrics(te["y"].to_numpy(), m.predict_proba(te[c]))})
            print(f"  {design:16s} {label:28s} macro-F1 {rows[-1]['macro_f1']:.3f}")
    out = pd.DataFrame(rows)
    out.to_csv(tables / "temporal_robustness.csv", index=False)
    return out


if __name__ == "__main__":
    main()
