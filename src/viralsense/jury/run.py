"""Phase 2: persona jury on a stratified subset, cached per (post, persona).

    python -m viralsense.jury.run [--limit N]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from viralsense.jury.client import BudgetExceeded, jury_tag, make_client
from viralsense.utils import ROOT, ensure_dir, load_config, seed_everything


def load_personas(cfg: dict) -> list[dict]:
    path = Path(cfg["jury"]["personas_file"])
    with open(path if path.is_absolute() else ROOT / path) as f:
        personas = yaml.safe_load(f)["personas"]
    assert len({p["id"] for p in personas}) == len(personas) == 12, "need 12 uniquely named personas"
    return personas


def select_subset(posts: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Stratified by split x follower tier x label, proportional allocation."""
    if n >= len(posts):
        return posts.copy()
    strata = posts["split"].astype(str) + "|" + posts["follower_tier"].astype(str) + "|" + posts["label"].astype(str)
    frac = n / len(posts)
    parts = [g.sample(max(1, int(round(frac * len(g)))), random_state=seed) for _, g in posts.groupby(strata)]
    sub = pd.concat(parts)
    if len(sub) > n:
        sub = sub.sample(n, random_state=seed)
    return sub.sort_values("post_id").reset_index(drop=True)


def cache_path(cache_dir: Path, post_id: str, persona_id: str) -> Path:
    return cache_dir / str(post_id) / f"{persona_id}.json"


def run_jury(posts: pd.DataFrame, cfg: dict, client=None, usage_path: Path | None = None) -> pd.DataFrame:
    """Score every (post, persona) pair not already cached. Returns the usage log of new calls.

    With usage_path, each call's usage row is appended as soon as it finishes, so an
    interrupted run keeps its token log."""
    import threading

    lock = threading.Lock()
    fields = ["post_id", "persona_id", "ok", "attempts", "input_tokens", "output_tokens", "seconds", "usd", "error"]
    client = client or make_client(cfg)
    personas = load_personas(cfg)
    cache_dir = ensure_dir(Path(cfg["paths"]["cache"]) / "jury" / jury_tag(cfg))
    todo = [(r, p) for r in posts.itertuples() for p in personas
            if not cache_path(cache_dir, r.post_id, p["id"]).exists()]

    def one(r, p) -> dict:
        rec = {"post_id": r.post_id, "persona_id": p["id"], "ok": False, "attempts": 0,
               "input_tokens": 0, "output_tokens": 0, "seconds": 0.0, "usd": 0.0, "error": ""}
        for attempt in range(cfg["jury"]["max_retries"] + 1):
            rec["attempts"] = attempt + 1
            try:
                out = client.score(r.image_path, r.caption, p)
            except BudgetExceeded:
                raise
            except Exception as e:  # network, JSON or validation failure: retry
                rec["error"] = f"{type(e).__name__}: {e}"[:300]
                continue
            for k in ("input_tokens", "output_tokens", "seconds"):
                rec[k] += out[k]
            rec["usd"] = rec.get("usd", 0.0) + out.get("usd", 0.0)
            path = cache_path(cache_dir, r.post_id, p["id"])
            ensure_dir(path.parent)
            path.write_text(json.dumps({"post_id": r.post_id, "persona_id": p["id"], "scores": out["scores"],
                                        "raw": out["text"], "model": cfg["jury"]["model"]}))
            rec["ok"], rec["error"] = True, ""
            break
        if usage_path is not None:
            with lock:
                new_file = not usage_path.exists()
                pd.DataFrame([rec], columns=fields).to_csv(usage_path, mode="a", header=new_file, index=False)
        return rec

    workers = cfg["openrouter"].get("workers", 4) if cfg["jury"]["backend"] == "openrouter" else cfg["jury"].get("workers", 1)
    if cfg["jury"].get("mode", "per_persona") == "panel":
        return _run_panel(posts, cfg, client, personas, cache_dir, usage_path, workers)
    if workers <= 1:
        log = [one(r, p) for r, p in tqdm(todo, desc="jury")]
    else:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(workers) as ex:
            log = list(tqdm(ex.map(lambda rp: one(*rp), todo), total=len(todo), desc="jury"))
    return pd.DataFrame(log)


def _run_panel(posts, cfg, client, personas, cache_dir, usage_path, workers) -> pd.DataFrame:
    """Panel mode: one request per post rates it for all personas; cached in the same per-persona layout."""
    import threading

    from viralsense.jury.client import score_panel

    lock = threading.Lock()
    todo = [r for r in posts.itertuples() if not all(cache_path(cache_dir, r.post_id, p["id"]).exists() for p in personas)]

    def one(r) -> dict:
        rec = {"post_id": r.post_id, "persona_id": "panel", "ok": False, "attempts": 0, "input_tokens": 0,
               "output_tokens": 0, "seconds": 0.0, "usd": 0.0, "error": ""}
        for attempt in range(cfg["jury"]["max_retries"] + 1):
            rec["attempts"] = attempt + 1
            try:
                out = score_panel(client, r.image_path, r.caption, personas)
            except BudgetExceeded:
                raise
            except Exception as e:
                rec["error"] = f"{type(e).__name__}: {e}"[:300]
                if "429" in rec["error"] or "rate" in rec["error"].lower():
                    time.sleep(30)
                continue
            for k in ("input_tokens", "output_tokens", "seconds", "usd"):
                rec[k] += out.get(k, 0)
            for pid, scores in out["panel"].items():
                path = cache_path(cache_dir, r.post_id, pid)
                ensure_dir(path.parent)
                path.write_text(json.dumps({"post_id": r.post_id, "persona_id": pid, "scores": scores, "mode": "panel",
                                            "model": getattr(client, "model", "")}))
            rec["ok"], rec["error"] = True, ""
            break
        if usage_path is not None:
            with lock:
                pd.DataFrame([rec]).to_csv(usage_path, mode="a", header=not usage_path.exists(), index=False)
        return rec

    if workers <= 1:
        log = [one(r) for r in tqdm(todo, desc="jury panel")]
    else:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(workers) as ex:
            log = list(tqdm(ex.map(one, todo), total=len(todo), desc="jury panel"))
    return pd.DataFrame(log)


def jury_features(post_ids, cfg: dict) -> pd.DataFrame:
    """Per-factor mean and std across personas, overall mean, and disagreement.

    disagreement = std across personas of each persona's mean score over the five factors.
    """
    factors = cfg["jury"]["factors"]
    cache_dir = Path(cfg["paths"]["cache"]) / "jury" / jury_tag(cfg)
    rows = []
    for pid in post_ids:
        files = sorted((cache_dir / str(pid)).glob("*.json"))
        if not files:
            continue
        S = np.array([[json.loads(f.read_text())["scores"][k] for k in factors] for f in files], dtype=float)
        row = {"post_id": pid, "jury_n_personas": len(files)}
        for j, k in enumerate(factors):
            row[f"jury_{k}_mean"] = S[:, j].mean()
            row[f"jury_{k}_std"] = S[:, j].std(ddof=0)
        row["jury_overall_mean"] = S.mean()
        row["jury_disagreement"] = S.mean(axis=1).std(ddof=0)
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_usage(log: pd.DataFrame, cfg: dict) -> dict:
    j = cfg["jury"]
    tin, tout = int(log["input_tokens"].sum()), int(log["output_tokens"].sum())
    return {
        "backend": j["backend"], "model": cfg["openrouter"]["model"] if j["backend"] == "openrouter" else j["model"], "calls": len(log), "failed_calls": int((~log["ok"]).sum()),
        "retries": int((log["attempts"] - 1).sum()), "input_tokens": tin, "output_tokens": tout,
        "usd": round(tin / 1000 * j["usd_per_1k_input_tokens"] + tout / 1000 * j["usd_per_1k_output_tokens"], 4),
        "wall_hours": round(log["seconds"].sum() / 3600, 3),
        "sec_per_call": round(log["seconds"].mean(), 2) if len(log) else 0.0,
    }


def main(cfg: dict | None = None, limit: int | None = None, features_only: bool = False) -> pd.DataFrame:
    cfg = cfg or load_config()
    seed_everything(cfg["seed"])
    proc, rep = Path(cfg["paths"]["processed"]), ensure_dir(Path(cfg["paths"]["reports"]) / "tables")
    posts = pd.read_parquet(proc / "posts.parquet")
    subset = select_subset(posts, cfg["jury"]["n_posts"], cfg["seed"])
    subset[["post_id"]].to_csv(proc / "jury_subset.csv", index=False)
    todo = subset.head(limit) if limit else subset

    tag = jury_tag(cfg)
    suffix = "" if tag == "ollama" else f"_{tag}"
    usage_path = rep / f"jury_usage_log{suffix}.csv"
    if not features_only:  # features_only: build features from the cache, no model calls
        run_jury(todo, cfg, usage_path=usage_path)
    if usage_path.exists():
        summary = summarize_usage(pd.read_csv(usage_path), cfg)  # all logged runs so far (pilot + full)
        cache_dir = Path(cfg["paths"]["cache"]) / "jury" / jury_tag(cfg)
        summary["cached_responses"] = sum(1 for _ in cache_dir.glob("*/*.json"))
        summary["responses_without_usage_log"] = summary["cached_responses"] - int(pd.read_csv(usage_path)["ok"].sum())
        if "usd" in pd.read_csv(usage_path):
            summary["usd"] = round(float(pd.read_csv(usage_path)["usd"].fillna(0).sum()), 4)
        pd.DataFrame([summary]).to_csv(rep / f"jury_usage_summary{suffix}.csv", index=False)
        print(json.dumps(summary, indent=2))
        if summary["failed_calls"]:
            print(f"WARNING: {summary['failed_calls']} calls failed after retries; see jury_usage_log.csv")

    feats = jury_features(subset["post_id"], cfg)
    complete = feats[feats["jury_n_personas"] == 12]
    complete.drop(columns="jury_n_personas").to_parquet(proc / f"jury_features{suffix}.parquet", index=False)
    print(f"jury features for {len(complete)}/{len(subset)} posts with all 12 personas")
    return complete


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="score only the first N subset posts (pilot)")
    ap.add_argument("--backend", choices=["ollama", "openrouter", "stub"], default=None)
    ap.add_argument("--provider", choices=["openrouter", "nvidia", "gemini"], default=None)
    ap.add_argument("--model", default=None, help="API model id, e.g. qwen/qwen3.8-27b:free")
    ap.add_argument("--features-only", action="store_true", help="build jury features from cached ratings, no new calls")
    ap.add_argument("--mode", choices=["per_persona", "panel"], default=None,
                    help="panel = one call per post for all 12 personas (12x fewer calls)")
    a = ap.parse_args()
    cfg = load_config()
    if a.backend:
        cfg["jury"]["backend"] = a.backend
    if a.provider:
        cfg["openrouter"]["provider"] = a.provider
    if a.model:
        cfg["openrouter"]["model"] = a.model
    if a.mode:
        cfg["jury"]["mode"] = a.mode
    main(cfg, limit=a.limit, features_only=a.features_only)
