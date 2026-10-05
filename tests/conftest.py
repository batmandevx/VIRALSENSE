"""Synthetic dataset in the on-disk format of the Instagram Influencer Dataset.

Used only to exercise the code paths; never used for any reported result.
"""
import copy
import json

import numpy as np
import pytest
from PIL import Image

from viralsense.utils import load_config

N_ACCOUNTS, N_POSTS = 48, 30


def write_fake_dataset(root, seed=0, n_accounts=N_ACCOUNTS, n_posts=N_POSTS, signal=False):
    """signal=True: likes rise with the caption's hashtag count (a known, learnable effect)."""
    rng = np.random.default_rng(seed)
    (root / "info").mkdir(parents=True)
    (root / "image").mkdir()
    inf = ["Username\tCategory\t#Followers\t#Followees\t#Posts"]
    mapping = ["Influencer_name\tJSON_PostMetadata_file_name\tImage_file_name"]
    for a in range(n_accounts):
        name, followers = f"user{a:03d}", int(10 ** rng.uniform(3, 7))
        inf.append(f"{name}\tfashion\t{followers}\t100\t{n_posts}")
        for k in range(n_posts):
            kind = rng.choice(["GraphImage", "GraphImage", "GraphImage", "GraphSidecar", "GraphVideo"])
            n_img = 2 if kind == "GraphSidecar" else 1
            imgs = [f"{name}-{k}-{i}.jpg" for i in range(n_img)]
            for im in imgs:
                arr = rng.integers(0, 255, (24, 24, 3), dtype=np.uint8)
                Image.fromarray(arr).save(root / "image" / im)
            n_tags = int(rng.integers(0, 6))
            likes = int(followers * rng.lognormal(-3.5 + (0.25 * n_tags if signal else 0), 0.8))
            node = {
                "__typename": kind, "id": f"{a}{k:04d}", "is_video": kind == "GraphVideo",
                "taken_at_timestamp": 1_500_000_000 + k * 86_400 + int(rng.integers(0, 86_400)),
                "edge_media_to_caption": {"edges": [{"node": {"text": f"post {k} " + " ".join(f"#tag{t}" for t in range(n_tags)) + " 😀 ok?"}}]},
                "edge_media_preview_like": {"count": likes},
                "edge_media_to_comment": {"count": int(likes * 0.02)},
            }
            if kind == "GraphSidecar":
                node["edge_sidecar_to_children"] = {"edges": [{}, {}]}
            (root / "info" / f"{name}-{k}.info").write_text(json.dumps(node))
            mapping.append(f"{name}\t{name}-{k}.info\t{imgs!r}")
    (root / "influencers.txt").write_text("\n".join(inf))
    (root / "JSON-Image_files_mapping.txt").write_text("\n".join(mapping))


@pytest.fixture(scope="session")
def cfg(tmp_path_factory):
    root = tmp_path_factory.mktemp("dataset")
    write_fake_dataset(root)
    c = copy.deepcopy(load_config())
    work = tmp_path_factory.mktemp("work")
    c["paths"].update(data_root=str(root), json_archive=None, image_archive_index=None, interim=str(work / "interim"),
                      processed=str(work / "processed"), cache=str(work / "cache"), reports=str(work / "reports"),
                      artifacts=str(work / "artifacts"))
    c["jury"]["backend"] = "stub"
    c["sample"].update(n_posts=800, n_accounts=N_ACCOUNTS, max_posts_per_account=20, min_posts_per_account=5)
    return c


@pytest.fixture(scope="session")
def posts(cfg):
    from viralsense.data.prepare import main

    return main(cfg)
