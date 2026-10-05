import numpy as np


def test_no_account_in_train_and_test(posts):
    train = set(posts.loc[posts.split == "train", "account"])
    test = set(posts.loc[posts.split == "test", "account"])
    assert train and test
    assert not train & test


def test_each_account_has_one_split_and_fold(posts):
    assert (posts.groupby("account")["split"].nunique() == 1).all()
    assert (posts.groupby("account")["fold"].nunique() == 1).all()


def test_cv_folds_are_account_disjoint(posts, cfg):
    tr = posts[posts.split == "train"]
    assert set(tr["fold"]) == set(range(cfg["split"]["n_folds"]))
    assert (posts.loc[posts.split == "test", "fold"] == -1).all()
    for f in range(cfg["split"]["n_folds"]):
        val_acc = set(tr.loc[tr.fold == f, "account"])
        fit_acc = set(tr.loc[tr.fold != f, "account"])
        assert not val_acc & fit_acc


def test_only_single_image_posts(posts):
    assert (posts["typename"] == "GraphImage").all()
    assert (posts["n_images"] == 1).all()


def test_label_proportions_within_tier_on_train(posts, cfg):
    tr = posts[posts.split == "train"]
    for _, g in tr.groupby("follower_tier"):
        share = g["label"].value_counts(normalize=True)
        assert abs(share.get("Low", 0) - 0.6) < 0.08
        assert abs(share.get("Viral", 0) - 0.1) < 0.08
    assert posts["label"].notna().all()


def test_post_ids_unique(posts):
    assert not posts["post_id"].duplicated().any()


def test_mapping_duplicates_dropped(tmp_path):
    from viralsense.data.loader import read_mapping

    (tmp_path / "m.txt").write_text("h1 h2 h3\na\t1.info\t['1.jpg']\na\t1.info\t['1.jpg']\nb\t2.info\t['2.jpg']\n")
    m = read_mapping(tmp_path, "m.txt")
    assert len(m) == 2
