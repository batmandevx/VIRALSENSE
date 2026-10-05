"""Caption features: MiniLM embeddings plus surface statistics and VADER sentiment."""
import re
from functools import lru_cache

import emoji
import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

HASHTAG = re.compile(r"#\w+")


def caption_stats(captions: pd.Series) -> pd.DataFrame:
    vader = SentimentIntensityAnalyzer()
    c = captions.fillna("")
    return pd.DataFrame({
        "cap_len": c.str.len(),
        "cap_emoji_count": c.map(emoji.emoji_count),
        "cap_hashtag_count": c.map(lambda s: len(HASHTAG.findall(s))),
        "cap_has_question": c.str.contains("?", regex=False).astype(int),
        "cap_sentiment": c.map(lambda s: vader.polarity_scores(s)["compound"]),
    })


@lru_cache(maxsize=2)
def load_text_model(model_name: str):
    from sentence_transformers import SentenceTransformer

    from viralsense.features.visual import _device

    return SentenceTransformer(model_name, device=_device())


def text_embeddings(captions: list[str], model_name: str, batch_size: int) -> np.ndarray:
    return load_text_model(model_name).encode(captions, batch_size=batch_size, normalize_embeddings=True,
                                              show_progress_bar=len(captions) > batch_size)


def text_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    f = cfg["features"]
    emb = text_embeddings(df["caption"].fillna("").tolist(), f["text_model"], f["batch_size"])
    emb_df = pd.DataFrame(emb, columns=[f"txt_{i:03d}" for i in range(emb.shape[1])])
    return pd.concat([df[["post_id"]].reset_index(drop=True), caption_stats(df["caption"]).reset_index(drop=True), emb_df], axis=1)
