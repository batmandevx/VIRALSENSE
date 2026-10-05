"""Pre-posting metadata. Timestamps in the dataset are UTC; local posting time is unknown."""
import numpy as np
import pandas as pd

from viralsense.data.prepare import HISTORY_COLS


def metadata_features(df: pd.DataFrame) -> pd.DataFrame:
    ts = df["timestamp"].dt
    hour = ts.hour
    return pd.DataFrame({
        "post_id": df["post_id"].values,
        "meta_hour": hour.values,
        "meta_hour_sin": np.sin(2 * np.pi * hour / 24).values,
        "meta_hour_cos": np.cos(2 * np.pi * hour / 24).values,
        "meta_weekday": ts.weekday.values,
        "follower_tier": df["follower_tier"].values,
        "meta_log_followers": np.log10(df["followers"].values),
        **{c: df[c].values for c in HISTORY_COLS if c in df},
    })
