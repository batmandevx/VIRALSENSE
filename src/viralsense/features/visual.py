"""Image features: CLIP ViT-B/32 embeddings and simple photographic statistics."""
from functools import lru_cache

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

_FACE = None


def _device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=2)
def load_clip(model_name: str, pretrained: str):
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=_device())
    return model.eval(), preprocess


def clip_embeddings(paths: list[str], model_name: str, pretrained: str, batch_size: int) -> np.ndarray:
    dev = _device()
    model, preprocess = load_clip(model_name, pretrained)
    out = []
    with torch.no_grad():
        for i in tqdm(range(0, len(paths), batch_size), desc="clip"):
            batch = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in paths[i : i + batch_size]]).to(dev)
            emb = model.encode_image(batch)
            out.append(torch.nn.functional.normalize(emb, dim=-1).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, model.visual.output_dim), dtype=np.float32)


def colourfulness(rgb: np.ndarray) -> float:
    """Hasler & Suesstrunk (2003) colourfulness metric."""
    r, g, b = (rgb[..., i].astype(float) for i in range(3))
    rg, yb = r - g, 0.5 * (r + g) - b
    return float(np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean()))


def image_stats(path: str, max_side: int = 640) -> dict:
    global _FACE
    if _FACE is None:
        _FACE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    img = Image.open(path).convert("RGB")
    img.thumbnail((max_side, max_side))
    rgb = np.asarray(img)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    faces = _FACE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24))
    return {
        "img_brightness": float(gray.mean() / 255),
        "img_contrast": float(gray.std() / 255),
        "img_colourfulness": colourfulness(rgb),
        "img_face_count": int(len(faces)),
    }


def concept_slug(c: str) -> str:
    return "img_concept_" + "".join(ch if ch.isalnum() else "_" for ch in c.split(" or ")[0]).strip("_")


@lru_cache(maxsize=4)
def concept_text_embeddings(model_name: str, pretrained: str, concepts: tuple[str, ...]) -> np.ndarray:
    import open_clip

    model, _ = load_clip(model_name, pretrained)
    tok = open_clip.get_tokenizer(model_name)
    with torch.no_grad():
        t = model.encode_text(tok([f"a photo of {c}" for c in concepts]).to(_device()))
    return torch.nn.functional.normalize(t, dim=-1).float().cpu().numpy()


def concept_scores(img_emb: np.ndarray, cfg: dict) -> pd.DataFrame:
    """Zero-shot CLIP: softmax over 'a photo of <concept>' prompts (temperature 100, as in CLIP)."""
    f = cfg["features"]
    concepts = tuple(f["concepts"])
    T = concept_text_embeddings(f["clip_model"], f["clip_pretrained"], concepts)
    logits = 100.0 * img_emb @ T.T
    p = np.exp(logits - logits.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    return pd.DataFrame(p, columns=[concept_slug(c) for c in concepts])


def visual_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    f = cfg["features"]
    emb = clip_embeddings(df["image_path"].tolist(), f["clip_model"], f["clip_pretrained"], f["batch_size"])
    stats = pd.DataFrame([image_stats(p) for p in tqdm(df["image_path"], desc="image stats", disable=len(df) < 50)])
    emb_df = pd.DataFrame(emb, columns=[f"clip_{i:03d}" for i in range(emb.shape[1])])
    parts = [df[["post_id"]].reset_index(drop=True), stats]
    if f.get("concepts"):
        parts.append(concept_scores(emb, cfg))
    return pd.concat(parts + [emb_df], axis=1)
