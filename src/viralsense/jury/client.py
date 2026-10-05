"""Vision-LLM clients that score one post for one persona."""
import base64
import hashlib
import io
import json
import time

import numpy as np
import requests
from PIL import Image

SYSTEM = (
    "You are role-playing an Instagram user. Stay in character and react as this person would "
    "when the post appears in their feed. Rate the post on each factor with an integer from 1 to 10. "
    "Reply with JSON only."
)


def persona_text(p: dict) -> str:
    return (
        f"You are: {p['name']}, age {p['age']}. Interests: {', '.join(p['interests'])}.\n"
        f"How you scroll: {p['scrolling']}\nWhat makes you save or share: {p['saves_or_shares']}"
    )


def build_prompt(persona: dict, caption: str, factors: list[str]) -> str:
    """Only the persona, the caption and the factor names: no engagement or follower numbers."""
    keys = ", ".join(f'"{f}"' for f in factors)
    return (
        f"{persona_text(persona)}\n\nThe post's caption is:\n\"\"\"{caption.strip() or '(no caption)'}\"\"\"\n\n"
        "Rate this post from your point of view:\n"
        "- scroll_stop: would it make you stop scrolling?\n"
        "- emotional_pull: how strongly does it make you feel something?\n"
        "- shareability: how likely are you to share it?\n"
        "- save_intent: how likely are you to save it?\n"
        "- comment_trigger: how likely are you to comment?\n"
        f"Return a JSON object with exactly these integer keys: {keys}."
    )


def score_schema(factors: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {f: {"type": "integer", "minimum": 1, "maximum": 10} for f in factors},
        "required": factors,
    }


def validate_scores(raw: str | dict, factors: list[str]) -> dict[str, int]:
    obj = json.loads(raw) if isinstance(raw, str) else raw
    out = {}
    for f in factors:
        v = obj[f]
        if isinstance(v, float) and v.is_integer():
            v = int(v)
        if not isinstance(v, int) or isinstance(v, bool) or not 1 <= v <= 10:
            raise ValueError(f"{f}={v!r} is not an integer in 1..10")
        out[f] = v
    return out


def encode_image(path: str, max_side: int) -> str:
    img = Image.open(path).convert("RGB")
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


class OllamaClient:
    def __init__(self, cfg: dict):
        j = cfg["jury"]
        self.model, self.host, self.factors = j["model"], j["host"].rstrip("/"), j["factors"]
        self.max_side, self.temperature, self.timeout = j["image_max_side"], j["temperature"], j["timeout_s"]
        self.seed = cfg["seed"]

    def chat(self, prompt: str, image_b64: str | None = None, fmt=None, temperature=None, system: str | None = None) -> dict:
        msg = {"role": "user", "content": prompt}
        if image_b64:
            msg["images"] = [image_b64]
        system = system if system is not None else (SYSTEM if fmt else None)
        body = {
            "model": self.model, "stream": False,
            "messages": [{"role": "system", "content": system}, msg] if system else [msg],
            "options": {"temperature": self.temperature if temperature is None else temperature, "seed": self.seed},
        }
        if fmt:
            body["format"] = fmt
        t0 = time.time()
        r = requests.post(f"{self.host}/api/chat", json=body, timeout=self.timeout)
        r.raise_for_status()
        d = r.json()
        return {"text": d["message"]["content"], "input_tokens": d.get("prompt_eval_count", 0),
                "output_tokens": d.get("eval_count", 0), "seconds": time.time() - t0}

    def score(self, image_path: str, caption: str, persona: dict) -> dict:
        out = self.chat(build_prompt(persona, caption, self.factors), encode_image(image_path, self.max_side),
                        fmt=score_schema(self.factors))
        out["scores"] = validate_scores(out["text"], self.factors)
        return out


def read_secret(name: str) -> str | None:
    """Environment variable, else a KEY=value line in the git-ignored .env at the project root."""
    import os

    from viralsense.utils import ROOT

    if os.environ.get(name):
        return os.environ[name]
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            k, _, v = line.partition("=")
            if k.strip() == name and v.strip():
                return v.strip().strip('"').strip("'")
    return None


class BudgetExceeded(RuntimeError):
    pass


PROVIDERS = {  # OpenAI-compatible chat-completions endpoints
    "openrouter": {"url": "https://openrouter.ai/api/v1/chat/completions", "key": "OPENROUTER_API_KEY", "json": "schema"},
    "nvidia": {"url": "https://integrate.api.nvidia.com/v1/chat/completions", "key": "NVIDIA_API_KEY", "json": "prompt"},
    "gemini": {"url": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions", "key": "GEMINI_API_KEY",
               "json": "schema"},
}


class OpenRouterClient:
    """Any OpenAI-compatible provider (OpenRouter, NVIDIA build.nvidia.com, Google Gemini) with JSON output,
    retries, per-call cost (when the provider reports it) and a hard spending cap."""

    def __init__(self, cfg: dict):
        import threading

        j, o = cfg["jury"], cfg["openrouter"]
        prov = PROVIDERS[o.get("provider", "openrouter")]
        self.provider, self.url, self.json_mode = o.get("provider", "openrouter"), prov["url"], o.get("json_mode", prov["json"])
        self.model, self.factors, self.max_side = o["model"], j["factors"], j["image_max_side"]
        self.temperature, self.timeout, self.seed = j["temperature"], j["timeout_s"], cfg["seed"]
        self.max_usd, self.spent, self.lock = o.get("max_usd", 25.0), 0.0, threading.Lock()
        self.min_interval, self._last = 60.0 / o["rpm"] if o.get("rpm") else 0.0, 0.0
        # Project rule: free models only. OpenRouter free models end in ":free"; other providers' free tiers
        # must be allowed explicitly by listing the model in openrouter.free_models.
        if o.get("free_only", True) and not (self.model.endswith(":free") or self.model in o.get("free_models", [])):
            raise RuntimeError(f"{self.model} is not a free model and openrouter.free_only is on")
        self.key = read_secret(prov["key"])
        if not self.key:
            raise RuntimeError(f"{prov['key']} not set (environment or .env in the project root)")
        self.session = requests.Session()

    def _throttle(self):
        if not self.min_interval:
            return
        with self.lock:
            wait = self._last + self.min_interval - time.time()
            self._last = max(time.time(), self._last + self.min_interval)
        if wait > 0:
            time.sleep(wait)

    def chat(self, prompt: str, image_b64: str | None = None, fmt=None, temperature=None, system: str | None = None) -> dict:
        if self.spent > self.max_usd:
            raise BudgetExceeded(f"spent ${self.spent:.2f} >= cap ${self.max_usd:.2f}")
        system = system if system is not None else (SYSTEM if fmt else None)
        content = [{"type": "text", "text": prompt}]
        if image_b64:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}})
        if fmt and self.json_mode == "prompt":  # provider without JSON-schema support: ask in the prompt, validate after
            content[0]["text"] += f"\n\nReply with a single JSON object matching this schema, nothing else: {json.dumps(fmt)}"
        body = {"model": self.model, "messages": ([{"role": "system", "content": system}] if system else [])
                + [{"role": "user", "content": content}],
                "temperature": self.temperature if temperature is None else temperature, "seed": self.seed,
                "max_tokens": 400}
        if self.provider == "openrouter":
            body.update({"usage": {"include": True}, "reasoning": {"enabled": False}})
        if fmt and self.json_mode == "schema":
            body["response_format"] = {"type": "json_schema", "json_schema": {"name": "reply", "strict": True, "schema": fmt}}
        elif fmt and self.json_mode == "object":
            body["response_format"] = {"type": "json_object"}
        t0 = time.time()
        for attempt in range(6):
            self._throttle()
            r = self.session.post(self.url, json=body, timeout=self.timeout,
                                  headers={"Authorization": f"Bearer {self.key}", "X-Title": "ViralSense"})
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(min(30, 2 ** attempt))
                continue
            r.raise_for_status()
            break
        else:
            r.raise_for_status()
        d = r.json()
        if "error" in d:
            raise RuntimeError(str(d["error"])[:300])
        u = d.get("usage", {}) or {}
        cost = float(u.get("cost") or 0.0)
        with self.lock:
            self.spent += cost
        text = d["choices"][0]["message"]["content"] or ""
        text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        if "{" in text and not text.startswith("{"):  # some models wrap JSON in prose
            text = text[text.index("{"): text.rindex("}") + 1]
        return {"text": text, "input_tokens": int(u.get("prompt_tokens", 0)), "output_tokens": int(u.get("completion_tokens", 0)),
                "seconds": time.time() - t0, "usd": cost}

    def score(self, image_path: str, caption: str, persona: dict) -> dict:
        out = self.chat(build_prompt(persona, caption, self.factors), encode_image(image_path, self.max_side),
                        fmt=score_schema(self.factors))
        out["scores"] = validate_scores(out["text"], self.factors)
        return out


class StubClient:
    """Deterministic fake scores for tests and smoke runs. Never use for reported results."""

    def __init__(self, cfg: dict):
        self.factors = cfg["jury"]["factors"]

    def score(self, image_path: str, caption: str, persona: dict) -> dict:
        h = int(hashlib.sha256(f"{image_path}|{caption}|{persona['id']}".encode()).hexdigest(), 16) % 2**32
        vals = np.random.default_rng(h).integers(1, 11, len(self.factors))
        text = json.dumps({f: int(v) for f, v in zip(self.factors, vals)})
        return {"text": text, "scores": validate_scores(text, self.factors), "input_tokens": 0,
                "output_tokens": 0, "seconds": 0.0}


def make_client(cfg: dict):
    return {"ollama": OllamaClient, "openrouter": OpenRouterClient, "stub": StubClient}[cfg["jury"]["backend"]](cfg)


def jury_tag(cfg: dict) -> str:
    """Cache / feature tag: the local per-persona jury keeps its original path; others are separated."""
    b = cfg["jury"]["backend"]
    tag = b
    if b == "openrouter":
        o = cfg["openrouter"]
        tag = f"{o.get('provider', 'openrouter')}__" + o["model"].replace("/", "_").replace(":", "_")
    return tag + ("__panel" if cfg["jury"].get("mode", "per_persona") == "panel" else "")


# ---------------------------------------------------------------- live agents (dashboard only)
BIG5 = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]


def personality_text(p: dict) -> str:
    b5 = p.get("big_five", {})
    traits = ", ".join(f"{k} {b5[k]}/5" for k in BIG5 if k in b5)
    return (f"{persona_text(p)}\nBackground: {p.get('bio', '')}\nPersonality (Big Five): {traits}.\n"
            f"Your voice: {p.get('voice', 'natural')}.")


def react_schema(factors: list[str]) -> dict:
    sch = score_schema(factors)
    sch["properties"]["reaction"] = {"type": "string", "maxLength": 200}
    sch["required"] = factors + ["reaction"]
    return sch


def react(client: "OllamaClient", image_path: str, caption: str, persona: dict) -> dict:
    """Scores plus a one-sentence in-character reaction (used by the live app, not the training jury)."""
    f = client.factors
    prompt = (f"{personality_text(persona)}\n\nThe post's caption is:\n\"\"\"{caption.strip() or '(no caption)'}\"\"\"\n\n"
              "Rate it from your point of view (integers 1-10): scroll_stop, emotional_pull, shareability, save_intent, "
              "comment_trigger. Then write `reaction`: one short sentence (max 25 words) saying what you think, "
              "in your own voice, as if commenting to a friend. Return JSON only.")
    out = client.chat(prompt, encode_image(image_path, client.max_side), fmt=react_schema(f), temperature=0.7)
    obj = json.loads(out["text"])
    return {"scores": validate_scores(obj, f), "reaction": str(obj.get("reaction", "")).strip()[:220],
            "seconds": out["seconds"], "input_tokens": out["input_tokens"], "output_tokens": out["output_tokens"]}


MODERATOR = ("You moderate a focus group of Instagram users who just reacted to a draft post. Summarise honestly "
             "and concretely. Reply with JSON only.")


def focus_group(client: "OllamaClient", caption: str, panel: list[dict]) -> dict:
    """panel: [{name, scores, reaction}]. Returns consensus, disagreement, three tips and a one-line verdict."""
    lines = "\n".join(f"- {r['name']} (avg {np.mean(list(r['scores'].values())):.1f}/10): {r['reaction']}" for r in panel)
    prompt = (f"Draft caption: \"{caption.strip()}\"\n\nPanel reactions:\n{lines}\n\n"
              "Return JSON with keys: consensus (one sentence on what most people felt), disagreement (one sentence on "
              "who disagreed and why), tips (exactly 3 short, concrete edits to the post), verdict (max 12 words).")
    schema = {"type": "object", "required": ["consensus", "disagreement", "tips", "verdict"],
              "properties": {"consensus": {"type": "string"}, "disagreement": {"type": "string"}, "verdict": {"type": "string"},
                             "tips": {"type": "array", "items": {"type": "string"}, "minItems": 3, "maxItems": 3}}}
    out = client.chat(prompt, fmt=schema, temperature=0.3, system=MODERATOR)
    return json.loads(out["text"])


# ---------------------------------------------------------------- panel mode: one call per post for all personas
def panel_prompt(personas: list[dict], caption: str, factors: list[str]) -> str:
    lines = "\n".join(f"- {p['id']}: {p['name']}, age {p['age']}; interests: {', '.join(p['interests'])}; "
                       f"{p['scrolling']} {p['saves_or_shares']}" for p in personas)
    return (f"You simulate a panel of {len(personas)} different Instagram users. Judge the attached post separately "
            f"from each person's point of view; they disagree when their tastes differ.\n\nPanel:\n{lines}\n\n"
            f"The post's caption is:\n\"\"\"{caption.strip() or '(no caption)'}\"\"\"\n\n"
            f"For every person, give integer scores 1-10 for: {', '.join(factors)}. "
            "Return JSON: {\"ratings\": {<person id>: {<factor>: <int>, ...}, ...}} covering every person id.")


def panel_schema(personas: list[dict], factors: list[str]) -> dict:
    one = score_schema(factors)
    return {"type": "object", "required": ["ratings"], "properties": {"ratings": {
        "type": "object", "required": [p["id"] for p in personas],
        "properties": {p["id"]: one for p in personas}}}}


def score_panel(client, image_path: str, caption: str, personas: list[dict]) -> dict:
    """One request scores the post for every persona. Returns {persona_id: scores} plus usage."""
    f = client.factors
    out = client.chat(panel_prompt(personas, caption, f), encode_image(image_path, client.max_side),
                      fmt=panel_schema(personas, f))
    ratings = json.loads(out["text"])["ratings"]
    out["panel"] = {p["id"]: validate_scores(ratings[p["id"]], f) for p in personas}
    return out
