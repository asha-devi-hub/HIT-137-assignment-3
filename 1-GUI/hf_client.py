# hf_client.py
import os, io, json
from typing import Optional, Dict
from PIL import Image
import requests

class HFClient:
    def __init__(self, token: Optional[str] = None):
        self._token = token or os.getenv("HUGGINGFACEHUB_API_TOKEN", "").strip()

    @property
    def token(self) -> str:
        return self._token

    def _json_headers(self) -> Dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self._token:
            h["Authorization"] = f"Bearer {self._token}"
        return h

    def _bin_headers(self) -> Dict[str, str]:
        h = {}
        if self._token:
            h["Authorization"] = f"Bearer {self._token}"
        return h

    # ---------- NEW: Text generation ----------
    def text_generation(self, model_id: str, prompt: str, max_new_tokens: int = 60) -> str:
        url = f"https://api-inference.huggingface.co/models/{model_id}"
        payload = {"inputs": prompt, "parameters": {"max_new_tokens": max_new_tokens}}
        r = requests.post(url, headers=self._json_headers(), data=json.dumps(payload), timeout=120)
        r.raise_for_status()
        out = r.json()
        if isinstance(out, list) and out and "generated_text" in out[0]:
            return out[0]["generated_text"]
        if isinstance(out, dict) and "generated_text" in out:
            return out["generated_text"]
        return json.dumps(out, indent=2)

    # ---------- Keep these (already in your file) ----------
    def text_to_image(self, model_id: str, prompt: str) -> Image.Image:
        url = f"https://api-inference.huggingface.co/models/{model_id}"
        r = requests.post(url, headers=self._json_headers(),
                          data=json.dumps({"inputs": prompt}), timeout=180)
        if r.status_code == 403:
            raise RuntimeError("403 Forbidden: model is gated or not available to your account/tier.")
        r.raise_for_status()
        ctype = r.headers.get("content-type", "")
        if ctype.startswith("image/"):
            return Image.open(io.BytesIO(r.content)).convert("RGB")
        try:
            err = r.json()
        except Exception:
            err = {"error": "Unknown error from the Inference API."}
        raise RuntimeError(err.get("error") or str(err))

    def image_classification(self, model_id: str, image_bytes: bytes) -> str:
        url = f"https://api-inference.huggingface.co/models/{model_id}"
        r = requests.post(url, headers=self._bin_headers(), data=image_bytes, timeout=120)
        r.raise_for_status()
        out = r.json()
        if isinstance(out, list) and out:
            topk = sorted(out, key=lambda x: x.get("score", 0), reverse=True)[:5]
            return "Top predictions:\n" + "\n".join(f"- {x['label']} ({x['score']:.3f})" for x in topk)
        return json.dumps(out, indent=2)
