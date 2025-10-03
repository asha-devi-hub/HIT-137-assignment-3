# hf_client.py
import io
from typing import List, Dict, Any
from PIL import Image
import torch
from diffusers import DiffusionPipeline
from transformers import AutoImageProcessor, AutoModelForImageClassification

_T2I_PIPE = None
_CLS_PROC = None
_CLS_MODEL = None

def _device_dtype():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    return device, dtype

def _get_t2i_pipe(model_id: str):
    global _T2I_PIPE
    if _T2I_PIPE is None:
        device, dtype = _device_dtype()
        _T2I_PIPE = DiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype).to(device)
        try:
            _T2I_PIPE.enable_vae_slicing()
            _T2I_PIPE.enable_attention_slicing()
        except Exception:
            pass
    return _T2I_PIPE

class HFClient:
    def __init__(self, token: str = ""):
        self.token = token or ""

    # --- Text -> Image (uses aiyouthalliance/Free-Image-Generation) ---
    def text_to_image(self, model_id: str, prompt: str, **gen_kwargs) -> Image.Image:
        pipe = _get_t2i_pipe(model_id)
        # sensible defaults; tweak if you want
        steps = int(gen_kwargs.get("num_inference_steps", 28))
        guidance = float(gen_kwargs.get("guidance_scale", 7.5))
        neg = gen_kwargs.get("negative_prompt")  # optional
        out = pipe(prompt, num_inference_steps=steps, guidance_scale=guidance, negative_prompt=neg)
        return out.images[0]

    # --- Image Classification (unchanged, local) ---
    def image_classification(self, model_id: str, image_bytes: bytes, top_k: int = 5) -> List[Dict[str, Any]]:
        global _CLS_PROC, _CLS_MODEL
        if _CLS_PROC is None or _CLS_MODEL is None:
            _CLS_PROC = AutoImageProcessor.from_pretrained(model_id)
            _CLS_MODEL = AutoModelForImageClassification.from_pretrained(model_id)
            _CLS_MODEL.eval()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        inputs = _CLS_PROC(images=img, return_tensors="pt")
        with torch.no_grad():
            logits = _CLS_MODEL(**inputs).logits
        probs = logits.softmax(-1)[0]
        k = min(top_k, probs.shape[-1])
        values, indices = torch.topk(probs, k=k)
        id2label = _CLS_MODEL.config.id2label
        return [{"label": id2label[int(i.item())], "score": float(v.item())}
                for v, i in zip(values, indices)]
