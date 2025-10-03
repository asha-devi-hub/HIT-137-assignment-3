from dataclasses import dataclass
from typing import Any
from PIL import Image

@dataclass
class ModelInfo:
    model_id: str
    task: str
    description: str

class BaseAdapter:
    def __init__(self, client, model_info: ModelInfo):
        self.client = client
        self.model_info = model_info

    def info_text(self) -> str:
        return (
            f"Model ID: {self.model_info.model_id}\n"
            f"Task: {self.model_info.task}\n\n"
            f"{self.model_info.description}\n"
        )

    def run(self, *args, **kwargs):
        raise NotImplementedError

class TextToImageAdapter(BaseAdapter):
    def run(self, prompt: str, **params) -> Image.Image:
        return self.client.text_to_image(self.model_info.model_id, prompt, **params)

class ImageClassificationAdapter(BaseAdapter):
    def run(self, image_bytes: bytes, top_k: int = 5) -> str:
        preds = self.client.image_classification(self.model_info.model_id, image_bytes, top_k=top_k)
        if not preds:
            return "No predictions."
        lines = [f"{i+1}. {p['label']} — {p['score']:.3f}" for i, p in enumerate(preds)]
        return "\n".join(lines)
