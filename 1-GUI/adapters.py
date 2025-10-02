# adapters.py
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Union
from PIL import Image
from hf_client import HFClient
from decorators import require_token

@dataclass
class ModelInfo:
    model_id: str
    task: str
    description: str

class BaseAdapter(ABC):
    """Abstract base demonstrating polymorphism + overriding."""
    def __init__(self, client: HFClient, info: ModelInfo):
        self.client = client
        self.info = info

    @abstractmethod
    def run(self, **kwargs) -> Union[str, Image.Image]:
        ...

    def info_text(self) -> str:
        return f"Model: {self.info.model_id}\nCategory: {self.info.task}\n\n{self.info.description}"

class TextToImageAdapter(BaseAdapter):
    @require_token
    def run(self, **kwargs) -> Image.Image:
        prompt = (kwargs.get("prompt") or "").strip()
        if not prompt:
            raise ValueError("Please type a prompt in the input box.")
        return self.client.text_to_image(self.info.model_id, prompt)

class ImageClassificationAdapter(BaseAdapter):
    @require_token
    def run(self, **kwargs) -> str:
        img_bytes = kwargs.get("image_bytes")
        if not img_bytes:
            raise ValueError("Please choose an image (Browse) first.")
        return self.client.image_classification(self.info.model_id, img_bytes)
