from pathlib import Path
from typing import Any, List, Dict

import numpy as np
from PIL import Image
import timm

from immich_ml.models.base import InferenceModel
from immich_ml.models.transforms import decode_pil, to_numpy, normalize, resize_pil, crop_pil
from immich_ml.schemas import ModelType, ModelTask

# Coarse mapping (ImageNet label substrings -> bucket name)
ANIMAL_BUCKETS = {
    "Dog": ["dog", "labrador", "retriever", "shepherd", "chihuahua", "pug", "husky", "terrier"],
    "Cat": ["cat", "tabby", "siamese", "persian", "lynx"],
    "Bird": ["bird", "parrot", "jay", "magpie", "penguin", "ostrich", "eagle", "owl", "king penguin"],
    "Horse": ["horse", "zebra", "donkey"],
    "Cattle": ["cow", "ox", "bison", "buffalo"],
    "Sheep": ["sheep", "ram"],
    "Goat": ["goat"],
    "Pig": ["pig", "boar", "hog"],
    "Rabbit": ["hare", "rabbit"],
    "Bear": ["bear", "panda"],
    "Feline": ["tiger", "lion", "leopard", "cheetah", "jaguar"],
    "Canine": ["wolf", "fox"],
    "Rodent": ["mouse", "rat", "squirrel", "hamster"],
    "Reptile": ["snake", "lizard", "crocodile", "turtle"],
    "Fish": ["fish", "shark", "ray", "goldfish"],
    "Insect": ["butterfly", "bee", "ant", "dragonfly", "ladybug", "beetle", "mosquito"],
}


class AnimalClassifier(InferenceModel):
    """
    Minimal ImageNet-based classifier that maps ImageNet class labels into coarse animal buckets.
    identity: recognition / animal-recognition
    """

    depends = []
    identity = (ModelType.RECOGNITION, ModelTask.ANIMAL_RECOGNITION)

    def __init__(self, model_name: str = "resnet34", **kwargs: Any) -> None:
        self.model_name = model_name
        self.model = None
        super().__init__(model_name, **kwargs)

    def _load(self) -> None:
        # Create timm model and eval mode
        self.model = timm.create_model(self.model_name, pretrained=True)
        self.model.eval()
        # Build a minimal preprocess used below
        # Use 224x224 standard
        self.input_size = (3, 224, 224)
        return super()._load()

    def predict(self, inputs: Image.Image, options: dict | None = None) -> list[Dict[str, float]]:
        # inputs: PIL.Image
        img = inputs.convert("RGB")
        img = resize_pil(img, (224, 224))
        img = crop_pil(img, (224, 224))
        arr = to_numpy(img).astype("float32") / 255.0
        # normalize using ImageNet standard
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = normalize(arr, mean, std)
        x = np.expand_dims(arr.transpose(2, 0, 1), 0)  # (1,C,H,W)
        import torch

        with torch.no_grad():
            t = torch.from_numpy(x)
            logits = self.model(t).cpu().numpy()[0]
            probs = np.exp(logits - np.max(logits))
            probs = probs / probs.sum()

        # Attempt to get ImageNet labels from timm if available
        try:
            from timm.data import IMAGENET_DEFAULT_LABELS as IMAGENET_LABELS  # type: ignore
            labels = IMAGENET_LABELS
        except Exception:
            # fallback simple labels
            labels = [f"class_{i}" for i in range(len(probs))]

        # map top-K labels to animal buckets
        topk = np.argsort(probs)[-20:][::-1]  # top 20
        found: dict[str, float] = {}
        for idx in topk:
            label = labels[idx].lower()
            score = float(probs[idx])
            for bucket, keywords in ANIMAL_BUCKETS.items():
                for k in keywords:
                    if k in label:
                        # keep max score for bucket
                        if bucket not in found or found[bucket] < score:
                            found[bucket] = score

        # convert to list of dicts sorted by score desc
        result = [{"label": key, "score": float(found[key])} for key in sorted(found, key=found.get, reverse=True)]
        return result

    def configure(self, **kwargs: Any) -> None:
        # nothing to configure for now
        return
