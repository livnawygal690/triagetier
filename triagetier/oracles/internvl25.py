from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from triagetier.oracles.awq import quantization_kwargs
from triagetier.oracles.base import Generation
from triagetier.oracles.imaging import to_pil


@dataclass
class InternVL25Backend:
    model_id: str = "OpenGVLab/InternVL2_5-8B"
    precision: str = "int4"
    device_map: str = "auto"
    max_new_tokens: int = 256
    name: str = "internvl2.5"
    _model: Any = field(default=None, repr=False)
    _tokenizer: Any = field(default=None, repr=False)

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModel, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        self._model = AutoModel.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            device_map=self.device_map,
            **quantization_kwargs(self.precision),
        )
        self._model.eval()

    def generate(self, prompt: str, images: Sequence[Any] = ()) -> Generation:
        self._ensure_loaded()
        pil_images = [to_pil(im) for im in images]
        pixel_values = _pixel_values(pil_images, self._model.device) if pil_images else None
        config = {"max_new_tokens": self.max_new_tokens, "do_sample": False}
        answer = self._model.chat(self._tokenizer, pixel_values, prompt, config)
        return Generation(str(answer).strip(), ())


def _pixel_values(images: list[Any], device: Any) -> Any:
    import torch
    import torchvision.transforms as transforms

    transform = transforms.Compose(
        [
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    tensors = [transform(img.convert("RGB")) for img in images]
    return torch.stack(tensors).to(device).to(torch.float16)
