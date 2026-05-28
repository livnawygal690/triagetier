from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from triagetier.oracles.base import Generation
from triagetier.oracles.imaging import to_pil


@dataclass
class LlavaMedBackend:
    model_id: str = "microsoft/llava-med-v1.5-mistral-7b"
    device_map: str = "auto"
    max_new_tokens: int = 256
    name: str = "llava-med"
    _model: Any = field(default=None, repr=False)
    _processor: Any = field(default=None, repr=False)

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoProcessor, LlavaForConditionalGeneration

        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = LlavaForConditionalGeneration.from_pretrained(
            self.model_id, device_map=self.device_map, torch_dtype=torch.float16
        )
        self._model.eval()

    def generate(self, prompt: str, images: Sequence[Any] = ()) -> Generation:
        import torch

        self._ensure_loaded()
        pil_images = [to_pil(im) for im in images]
        tagged = ("<image>\n" * len(pil_images)) + prompt
        inputs = self._processor(
            text=tagged, images=pil_images or None, return_tensors="pt"
        ).to(self._model.device)
        with torch.no_grad():
            output = self._model.generate(
                **inputs, do_sample=False, max_new_tokens=self.max_new_tokens
            )
        prompt_len = int(inputs["input_ids"].shape[1])
        text = self._processor.tokenizer.decode(
            output[0][prompt_len:], skip_special_tokens=True
        )
        return Generation(text.strip(), ())
