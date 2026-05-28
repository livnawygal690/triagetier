from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from triagetier.oracles.awq import quantization_kwargs
from triagetier.oracles.base import Generation
from triagetier.oracles.imaging import to_pil


@dataclass
class Qwen25VLBackend:
    model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    precision: str = "fp16"
    device_map: str = "auto"
    max_new_tokens: int = 256
    name: str = "qwen2.5-vl"
    _model: Any = field(default=None, repr=False)
    _processor: Any = field(default=None, repr=False)

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id, device_map=self.device_map, **quantization_kwargs(self.precision)
        )
        self._model.eval()

    def generate(self, prompt: str, images: Sequence[Any] = ()) -> Generation:
        import torch

        self._ensure_loaded()
        pil_images = [to_pil(im) for im in images]
        content: list[dict[str, Any]] = [{"type": "image"} for _ in pil_images]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        chat = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(
            text=[chat], images=pil_images or None, return_tensors="pt"
        ).to(self._model.device)

        with torch.no_grad():
            output = self._model.generate(
                **inputs,
                do_sample=False,
                temperature=0.0,
                max_new_tokens=self.max_new_tokens,
                output_scores=True,
                return_dict_in_generate=True,
            )

        prompt_len = int(inputs["input_ids"].shape[1])
        sequence = output.sequences[0][prompt_len:]
        text = self._processor.tokenizer.decode(sequence, skip_special_tokens=True)
        logprobs = _sequence_logprobs(output.scores, sequence)
        return Generation(text.strip(), logprobs)


def _sequence_logprobs(scores: Any, sequence: Any) -> tuple[float, ...]:
    import torch

    values: list[float] = []
    for step, logits in enumerate(scores):
        if step >= len(sequence):
            break
        logp = torch.log_softmax(logits[0], dim=-1)
        values.append(float(logp[int(sequence[step])]))
    return tuple(values)
