from __future__ import annotations

from typing import Any

from triagetier.oracles.base import VLMBackend
from triagetier.oracles.mock import MockVLM


def build_backend(model_cfg: dict[str, Any]) -> VLMBackend:
    kind = str(model_cfg.get("kind", "mock"))
    if kind == "mock":
        return MockVLM()
    if kind == "qwen2.5-vl":
        from triagetier.oracles.qwen25vl import Qwen25VLBackend

        return Qwen25VLBackend(
            model_id=str(model_cfg.get("model_id", "Qwen/Qwen2.5-VL-7B-Instruct")),
            precision=str(model_cfg.get("precision", "fp16")),
        )
    if kind == "internvl2.5":
        from triagetier.oracles.internvl25 import InternVL25Backend

        return InternVL25Backend(
            model_id=str(model_cfg.get("model_id", "OpenGVLab/InternVL2_5-8B")),
            precision=str(model_cfg.get("precision", "int4")),
        )
    if kind == "llava-med":
        from triagetier.oracles.llava_med import LlavaMedBackend

        return LlavaMedBackend(
            model_id=str(model_cfg.get("model_id", "microsoft/llava-med-v1.5-mistral-7b"))
        )
    raise ValueError(f"unknown backend kind '{kind}'")
