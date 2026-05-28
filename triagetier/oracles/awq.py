from __future__ import annotations

from typing import Any


def awq_config(bits: int) -> Any:
    from transformers import AwqConfig

    if bits not in (4, 8):
        raise ValueError(f"AWQ supports 4 or 8 bit, got {bits}")
    return AwqConfig(bits=bits, group_size=128, version="gemm")


def quantization_kwargs(precision: str) -> dict[str, Any]:
    import torch

    if precision == "fp16":
        return {"torch_dtype": torch.float16}
    if precision == "int8":
        return {"quantization_config": awq_config(8)}
    if precision == "int4":
        return {"quantization_config": awq_config(4)}
    raise ValueError(f"unknown precision '{precision}'")
