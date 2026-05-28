from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class Generation:
    text: str
    token_logprobs: tuple[float, ...] = field(default_factory=tuple)


@runtime_checkable
class VLMBackend(Protocol):
    name: str

    def generate(self, prompt: str, images: Sequence[Any] = ()) -> Generation: ...
