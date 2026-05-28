from __future__ import annotations

import json
from importlib import resources
from typing import Any


def _read(name: str) -> dict[str, Any]:
    with (
        resources.files("triagetier.lexicon.assets")
        .joinpath(name)
        .open("r", encoding="utf-8") as handle
    ):
        payload: dict[str, Any] = json.load(handle)
    return payload


def load_aast() -> dict[str, Any]:
    return _read("aast.json")


def load_iss_regions() -> dict[str, Any]:
    return _read("iss_regions.json")


def load_esi() -> dict[str, Any]:
    return _read("esi.json")
