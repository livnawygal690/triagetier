from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict
from typing import Any

from triagetier.host.config import RunConfig
from triagetier.ir.records import CaseResult


def parse_invocation(argv: Sequence[str]) -> tuple[str, list[str]]:
    name = "main"
    overrides: list[str] = []
    for token in argv:
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        if key == "experiment":
            name = value
        else:
            overrides.append(token)
    return name, overrides


def result_to_dict(result: CaseResult, config: RunConfig) -> dict[str, Any]:
    return {
        "experiment": config.name,
        "seed": result.seed,
        "backbone": config.pipeline.backbone,
        "precision": config.pipeline.precision,
        "case": asdict(result),
    }
