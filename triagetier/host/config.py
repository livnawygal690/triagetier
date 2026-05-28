from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from omegaconf import DictConfig, OmegaConf


@dataclass(frozen=True)
class PipelineConfig:
    backbone: str = "qwen2.5-vl-7b"
    precision: str = "fp16"
    use_knowledge_graph: bool = True
    use_slice_selection: bool = True
    use_qa: bool = True
    multi_window: bool = True
    single_pass: bool = False
    slice_strategy: str = "landmark"
    k_solid: int = 7
    k_bowel: int = 10
    jitter_fraction: float = 0.15
    max_qa_iter: int = 3
    confidence_threshold: float = 0.3
    control: str = "none"


@dataclass(frozen=True)
class RunConfig:
    name: str
    model: dict[str, Any]
    data: dict[str, Any]
    pipeline: PipelineConfig
    seeds: tuple[int, ...]
    expected: dict[str, Any] = field(default_factory=dict)


def _resolve_config_dir(config_dir: str | Path | None) -> Path:
    if config_dir is not None:
        return Path(config_dir)
    here = Path.cwd()
    for candidate in (here, *here.parents):
        option = candidate / "settings"
        if (option / "experiment").is_dir():
            return option
    raise FileNotFoundError("could not locate a settings/ directory")


def _load_yaml(path: Path) -> DictConfig:
    loaded = OmegaConf.load(path)
    if not isinstance(loaded, DictConfig):
        raise TypeError(f"expected mapping config at {path}")
    return loaded


def _to_plain(config: Any) -> dict[str, Any]:
    plain = OmegaConf.to_container(config, resolve=True)
    if not isinstance(plain, dict):
        return {}
    return {str(k): v for k, v in plain.items()}


def load_run_config(
    name: str,
    overrides: Sequence[str] = (),
    config_dir: str | Path | None = None,
) -> RunConfig:
    root = _resolve_config_dir(config_dir)
    experiment = _load_yaml(root / "experiment" / f"{name}.yaml")

    pipeline_cfg = _load_yaml(root / "pipeline" / "default.yaml")
    pipeline_cfg = cast(
        DictConfig,
        OmegaConf.merge(pipeline_cfg, experiment.get("pipeline", OmegaConf.create())),
    )

    model_name = str(experiment.get("model", "mock"))
    data_name = str(experiment.get("data", "synthetic"))
    model_cfg = _load_yaml(root / "model" / f"{model_name}.yaml")
    data_cfg = _load_yaml(root / "data" / f"{data_name}.yaml")

    if overrides:
        dotlist = OmegaConf.from_dotlist(list(overrides))
        pipeline_cfg = cast(
            DictConfig,
            OmegaConf.merge(pipeline_cfg, dotlist.get("pipeline", OmegaConf.create())),
        )
        experiment = cast(DictConfig, OmegaConf.merge(experiment, dotlist))

    pipeline = PipelineConfig(**_to_plain(pipeline_cfg))
    seeds = tuple(int(s) for s in experiment.get("seeds", [0]))
    return RunConfig(
        name=name,
        model=_to_plain(model_cfg),
        data=_to_plain(data_cfg),
        pipeline=pipeline,
        seeds=seeds,
        expected=_to_plain(experiment.get("expected", OmegaConf.create())),
    )
