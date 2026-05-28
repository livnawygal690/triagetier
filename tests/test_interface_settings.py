from __future__ import annotations

from pathlib import Path

import pytest

from triagetier.host.config import load_run_config
from triagetier.toolchain import evaluate as evaluate_cli
from triagetier.toolchain import run as run_cli


def test_load_smoke(config_dir: Path) -> None:
    config = load_run_config("_smoke", config_dir=config_dir)
    assert config.model["kind"] == "mock"
    assert config.data["kind"] == "synthetic"
    assert config.seeds == (0,)


def test_load_main_expected(config_dir: Path) -> None:
    config = load_run_config("main", config_dir=config_dir)
    assert config.expected["ratic_mean_organ_auc"] == 0.792
    assert config.pipeline.use_qa is True
    assert config.pipeline.backbone == "qwen2.5-vl-7b"


def test_override_pipeline_flag(config_dir: Path) -> None:
    config = load_run_config("_smoke", ["pipeline.use_qa=false"], config_dir)
    assert config.pipeline.use_qa is False


def test_run_cli_smoke(monkeypatch: pytest.MonkeyPatch, config_dir: Path) -> None:
    monkeypatch.chdir(config_dir.parent)
    result = run_cli.main(["experiment=_smoke"])
    assert 1 <= result.triage.esi <= 5


def test_evaluate_cli_smoke(monkeypatch: pytest.MonkeyPatch, config_dir: Path) -> None:
    monkeypatch.chdir(config_dir.parent)
    metrics = evaluate_cli.main(["experiment=_smoke"])
    assert metrics["n_cases"] == 6
    assert "mean_organ_auc" in metrics
    assert "esi_within_one" in metrics
