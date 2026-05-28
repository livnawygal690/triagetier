from __future__ import annotations

from pathlib import Path

import pytest

from triagetier.lexicon.graph import KnowledgeGraph, load_graph

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "settings"


@pytest.fixture(scope="session")
def graph() -> KnowledgeGraph:
    return load_graph()


@pytest.fixture(scope="session")
def config_dir() -> Path:
    return CONFIG_DIR
