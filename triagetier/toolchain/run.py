from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

from triagetier.driver.orchestrator import Orchestrator
from triagetier.frontends.registry import build_dataset
from triagetier.host.artifacts import write_manifest
from triagetier.host.config import load_run_config
from triagetier.host.logsetup import configure
from triagetier.ir.records import CaseResult
from triagetier.oracles.factory import build_backend
from triagetier.toolchain._common import parse_invocation, result_to_dict


def main(argv: Sequence[str] | None = None) -> CaseResult:
    log = configure()
    name, overrides = parse_invocation(argv if argv is not None else sys.argv[1:])
    config = load_run_config(name, overrides)
    backend = build_backend(config.model)
    orchestrator = Orchestrator(backend, config.pipeline)

    cases = build_dataset(config.data)
    if not cases:
        raise RuntimeError(f"dataset for experiment '{name}' produced no cases")

    result = orchestrator.run_case(cases[0], config.seeds[0])
    log.info(
        "case=%s esi=%d iss=%d partial=%s confidence=%.3f flagged=%s qa_iters=%d",
        result.case_id,
        result.triage.esi,
        result.iss.score,
        result.iss.is_partial,
        result.confidence.score,
        result.confidence.flagged_for_review,
        result.qa.iterations,
    )
    manifest = Path("runs") / f"{name}_{result.case_id}.manifest.json"
    write_manifest(manifest, result_to_dict(result, config))
    log.info("manifest -> %s", manifest)
    return result


if __name__ == "__main__":
    main()
