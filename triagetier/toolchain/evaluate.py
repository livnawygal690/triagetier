from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from triagetier.driver.orchestrator import Orchestrator
from triagetier.frontends.registry import build_dataset
from triagetier.host.artifacts import write_manifest
from triagetier.host.config import RunConfig, load_run_config
from triagetier.host.logsetup import configure
from triagetier.ir.records import Case, CaseResult
from triagetier.oracles.factory import build_backend
from triagetier.reports.detection_eval import mean_organ_auc, per_organ_auc
from triagetier.reports.stats import cohen_kappa, expected_calibration_error
from triagetier.reports.triage_eval import esi_exact_match, off_by_two_rate, within_one_level
from triagetier.toolchain._common import parse_invocation


def _organ_score(result: CaseResult, organ: str) -> float:
    for finding in result.findings:
        if finding.organ == organ:
            return float(finding.present)
    return 0.0


def evaluate(config: RunConfig, cases: Sequence[Case]) -> dict[str, Any]:
    backend = build_backend(config.model)
    orchestrator = Orchestrator(backend, config.pipeline)
    seed = config.seeds[0]
    rng = np.random.default_rng(seed)
    control = config.pipeline.control

    esi_pred: list[float] = []
    esi_true: list[float] = []
    confidences: list[float] = []
    correct: list[float] = []
    organ_labels: dict[str, list[float]] = {}
    organ_scores: dict[str, list[float]] = {}

    for case in cases:
        result = orchestrator.run_case(case, seed)
        truth = float(case.labels.get("esi", float("nan")))
        if control == "random":
            prediction = float(rng.integers(1, 6))
        elif control == "majority":
            prediction = 3.0
        else:
            prediction = float(result.triage.esi)
        esi_pred.append(prediction)
        esi_true.append(truth)
        confidences.append(result.confidence.score)
        correct.append(float(prediction == truth))
        for organ in case.organs:
            if organ in case.labels:
                organ_labels.setdefault(organ, []).append(case.labels[organ])
                score = (
                    0.5 if control in {"random", "majority"} else _organ_score(result, organ)
                )
                organ_scores.setdefault(organ, []).append(score)

    organ_auc = per_organ_auc(organ_labels, organ_scores, seed=seed) if organ_labels else []
    metrics: dict[str, Any] = {
        "experiment": config.name,
        "n_cases": len(cases),
        "esi_exact_match": esi_exact_match(esi_pred, esi_true),
        "esi_within_one": within_one_level(esi_pred, esi_true),
        "esi_off_by_two": off_by_two_rate(esi_pred, esi_true),
        "esi_cohen_kappa": (
            cohen_kappa(esi_pred, esi_true) if len(set(esi_true)) > 1 else float("nan")
        ),
        "ece": expected_calibration_error(confidences, correct),
        "mean_organ_auc": mean_organ_auc(organ_auc),
        "per_organ_auc": {r.organ: [r.auc, r.ci_low, r.ci_high] for r in organ_auc},
        "expected": config.expected,
    }
    return metrics


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    log = configure()
    name, overrides = parse_invocation(argv if argv is not None else sys.argv[1:])
    config = load_run_config(name, overrides)
    cases = build_dataset(config.data)
    metrics = evaluate(config, cases)
    out = Path("runs") / f"{name}_metrics.json"
    write_manifest(out, metrics)
    log.info(
        "experiment=%s n=%d esi_match=%.3f within_one=%.3f mean_auc=%.3f ece=%.3f",
        name,
        metrics["n_cases"],
        metrics["esi_exact_match"],
        metrics["esi_within_one"],
        metrics["mean_organ_auc"],
        metrics["ece"],
    )
    log.info("metrics -> %s", out)
    return metrics


if __name__ == "__main__":
    main()
