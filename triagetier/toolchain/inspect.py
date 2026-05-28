from __future__ import annotations

import sys
from collections.abc import Sequence

import numpy as np

from triagetier.frontends.registry import build_dataset
from triagetier.host.config import load_run_config
from triagetier.host.logsetup import configure
from triagetier.lexicon.graph import load_graph
from triagetier.lowering.localize import localize_by_name
from triagetier.lowering.sampling import percentiles_for, select_slices
from triagetier.toolchain._common import parse_invocation


def main(argv: Sequence[str] | None = None) -> None:
    log = configure()
    name, overrides = parse_invocation(argv if argv is not None else sys.argv[1:])
    config = load_run_config(name, overrides)
    graph = load_graph()
    cases = build_dataset(config.data)
    case = cases[0]
    log.info("case=%s modality=%s organs=%s", case.case_id, case.modality, case.organs)

    if case.volume is not None:
        rng = np.random.default_rng(config.seeds[0])
        for organ in case.organs:
            extent = localize_by_name(case.volume, organ)
            slices = select_slices(
                extent, percentiles_for(organ), rng, config.pipeline.jitter_fraction
            )
            log.info(
                "organ=%s extent=[%d,%d] found=%s slices=%s",
                organ,
                extent.z_start,
                extent.z_end,
                extent.found,
                slices,
            )

    thresholds = graph.esi_thresholds()
    log.info("imaged_regions=%s", graph.imaged_regions())
    log.info(
        "esi_thresholds lifesaving>=AIS%d high_risk>=AIS%d or ISS%d multi_resource>=ISS%d",
        thresholds.lifesaving_min_ais,
        thresholds.high_risk_min_ais,
        thresholds.high_risk_min_iss,
        thresholds.multi_resource_min_iss,
    )


if __name__ == "__main__":
    main()
