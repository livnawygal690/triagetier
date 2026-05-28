from __future__ import annotations

from collections.abc import Mapping

from triagetier.lexicon.graph import ESIThresholds


def vitals_unstable(vitals: Mapping[str, float], thresholds: ESIThresholds) -> bool:
    sbp = vitals.get("systolic_bp")
    hr = vitals.get("heart_rate")
    hypotensive = sbp is not None and sbp < thresholds.shock_systolic_bp
    tachycardic = hr is not None and hr > thresholds.tachycardia_hr
    return bool(hypotensive or tachycardic)


def esi_level(
    iss: int,
    max_ais: int,
    vitals: Mapping[str, float],
    thresholds: ESIThresholds,
) -> tuple[int, tuple[str, ...]]:
    unstable = vitals_unstable(vitals, thresholds)

    if max_ais >= thresholds.lifesaving_min_ais or (
        unstable and max_ais >= thresholds.high_risk_min_ais
    ):
        return 1, ("life-saving intervention indicated",)

    if (
        max_ais >= thresholds.high_risk_min_ais
        or iss >= thresholds.high_risk_min_iss
        or unstable
    ):
        drivers = []
        if max_ais >= thresholds.high_risk_min_ais:
            drivers.append(f"high-grade injury (AIS {max_ais})")
        if iss >= thresholds.high_risk_min_iss:
            drivers.append(f"ISS {iss}")
        if unstable:
            drivers.append("unstable vitals")
        return 2, tuple(drivers)

    if iss >= thresholds.multi_resource_min_iss:
        return 3, (f"ISS {iss}; multiple resources anticipated",)

    if iss >= thresholds.single_resource_min_iss:
        return 4, (f"ISS {iss}; single resource anticipated",)

    return 5, ("no injury severity above triage floor",)
