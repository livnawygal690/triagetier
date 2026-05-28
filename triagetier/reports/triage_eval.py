from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _arrays(pred: Sequence[float], true: Sequence[float]) -> tuple[FloatArray, FloatArray]:
    return np.asarray(pred, dtype=np.float64), np.asarray(true, dtype=np.float64)


def esi_exact_match(pred: Sequence[float], true: Sequence[float]) -> float:
    p, t = _arrays(pred, true)
    return float(np.mean(p == t)) if len(p) else float("nan")


def within_one_level(pred: Sequence[float], true: Sequence[float]) -> float:
    p, t = _arrays(pred, true)
    return float(np.mean(np.abs(p - t) <= 1)) if len(p) else float("nan")


def off_by_two_rate(pred: Sequence[float], true: Sequence[float]) -> float:
    p, t = _arrays(pred, true)
    return float(np.mean(np.abs(p - t) >= 2)) if len(p) else float("nan")
