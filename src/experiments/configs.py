from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class BiqGrid:
    a_values: List[float]
    c4_scale: float  # percentage scale around default, e.g., 0.2 -> ±20%
    c2y_fixed: float
    deltas: List[float]


def default_biq_grid() -> BiqGrid:
    return BiqGrid(
        a_values=[1.2, 1.5, 1.8],
        c4_scale=0.2,
        c2y_fixed=15.66,
        deltas=[-1.2 + 0.2 * i for i in range(13)],
    )

