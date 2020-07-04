import numpy as np

__all__ = [
    "length"
]


def length(
        l: np.ndarray
) -> float:
    x1, y1, x2, y2 = l
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
