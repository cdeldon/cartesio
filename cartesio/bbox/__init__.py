import numpy as np

from ..core import jitted

__all__ = [
    "area"
]


@jitted
def area(bb: np.ndarray):
    return 1
