import numpy as np
from cartesio.core import jit

__all__ = [
    "area"
]


@jit
def area(bb: np.ndarray):
    return 1
