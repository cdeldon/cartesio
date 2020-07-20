import numpy as np
from cartesio.bbox import area

if __name__ == '__main__':
    a = area(np.array([1, 2, 3, 4]))
    print(a)
