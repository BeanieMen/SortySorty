import math
from typing import List
import numpy as np
import numpy.typing as npt


def cosine_similarity(a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]) -> float:
    if len(a) != len(b):
        return 0.0
    
    dot_product = np.dot(a, b)
    magnitude_a = np.linalg.norm(a)
    magnitude_b = np.linalg.norm(b)
    
    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0
    
    return float(dot_product / (magnitude_a * magnitude_b))


def euclidean_distance(a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]) -> float:
    if len(a) != len(b):
        return float('inf')
    
    return float(np.linalg.norm(a - b))

