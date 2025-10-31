import numpy as np
import numpy.typing as npt


def euclidean_distance(a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]) -> float:
    """Calculate Euclidean distance between two embeddings."""
    return float(np.linalg.norm(a - b))

