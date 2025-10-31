import numpy as np
import numpy.typing as npt


def euclidean_distance(a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]) -> float:
    return float(np.linalg.norm(a - b))


def cosine_similarity(a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]) -> float:
    """Cosine similarity for normalized embeddings (dot product)."""
    return float(np.dot(a, b))

