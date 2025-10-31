import numpy as np
import numpy.typing as npt


def euclidean_distance(a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]) -> float:
    """Calculate Euclidean distance between two embeddings."""
    return float(np.linalg.norm(a - b))


def cosine_similarity(a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]) -> float:
    """
    Calculate cosine similarity between two normalized embeddings.
    Returns value between -1 and 1, where 1 means identical.
    Optimized for unit vectors (normalized embeddings).
    """
    # For normalized vectors, cosine similarity is just the dot product
    return float(np.dot(a, b))

