from typing import List
import numpy as np
import numpy.typing as npt
from sklearn.cluster import DBSCAN

from ..types.scan import FaceCluster


class ClusterService:
    
    def __init__(self, eps: float = 0.4, min_samples: int = 2):
        self.eps = eps
        self.min_samples = min_samples
    
    def cluster_faces(
        self,
        embeddings: List[npt.NDArray[np.float64]]
    ) -> List[FaceCluster]:
        if len(embeddings) < self.min_samples:
            return []
        
        X = np.array(embeddings)
        clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric='cosine')
        labels = clustering.fit_predict(X)
        
        clusters: dict[int, List[npt.NDArray[np.float64]]] = {}
        for idx, label in enumerate(labels):
            if label == -1:
                continue
            
            if label not in clusters:
                clusters[label] = []
            
            clusters[label].append(embeddings[idx])
        
        result: List[FaceCluster] = []
        for cluster_id, members in clusters.items():
            if len(members) >= self.min_samples:
                centroid = np.mean(members, axis=0)
                
                cluster = FaceCluster(
                    members=members,
                    representative={
                        "clusterId": int(cluster_id),
                        "size": len(members),
                        "centroid": centroid.tolist()
                    }
                )
                result.append(cluster)
        
        return result
