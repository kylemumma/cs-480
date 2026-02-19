from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from unstructured.documents.elements import Element


class VectorSpace:
    def __init__(
        self,
        title: str,
        chunks: list[Element],
        embeddings: np.ndarray,
    ):
        self.title = title
        self.df = pd.DataFrame()
        self.df["chunk"] = [e.text for e in chunks]
        self.embeddings = embeddings
        # cluster label to vector index
        self.clusters = None

    def define_clusters(self, clusters: list[str]):
        if len(clusters) != len(self.embeddings):
            raise ValueError(
                f"There are {len(self.embeddings)} embeddings but {len(clusters)} clusters were provided"
            )
        self.df["cluster"] = clusters
        # self.clusters
        self._load_cluster_cache()

    def set_positions_3d(self, matrix: np.ndarray):
        num_vectors = matrix.shape[0]
        num_dimensions = matrix.shape[1]
        if num_vectors != len(self.embeddings) or num_dimensions != 3:
            raise ValueError(
                f"Expected input matrix to have shape ({len(self.embeddings)}, 3) but got {matrix.shape}"
            )
        self.df[["3d_x", "3d_y", "3d_z"]] = matrix

    def set_positions_2d(self, matrix: np.ndarray):
        num_vectors = matrix.shape[0]
        num_dimensions = matrix.shape[1]
        if num_vectors != len(self.embeddings) or num_dimensions != 2:
            raise ValueError(
                f"Expected input matrix to have shape ({len(self.embeddings)}, 2) but got {matrix.shape}"
            )
        self.df[["2d_x", "2d_y"]] = matrix

    def get_cluster_topic(self, cluster: str, n: int = 5) -> list[str]:
        if "cluster" not in self.df:
            raise ValueError(
                "clusters are not defined yet in your vector space, see define_clusters"
            )
        if cluster not in self.df["cluster"].values:
            raise ValueError(f"Cluster '{cluster}' not found")
        mask = np.where(self.df["cluster"].values == cluster)[0]
        cluster_embeddings = self.embeddings[mask]
        cluster_docs = self.df["chunk"].iloc[mask].tolist()

        centroid = cluster_embeddings.mean(axis=0)
        sims = cosine_similarity([centroid], cluster_embeddings)[0]
        top_k = sims.argsort()[::-1][:n]

        return [cluster_docs[i] for i in top_k]

    def _load_cluster_cache(self):
        self.clusters = dict()
        for i, e in enumerate(self.df["cluster"]):
            curr = str(e)
            if curr not in self.clusters:
                self.clusters[curr] = set()
            self.clusters[curr].add(i)

    def save(self, save_name: str):
        Path("save").mkdir(exist_ok=True)
        self.df.to_parquet(Path(f"save/{save_name}.parquet"))
        np.save(f"save/{save_name}.npy", self.embeddings)

    @classmethod
    def load(cls, save_name: str) -> "VectorSpace":
        instance = cls.__new__(cls)
        instance.df = pd.read_parquet(Path(f"save/{save_name}.parquet"))
        instance.embeddings = np.load(f"save/{save_name}.npy")
        instance.title = save_name
        instance._load_cluster_cache()
        return instance


@dataclass
class Star:
    star_id: int
    text: str
    embedding: np.ndarray
    # metadata
    source: str
    cluster: str
    position_3d: np.ndarray
