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
        self.cluster_cache: dict[str, set[int]] = dict()
        self.clusters = (
            pd.DataFrame()
        )  # will start maintaining a 2nd table for clusters

    def set_cluster_attribute(self, name: str, series: pd.Series):
        if len(series) != len(self.clusters):
            raise ValueError(
                f"Series length {len(series)} does not match clusters length {len(self.clusters)}"
            )
        self.clusters[name] = series

    def define_clusters(self, clusters: list[str]):
        if len(clusters) != len(self.embeddings):
            raise ValueError(
                f"There are {len(self.embeddings)} embeddings but {len(clusters)} clusters were provided"
            )
        self.df["cluster"] = clusters
        self._load_clusters()

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

    def _load_clusters(self):
        self.cluster_cache = dict()
        for i, e in enumerate(self.df["cluster"]):
            curr = str(e)
            if curr not in self.cluster_cache:
                self.cluster_cache[curr] = set()
            self.cluster_cache[curr].add(i)

    def save(self, save_name: str):
        Path("save").mkdir(exist_ok=True)
        self.df.to_parquet(Path(f"save/{save_name}.parquet"))
        self.clusters.to_parquet(Path(f"save/{save_name}_clusters.parquet"))
        np.save(f"save/{save_name}.npy", self.embeddings)

    @classmethod
    def load(cls, save_name: str) -> "VectorSpace":
        instance = cls.__new__(cls)
        instance.df = pd.read_parquet(Path(f"save/{save_name}.parquet"))
        instance.embeddings = np.load(f"save/{save_name}.npy")
        instance.clusters = pd.read_parquet(Path(f"save/{save_name}_clusters.parquet"))
        instance.title = save_name
        instance._load_clusters()
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
