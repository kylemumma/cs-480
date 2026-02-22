from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from hdbscan.plots import CondensedTree, SingleLinkageTree
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
        self.cluster_cache: dict[str, set[int]] = dict()  # todo: can I get rid of this?
        self.clusters = (
            pd.DataFrame()
        )  # will start maintaining a 2nd table for clusters
        self.has_position_3d = False

    def as_df(self) -> pd.DataFrame:
        if "topic" not in self.df:
            self.df["topic"] = ""
        return self.df

    def set_cluster_attribute(self, name: str, series: pd.Series):
        if len(series) != len(self.clusters):
            raise ValueError(
                f"Series length {len(series)} does not match clusters length {len(self.clusters)}"
            )
        self.clusters[name] = series

    def define_clusters(
        self,
        clusters: list[str],
        condensed_tree: CondensedTree,
        single_linkage_tree: SingleLinkageTree,
    ):
        if len(clusters) != len(self.embeddings):
            raise ValueError(
                f"There are {len(self.embeddings)} embeddings but {len(clusters)} clusters were provided"
            )
        self.df["cluster"] = clusters
        self.clusters["cluster"] = clusters
        self._load_clusters()
        self.condensed_tree = condensed_tree
        self.single_linkage_tree = single_linkage_tree

    def set_positions_3d(self, matrix: np.ndarray):
        num_vectors = matrix.shape[0]
        num_dimensions = matrix.shape[1]
        if num_vectors != len(self.embeddings) or num_dimensions != 3:
            raise ValueError(
                f"Expected input matrix to have shape ({len(self.embeddings)}, 3) but got {matrix.shape}"
            )
        self.df[["3d_x", "3d_y", "3d_z"]] = matrix
        self.has_position_3d = True

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
        root = Path(f"save/{save_name}")
        root.mkdir(exist_ok=True)
        self.df.to_parquet(root / "df.parquet")
        self.clusters.to_parquet(root / "clusters.parquet")
        self.single_linkage_tree.to_pandas().to_parquet(root / "slt.parquet")
        self.condensed_tree.to_pandas().to_parquet(root / "ct.parquet")
        np.save(root / "embeds.npy", self.embeddings)

    @classmethod
    def load(cls, save_name: str) -> "VectorSpace":
        from crash_course.module_2.atlas.cluster import single_linkage_tree_from_df

        root = Path(f"save/{save_name}")
        instance = cls.__new__(cls)
        instance.df = pd.read_parquet(root / "df.parquet")
        instance.embeddings = np.load(root / "embeds.npy")

        # clusters
        instance.clusters = pd.read_parquet(root / "clusters.parquet")
        ct_df = pd.read_parquet(root / "ct.parquet")
        labels = instance.df["cluster"].astype(int).values
        instance.condensed_tree = CondensedTree(ct_df.to_records(index=False), labels)
        instance.single_linkage_tree = single_linkage_tree_from_df(
            pd.read_parquet(root / "slt.parquet")
        )
        instance.has_position_3d = all(
            e in instance.df.columns for e in ["3d_x", "3d_y", "3d_z"]
        )
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
