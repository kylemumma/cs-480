from dataclasses import dataclass, field

import numpy as np
import pandas as pd
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

    def define_clusters(self, clusters: list[str]):
        if len(clusters) != len(self.embeddings):
            raise ValueError(
                f"There are {len(self.embeddings)} embeddings but {len(clusters)} clusters were provided"
            )
        self.df["cluster"] = clusters

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


@dataclass
class Star:
    star_id: int
    text: str
    embedding: np.ndarray
    # metadata
    source: str
    cluster: str
    position_3d: np.ndarray
