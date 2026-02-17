import logging

import numpy as np
import umap

from crash_course.module_2.atlas.vector_space import VectorSpace

logger = logging.getLogger(__name__)


def transform_to_3d(space: VectorSpace) -> None:
    """
    Generates a 3d transformation for every vector
    in space, adds it to VectorSpace
    """
    logger.info("Transforming to 3d...")
    embeds_3d = reduce_to_nd(space, 3)
    space.set_positions_3d(embeds_3d)


def transform_to_2d(space: VectorSpace) -> None:
    """
    Generates a 2d transformation for every vector
    in space, adds it to VectorSpace
    """
    logger.info("Transforming to 2d...")
    embeds_2d = reduce_to_nd(space, 2)
    space.set_positions_2d(embeds_2d)


def reduce_to_nd(space: VectorSpace, n: int) -> np.ndarray:
    reducer = umap.UMAP(n_components=n, n_neighbors=7, min_dist=0.05, metric="cosine")
    res = reducer.fit_transform(space.embeddings)
    return res  # type: ignore
