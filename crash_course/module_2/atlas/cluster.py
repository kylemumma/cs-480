import logging

import hdbscan
import matplotlib.pyplot as plt

from crash_course.module_2.atlas.dimensionality_reducer import reduce_to_nd
from crash_course.module_2.atlas.vector_space import VectorSpace

logger = logging.getLogger(__name__)


def generate_clusters(space: VectorSpace, debug: bool = False) -> None:
    """
    Classifies each vector in space with a cluster.
    Debug flag will graph the condensed tree and single linkage tree.
    """
    logger.info("Generating clusters...")
    embeds_8d = reduce_to_nd(space, 8)
    clusterer = hdbscan.HDBSCAN(
        gen_min_span_tree=True,
        min_cluster_size=15,
    )
    clusters = clusterer.fit(embeds_8d)
    if debug:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        clusterer.condensed_tree_.plot(axis=ax1)
        ax1.set_title("Condensed Tree")
        clusterer.single_linkage_tree_.plot(axis=ax2)
        ax2.set_title("Single Linkage Tree")
        plt.tight_layout()
        plt.show()
    space.define_clusters([str(e) for e in clusters.labels_])

    # todo modify spacei
    #
