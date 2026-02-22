import logging
from pathlib import Path

import hdbscan
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hdbscan.plots import SingleLinkageTree
from sklearn.metrics.pairwise import cosine_similarity

from crash_course.module_2.atlas import llm
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
    space.define_clusters(
        [str(e) for e in clusters.labels_],
        clusterer.condensed_tree_,
        clusterer.single_linkage_tree_,
    )


def get_n_centroids(
    space: VectorSpace, cluster_name: str, n: int = 5
) -> list[tuple[int, str]]:
    if "cluster" not in space.df:
        raise ValueError(
            "clusters are not defined yet in your vector space, see define_clusters"
        )
    if cluster_name not in space.cluster_cache:
        raise ValueError(f"Cluster '{cluster_name}' not found")
    indices = list(space.cluster_cache[cluster_name])
    cluster_embeddings = space.embeddings[indices]
    cluster_docs = space.df["chunk"].iloc[indices].tolist()

    centroid = cluster_embeddings.mean(axis=0)
    sims = cosine_similarity([centroid], cluster_embeddings)[0]
    top_k = sims.argsort()[::-1][:n]

    return [(int(i), cluster_docs[i]) for i in top_k]


def get_cluster_centroids(space: VectorSpace, n_centroids: int) -> None:
    results = [get_n_centroids(space, cluster) for cluster in space.clusters["cluster"]]
    centroid_vectors = [[idx for idx, _ in r] for r in results]
    centroid_docs = [[doc for _, doc in r] for r in results]
    space.set_cluster_attribute("centroid_vectors", pd.Series(centroid_vectors))
    space.set_cluster_attribute("centroid_docs", pd.Series(centroid_docs))


def get_cluster_topics(space: VectorSpace):
    if "centroid_docs" not in space.clusters:
        raise ValueError(
            "centroid_docs not in space.clusters, please see get_cluster_centroids"
        )
    cluster_sections = []
    for _, row in space.clusters.iterrows():
        docs = "\n".join(f"  - {doc}" for doc in row["centroid_docs"])
        cluster_sections.append(f"Cluster {row['cluster']}:\n{docs}")

    query = (
        "Below are representative excerpts for each document cluster. "
        "For each cluster, return a short topic label (3-6 words) that best describes it. "
        "Consider how clusters differ from one another when choosing labels. "
        "Reply with one line per cluster in the exact format: 'Cluster <id>: <topic>'\n\n"
        + "\n\n".join(cluster_sections)
    )
    response = llm.query_llm(query)

    topic_map = {}
    for line in response.strip().splitlines():
        if line.startswith("Cluster ") and ": " in line:
            cluster_id, topic = line.split(": ", 1)
            topic_map[cluster_id.replace("Cluster ", "").strip()] = topic.strip()

    topics = [
        topic_map.get(str(row["cluster"]), "") for _, row in space.clusters.iterrows()
    ]
    space.set_cluster_attribute("topic", pd.Series(topics))
    return topics


def single_linkage_tree_from_df(df: pd.DataFrame) -> SingleLinkageTree:
    return SingleLinkageTree(
        df[["left_child", "right_child", "distance", "size"]].values
    )


def get_min_max_distance(slt: SingleLinkageTree) -> tuple[np.float64, np.float64]:
    """returns (min, max)"""
    linkage = slt.to_numpy()
    distances = linkage[:, 2]
    out = (distances.min(), distances.max())
    logger.debug(out)
    return out
