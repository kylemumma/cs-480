import argparse
import logging

import pandas as pd

from crash_course.module_2.atlas.classification import classify_cluster_topics
from crash_course.module_2.atlas.cluster import (
    generate_clusters,
    get_cluster_centroids,
    get_cluster_topics,
)
from crash_course.module_2.atlas.dimensionality_reducer import transform_to_3d
from crash_course.module_2.atlas.ingest import ingest_pdf
from crash_course.module_2.atlas.vector_space import VectorSpace
from crash_course.module_2.atlas.visualize import visualize
from crash_course.module_2.utils import find_a_pdf, where_am_i

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", metavar="SAVE_NAME", help="Load a saved vector space")
    args = parser.parse_args()

    if args.load:
        space = VectorSpace.load(args.load)
    else:
        with open(find_a_pdf(where_am_i(__file__)), mode="rb") as f:
            space = ingest_pdf("Fundamentals of Data Engineering", f)
        generate_clusters(space, debug=False)
        transform_to_3d(space)  # for visualization
        classify_cluster_topics(space)
        get_cluster_centroids(space, 5)
        space.save("fde")

    # space.set_cluster_attribute("topic", pd.Series(get_cluster_topics(space)))
    # space.save("fde")
    visualize(space)
else:
    print(__name__)
