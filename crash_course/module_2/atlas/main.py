import logging

from crash_course.module_2.atlas.classification import classify_cluster_topics
from crash_course.module_2.atlas.cluster import generate_clusters
from crash_course.module_2.atlas.dimensionality_reducer import transform_to_3d
from crash_course.module_2.atlas.ingest import ingest_pdf
from crash_course.module_2.atlas.visualize import visualize
from crash_course.module_2.utils import find_a_pdf, where_am_i

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    with open(find_a_pdf(where_am_i(__file__)), mode="rb") as f:
        space = ingest_pdf("Fundamentals of Data Engineering", f)
    generate_clusters(space, debug=False)
    transform_to_3d(space)  # for visualization
    classify_cluster_topics(space)
    visualize(space)
else:
    print(__name__)
