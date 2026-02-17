import logging

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from crash_course.module_2.atlas.vector_space import VectorSpace

logger = logging.getLogger(__name__)


def classify_cluster_topics(space: VectorSpace):
    # todo: this shouldnt access internal details of the VectorSpace object
    logger.info("Generating cluster topics...")
    topics_df = _get_cluster_topics(space.df)
    space.df = space.df.merge(
        topics_df, left_on="cluster", right_on="cluster", how="left"
    )


def _get_cluster_topics(df: pd.DataFrame):
    # combine all text chunks for a cluster into one continuous string
    cluster_superdocs = df.groupby(["cluster"], as_index=False).agg({"chunk": " ".join})
    topics = _get_topics_ctfidf(cluster_superdocs)  # type: ignore
    return topics


def _get_topics_ctfidf(cluster_docs: pd.DataFrame, n_words: int = 5) -> pd.DataFrame:
    # run c-TF-IDF on each superdoc to get scores for each word
    tfidf = TfidfVectorizer(
        stop_words="english", max_df=0.8, token_pattern=r"\b[a-zA-Z]{2}[a-zA-Z0-9]*\b"
    )
    # each row is a cluster, each column is a word score
    tftid_matrix = tfidf.fit_transform(cluster_docs["chunk"])
    # this correlates the word score column numbers in the matrix with the actual word
    feature_names = np.array(tfidf.get_feature_names_out())
    cluster_topics = []
    for i, cluster in enumerate(cluster_docs.cluster):
        top_indices = tftid_matrix[i].toarray()[0].argsort()[::-1][:n_words]  # type: ignore
        keywords = feature_names[top_indices]
        cluster_topics.append({"cluster": cluster, "topic": ", ".join(keywords)})
    return pd.DataFrame(cluster_topics)
