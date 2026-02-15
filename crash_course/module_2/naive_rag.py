import argparse
import logging
import os
from pathlib import Path

import hdbscan
import numpy as np
import pandas as pd
import plotly.express as px
import umap
from document_embedder import DocumentEmbedder
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from unstructured.chunking.title import chunk_by_title
from unstructured.documents.elements import Element
from unstructured.partition.pdf import partition_pdf
from utils import find_a_pdf, where_am_i

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def find_pdf() -> Path:
    docpath = None
    for filename in os.listdir(Path(__file__).parent):
        if filename.endswith(".pdf"):
            docpath = Path(__file__).parent / filename
    if docpath is None:
        raise RuntimeError("error unable to find pdf")
    else:
        print(f"found: {docpath}")
    return docpath


# chunk
def chunk(pdf_path: Path) -> list[Element]:
    print("Chunking...")
    elements = partition_pdf(filename=str(pdf_path), languages=["eng"])
    chunks = chunk_by_title(
        elements,
        max_characters=1500,  # hard max per chunk
        new_after_n_chars=1000,  # soft max, starts a new chunk after this
        combine_text_under_n_chars=300,  # merge small elements together
        multipage_sections=True,  # allow chunks to span pages
        overlap=150,  # character overlap between chunks
    )
    print(f"Successfully chunked into {len(chunks)} chunks")
    return chunks


def embed(model: SentenceTransformer, chunks: list[str]) -> NDArray:
    embeddings = None
    for i in range(0, len(chunks), 256):
        print(f"processing {i} to {min(i + 256, len(chunks))}")
        new_embeds = model.encode(
            [f"search_document: {c}" for c in chunks[i : min(i + 256, len(chunks))]]
        )
        if embeddings is None:
            embeddings = new_embeds
        else:
            embeddings = np.concatenate([embeddings, new_embeds])
    print("done embedding")
    if embeddings is None:
        return np.ndarray(0)
    return embeddings


# pandas
def reduce_to_nd(embeddings: NDArray, n: int) -> NDArray:
    print(f"transforming to {n}-d...")
    reducer = umap.UMAP(n_components=n, n_neighbors=7, min_dist=0.05, metric="cosine")
    res = reducer.fit_transform(embeddings)
    return res  # type: ignore


def find_clusters(embeddings: NDArray):
    print("finding clusters...")
    clusterer = hdbscan.HDBSCAN(
        gen_min_span_tree=True,
        min_cluster_size=15,
    )
    res = clusterer.fit(embeddings)
    noise_ctr = 0
    cluster_set = set()
    for e in res.labels_:
        if e == -1:
            noise_ctr += 1
        else:
            cluster_set.add(e)
    print(f"successfully found clusters: {cluster_set} with {noise_ctr} outliers")
    return res


def visualize(
    title: str,
    df: pd.DataFrame,
    width: int = 1200,
    height: int = 800,
    show_noise: bool = True,
    noise_label: str = "-1: noise",
    dot_size: int = 4,
    opacity: float = 0.5,
):
    # validate input
    for col in ["cluster_label", "x", "y", "z", "keywords"]:
        if col not in df.columns:
            raise ValueError(f"column {col} not found in given dataframe")

    # graph clusters
    clusters = df[df["cluster_label"] != noise_label]
    fig = px.scatter_3d(
        clusters,
        x="x",
        y="y",
        z="z",
        opacity=opacity,
        color="cluster_label",
        color_discrete_sequence=px.colors.qualitative.Bold,
        hover_data={
            "x": False,
            "y": False,
            "z": False,
        },
    )
    fig.update_traces(marker=dict(size=dot_size))

    if show_noise:
        noise = df[df["cluster_label"] == noise_label]
        fig.add_scatter3d(
            x=noise["x"],
            y=noise["y"],
            z=noise["z"],
            mode="markers",
            marker=dict(color="lightgray", opacity=opacity * 0.7, size=dot_size - 1),
            name=noise_label,
            hovertemplate="%{fullData.name}",
            hoverlabel=dict(
                bgcolor="#cccccc",  # Dark background
                font_color="black",  # Force white text
            ),
        )

    # title
    fig.update_layout(title=title)

    # figure size
    fig.update_layout(width=width, height=height)

    # background color
    fig.update_layout(
        scene=dict(
            bgcolor="black",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
        )
    )

    # remove the grid for a cleaner space-like look
    """
    fig.update_layout(
        scene=dict(
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False),
            zaxis=dict(showgrid=False, zeroline=False),
        )
    )
    """
    fig.show()


def _get_topics_ctfidf(cluster_docs: pd.DataFrame, n_words: int = 5) -> pd.DataFrame:
    # run c-TF-IDF on each superdoc to get scores for each word
    tfidf = TfidfVectorizer(
        stop_words="english", max_df=12, token_pattern=r"\b[a-zA-Z]{3,}\b"
    )
    # each row is a cluster, each column is a word score
    tftid_matrix = tfidf.fit_transform(cluster_docs.text)
    # this correlates the word score column numbers in the matrix with the actual word
    feature_names = np.array(tfidf.get_feature_names_out())
    cluster_topics = []
    for i, cluster in enumerate(cluster_docs.cluster):
        top_indices = tftid_matrix[i].toarray()[0].argsort()[::-1][:n_words]  # type: ignore
        keywords = feature_names[top_indices]
        cluster_topics.append({"cluster": cluster, "keywords": ", ".join(keywords)})
    return pd.DataFrame(cluster_topics)


def get_cluster_topics(df: pd.DataFrame):
    # combine all text chunks for a cluster into one continuous string
    cluster_superdocs = df.groupby(["cluster"], as_index=False).agg({"text": " ".join})
    topics = _get_topics_ctfidf(cluster_superdocs)  # type: ignore
    return topics


def save_df(df: pd.DataFrame, path="data.parquet"):
    df.to_parquet(path)


def load_df(path="data.parquet") -> pd.DataFrame | None:
    if not Path(path).exists():
        return None
    return pd.read_parquet(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--regen-umap", action="store_true", help="Force regenerate UMAP projection"
    )
    parser.add_argument(
        "--regen-chunks", action="store_true", help="Force regenerate Chunks"
    )
    parser.add_argument(
        "--recluster", action="store_true", help="Force regenerate Clusterings"
    )
    args = parser.parse_args()

    print(type(__file__))

    df = load_df()
    if not args.regen_chunks and df is not None:
        logger.info("Successfully loaded saved dataframe")
    else:
        path = find_a_pdf(where_am_i(__file__))
        embedder = DocumentEmbedder(path)
        chunks = embedder.get_chunks()
        df = pd.DataFrame()
        df[["text", "page_number"]] = [(c.text, c.metadata.page_number) for c in chunks]
        save_df(df)

    # embed the chunks
    regen_embeds = args.regen_chunks or not Path("embeds.npy").exists()
    if not regen_embeds:
        embeds = np.load("embeds.npy")
        print("successfully loaded embeddings")
    else:
        model = SentenceTransformer(
            "nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True
        )
        embeds = embed(model, df["text"].tolist())
        np.save("embeds.npy", embeds)

    # HDBSCAN Clustering
    regen_clusters = args.recluster or regen_embeds or "cluster" not in df
    if regen_clusters:
        clusters = find_clusters(reduce_to_nd(embeds, 8))
        df["cluster"] = [str(e) for e in clusters.labels_]
        save_df(df)
    else:
        print("successfully loaded clusters")

    topics_df = get_cluster_topics(df)
    df = df.merge(topics_df, left_on="cluster", right_on="cluster", how="left")
    df["cluster_label"] = df["cluster"] + ": " + df["keywords"]
    df.loc[df["cluster"] == "-1", "cluster_label"] = "-1: noise"

    # use umap to transform to 3d (for visualization)
    regen_umap = (
        args.regen_umap or regen_embeds or not all(e in df for e in ["x", "y", "z"])
    )
    if not regen_umap:
        print("successfully loaded umap")
    else:
        df[["x", "y", "z"]] = reduce_to_nd(embeds, 3)
        df.to_csv("chunks.csv")

    visualize("Fundamentals of Data Engineering", df)
