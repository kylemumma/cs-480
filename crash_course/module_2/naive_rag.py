import argparse
import os
from pathlib import Path

import hdbscan
import numpy as np
import pandas as pd
import plotly.express as px
import umap
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer
from unstructured.chunking.title import chunk_by_title
from unstructured.documents.elements import Element
from unstructured.partition.pdf import partition_pdf


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


def visualize(df):
    fig = px.scatter_3d(
        df,
        x="x",
        y="y",
        z="z",
        hover_data=["text"],
        opacity=0.5,
        color="cluster",
        color_discrete_map={"-1": "lightgray"},  # Override for outliers
        color_discrete_sequence=px.colors.qualitative.Bold,
    )
    # dot size
    fig.update_traces(marker=dict(size=4))

    # background color
    fig.update_layout(
        scene=dict(
            bgcolor="black",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
        )
    )

    # title
    fig.update_layout(title="Knowledge Map")

    # figure size
    fig.update_layout(width=1200, height=800)

    # remove the grid for a cleaner space-like look
    fig.update_layout(
        scene=dict(
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False),
            zaxis=dict(showgrid=False, zeroline=False),
        )
    )
    fig.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--umap", action="store_true", help="Force regenerate UMAP projection"
    )
    parser.add_argument("--chunk", action="store_true", help="Force regenerate Chunks")
    parser.add_argument(
        "--cluster", action="store_true", help="Force regenerate Clusterings"
    )
    args = parser.parse_args()

    # read and chunk pdf
    regen_chunks = args.chunk or not Path("chunks.csv").exists()
    if not regen_chunks:
        df = pd.read_csv(Path("chunks.csv"))
        print("successfully loaded dataframe")
    else:
        path = find_pdf()
        chunks = chunk(path)
        df = pd.DataFrame()
        df[["text", "page_number"]] = [(c.text, c.metadata.page_number) for c in chunks]
        df.to_csv("chunks.csv")

    # embed the chunks
    regen_embeds = regen_chunks or not Path("embeds.npy").exists()
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
    regen_clusters = args.cluster or regen_embeds or "cluster" not in df
    if regen_clusters:
        clusters = find_clusters(reduce_to_nd(embeds, 8))
        df["cluster"] = [str(e) for e in clusters.labels_]
        df.to_csv("chunks.csv")
    else:
        print("successfully loaded clusters")

    # use umap to transform to 3d (for visualization)
    regen_umap = args.umap or regen_embeds or not all(e in df for e in ["x", "y", "z"])
    if not regen_umap:
        print("successfully loaded umap")
    else:
        df[["x", "y", "z"]] = reduce_to_nd(embeds, 3)
        df.to_csv("chunks.csv")

    visualize(df)
