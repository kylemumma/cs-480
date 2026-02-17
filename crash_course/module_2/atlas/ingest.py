import logging
from typing import IO

import numpy as np
from sentence_transformers import SentenceTransformer
from unstructured.chunking.title import chunk_by_title
from unstructured.documents.elements import Element
from unstructured.partition.pdf import partition_pdf

from crash_course.module_2.atlas.vector_space import VectorSpace

logger = logging.getLogger(__name__)


def ingest_pdf(name: str, file: IO[bytes]) -> VectorSpace:
    """
    Given a file, ingests it into VectorSpace
    -- file --> [chunker] -- chunks -->
    -- chunks --> [embedding_mode] -- vectors -->
    -- vectors --> [ VectorSpace ]
    """
    logger.info("Ingesting pdf...")
    chunks = _chunk(file)
    embeddings = _embed([c.text for c in chunks])
    return VectorSpace(name, chunks, embeddings)


def _chunk(file: IO[bytes]) -> list[Element]:
    elements = partition_pdf(file=file, languages=["eng"])
    # @linear get rid of these magic numbers
    chunks = chunk_by_title(
        elements,
        max_characters=1500,  # hard max per chunk
        new_after_n_chars=1000,  # soft max, starts a new chunk after this
        combine_text_under_n_chars=300,  # merge small elements together
        multipage_sections=True,  # allow chunks to span pages
        overlap=150,  # character overlap between chunks
    )
    return chunks


def _embed(chunks: list[str]) -> np.ndarray:
    model = SentenceTransformer(
        "nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True
    )
    return model.encode([f"search_document: {c}" for c in chunks])
