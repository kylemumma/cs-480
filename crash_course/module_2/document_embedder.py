import logging
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer
from unstructured.chunking.title import chunk_by_title
from unstructured.documents.elements import Element
from unstructured.partition.pdf import partition_pdf

logger = logging.getLogger(__name__)
SUPPORTED_FILE_TYPES = [".pdf"]


class DocumentEmbedder:
    """
    This class handles document parsing, chunking, and embedding.
    i.e. takes a pdf and turns it into vector embeddings
    """

    def __init__(
        self, document_path: Path, model: str = "nomic-ai/nomic-embed-text-v1.5"
    ):
        """
        Input:
            document_path - path to the document we will generate embeddings for
            model - optional embedding model name
        """
        if not document_path.exists():
            raise ValueError(f"{document_path} does not exist")
        if document_path.suffix not in SUPPORTED_FILE_TYPES:
            raise ValueError(
                f"file type {document_path.suffix} not supported. supported file types: {SUPPORTED_FILE_TYPES}"
            )

        self._document_path = document_path
        logger.info(f"Loading model {model} ...")
        self._model = SentenceTransformer(model, trust_remote_code=True)

        # @linear build cache so it doesnt regen chunks and embeddings every time get_embedding or get_chunks is called
        #   requirements:
        #       * regen embeddings without regen chunks
        #       * regen chunks and embeddings
        #       * don't regen either, just get them from cache
        self._chunks = None
        self._embeds = None

    @classmethod
    def from_save(cls, path: Path) -> "DocumentEmbedder":
        # @linear
        #   requirements
        #   * load chunks and optional embeds from save file
        raise NotImplementedError()

    def to_parquet(self, path: Path):
        # save chunks and optional embeds to file
        raise NotImplementedError()

    def regenerate_embeddings(self):
        """
        Regenerate the embeddings from the existing chunks
        """
        raise NotImplementedError()

    def regenerate(self):
        """
        Regenerate chunks and embeddings
        """
        raise NotImplementedError()

    def get_embedding(
        self,
    ) -> NDArray:
        chunks = self.get_chunks()
        return self._embed(self._model, [e.text for e in chunks])

    def get_chunks(self) -> list[Element]:
        """
        Generate chunks from the document if they don't exist and return them
        """
        logger.info(f"Chunking {self._document_path} ...")
        elements = partition_pdf(filename=str(self._document_path), languages=["eng"])
        # @linear get rid of these magic numbers
        chunks = chunk_by_title(
            elements,
            max_characters=1500,  # hard max per chunk
            new_after_n_chars=1000,  # soft max, starts a new chunk after this
            combine_text_under_n_chars=300,  # merge small elements together
            multipage_sections=True,  # allow chunks to span pages
            overlap=150,  # character overlap between chunks
        )
        logger.info(f"Successfully generated {len(chunks)} chunks")
        return chunks

    def _embed(self, model: SentenceTransformer, chunks: list[str]) -> NDArray:
        "Given chunks return their embedding using model"
        logger.info("Embedding chunks...")
        embeddings = None
        for i in range(0, len(chunks), 256):
            new_embeds = model.encode(
                [f"search_document: {c}" for c in chunks[i : min(i + 256, len(chunks))]]
            )
            if embeddings is None:
                embeddings = new_embeds
            else:
                embeddings = np.concatenate([embeddings, new_embeds])
        logger.info(f"Successfully embedded {len(chunks)} chunks")
        if embeddings is None:
            return np.ndarray(0)
        return embeddings
