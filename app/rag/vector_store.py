import chromadb

from .chunker import SubtitleChunk
from .embeddings import EmbeddingModel


CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "movie_subtitles"


class VectorStore:

    def __init__(
        self,
        persist_directory: str = CHROMA_PATH,
    ):
        self.client = chromadb.PersistentClient(
            path=persist_directory
        )

        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={
                "description": "Movie subtitle chunks for RAG"
            },
        )

        self.embedding_model = EmbeddingModel()

    def add_chunks(
        self,
        chunks: list[SubtitleChunk],
    ) -> None:

        if not chunks:
            return

        documents = [
            chunk.text
            for chunk in chunks
        ]

        embeddings = self.embedding_model.embed_texts(
            documents
        )

        ids = [
            chunk.chunk_id
            for chunk in chunks
        ]

        metadatas = [
            {
                "movie_title": chunk.movie_title,
                "start_time": chunk.start_time,
                "end_time": chunk.end_time,
                "start_seconds": chunk.start_seconds,
                "end_seconds": chunk.end_seconds,
            }
            for chunk in chunks
        ]

        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def count(self) -> int:
        return self.collection.count()

    def search(
        self,
        query: str,
        n_results: int = 5,
        where: dict | None = None,
    ):
        query_embedding = self.embedding_model.embed_query(
            query
        )

        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
        )