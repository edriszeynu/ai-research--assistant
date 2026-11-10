import uuid
from functools import lru_cache

from qdrant_client import QdrantClient, models

from app.core.config import settings
from app.services.llm import get_embeddings

BATCH_SIZE = 64


@lru_cache
def get_client() -> QdrantClient:
    return QdrantClient(url=settings.qdrant_url)


def ensure_collection() -> None:
    client = get_client()
    if client.collection_exists(settings.qdrant_collection):
        return
    client.create_collection(
        collection_name=settings.qdrant_collection,
        vectors_config=models.VectorParams(
            size=settings.embedding_dimensions, distance=models.Distance.COSINE
        ),
    )
    for field in ("knowledge_base_id", "document_id"):
        client.create_payload_index(
            settings.qdrant_collection,
            field_name=field,
            field_schema=models.PayloadSchemaType.KEYWORD,
        )


def index_chunks(kb_id, doc_id, filename: str, chunks: list[str]) -> None:
    client = get_client()
    embeddings = get_embeddings()
    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start : start + BATCH_SIZE]
        vectors = embeddings.embed_documents(batch)
        points = [
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=vec,
                payload={
                    "knowledge_base_id": str(kb_id),
                    "document_id": str(doc_id),
                    "filename": filename,
                    "chunk_index": start + i,
                    "text": text,
                },
            )
            for i, (text, vec) in enumerate(zip(batch, vectors))
        ]
        client.upsert(settings.qdrant_collection, points=points)


def _delete_where(field: str, value) -> None:
    get_client().delete(
        settings.qdrant_collection,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[models.FieldCondition(key=field, match=models.MatchValue(value=str(value)))]
            )
        ),
    )


def delete_document_vectors(doc_id) -> None:
    _delete_where("document_id", doc_id)


def delete_kb_vectors(kb_id) -> None:
    _delete_where("knowledge_base_id", kb_id)


def search(kb_id, query: str, limit: int = 5) -> list[dict]:
    vector = get_embeddings().embed_query(query)
    result = get_client().query_points(
        settings.qdrant_collection,
        query=vector,
        limit=limit,
        query_filter=models.Filter(
            must=[models.FieldCondition(key="knowledge_base_id", match=models.MatchValue(value=str(kb_id)))]
        ),
        with_payload=True,
    )
    return [
        {
            "score": p.score,
            "document_id": p.payload["document_id"],
            "filename": p.payload["filename"],
            "chunk_index": p.payload["chunk_index"],
            "text": p.payload["text"],
        }
        for p in result.points
    ]