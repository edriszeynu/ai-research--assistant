import uuid
from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Response, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.session import get_db
from app.models import Document, KnowledgeBase
from app.schemas.document import DocumentRead
from app.services.ingestion import ALLOWED_EXTENSIONS, chunk_text, extract_text

router = APIRouter(tags=["documents"])


def get_kb_or_404(db: Session, kb_id: UUID) -> KnowledgeBase:
    kb = db.get(KnowledgeBase, kb_id)
    if kb is None:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    return kb


def get_doc_or_404(db: Session, kb_id: UUID, doc_id: UUID) -> Document:
    doc = db.get(Document, doc_id)
    if doc is None or doc.knowledge_base_id != kb_id:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


@router.post(
    "/knowledge-bases/{kb_id}/documents",
    response_model=DocumentRead,
    status_code=status.HTTP_201_CREATED,
)
async def upload_document(
    kb_id: UUID, file: UploadFile = File(...), db: Session = Depends(get_db)
):
    get_kb_or_404(db, kb_id)

    original_name = Path(file.filename or "").name  # strip any path components
    ext = Path(original_name).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )

    content = await file.read()
    if len(content) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large")
    if not content:
        raise HTTPException(status_code=400, detail="File is empty")

    upload_dir = Path(settings.upload_dir) / str(kb_id)
    upload_dir.mkdir(parents=True, exist_ok=True)
    stored_path = upload_dir / f"{uuid.uuid4()}{ext}"  # never trust the client's name
    stored_path.write_bytes(content)

    doc = Document(
        knowledge_base_id=kb_id,
        filename=original_name,
        file_path=str(stored_path),
        file_type=ext.lstrip("."),
        size_bytes=len(content),
        status="processing",
        chunk_count=0,
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)

    try:
        text = extract_text(stored_path)
        if not text.strip():
            raise ValueError("No extractable text (scanned PDF?)")
        chunks = chunk_text(text, settings.chunk_size, settings.chunk_overlap)
        doc.chunk_count = len(chunks)
        doc.status = "processed"
    except Exception as exc:
        doc.status = "failed"
        doc.error_message = str(exc)[:500]
    db.commit()
    db.refresh(doc)
    return doc


@router.get("/knowledge-bases/{kb_id}/documents", response_model=list[DocumentRead])
def list_documents(kb_id: UUID, db: Session = Depends(get_db)):
    get_kb_or_404(db, kb_id)
    return db.scalars(
        select(Document)
        .where(Document.knowledge_base_id == kb_id)
        .order_by(Document.created_at.desc())
    ).all()


@router.get("/knowledge-bases/{kb_id}/documents/{doc_id}", response_model=DocumentRead)
def get_document(kb_id: UUID, doc_id: UUID, db: Session = Depends(get_db)):
    return get_doc_or_404(db, kb_id, doc_id)


@router.delete(
    "/knowledge-bases/{kb_id}/documents/{doc_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_document(kb_id: UUID, doc_id: UUID, db: Session = Depends(get_db)):
    doc = get_doc_or_404(db, kb_id, doc_id)
    Path(doc.file_path).unlink(missing_ok=True)
    db.delete(doc)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)