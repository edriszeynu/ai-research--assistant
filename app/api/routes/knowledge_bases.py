from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models import Document, KnowledgeBase
from app.schemas.knowledge_base import (
    KnowledgeBaseCreate,
    KnowledgeBaseRead,
    KnowledgeBaseUpdate,
)

router = APIRouter(prefix="/knowledge-bases", tags=["knowledge-bases"])


def get_kb_or_404(db: Session, kb_id: UUID) -> KnowledgeBase:
    kb = db.get(KnowledgeBase, kb_id)
    if kb is None:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    return kb


def with_count(db: Session, kb: KnowledgeBase) -> KnowledgeBaseRead:
    count = db.scalar(
        select(func.count(Document.id)).where(Document.knowledge_base_id == kb.id)
    )
    return KnowledgeBaseRead.model_validate(kb).model_copy(update={"document_count": count or 0})


@router.post("", response_model=KnowledgeBaseRead, status_code=status.HTTP_201_CREATED)
def create_knowledge_base(payload: KnowledgeBaseCreate, db: Session = Depends(get_db)):
    kb = KnowledgeBase(name=payload.name.strip(), description=payload.description)
    db.add(kb)
    db.commit()
    db.refresh(kb)
    return with_count(db, kb)


@router.get("", response_model=list[KnowledgeBaseRead])
def list_knowledge_bases(db: Session = Depends(get_db)):
    rows = db.execute(
        select(KnowledgeBase, func.count(Document.id))
        .outerjoin(Document, Document.knowledge_base_id == KnowledgeBase.id)
        .group_by(KnowledgeBase.id)
        .order_by(KnowledgeBase.created_at.desc())
    ).all()
    return [
        KnowledgeBaseRead.model_validate(kb).model_copy(update={"document_count": count})
        for kb, count in rows
    ]


@router.get("/{kb_id}", response_model=KnowledgeBaseRead)
def get_knowledge_base(kb_id: UUID, db: Session = Depends(get_db)):
    return with_count(db, get_kb_or_404(db, kb_id))


@router.patch("/{kb_id}", response_model=KnowledgeBaseRead)
def update_knowledge_base(kb_id: UUID, payload: KnowledgeBaseUpdate, db: Session = Depends(get_db)):
    kb = get_kb_or_404(db, kb_id)
    data = payload.model_dump(exclude_unset=True)
    if data.get("name") is None:
        data.pop("name", None)  # the name can't be cleared
    for field, value in data.items():
        setattr(kb, field, value)
    db.commit()
    db.refresh(kb)
    return with_count(db, kb)


@router.delete("/{kb_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_knowledge_base(kb_id: UUID, db: Session = Depends(get_db)):
    kb = get_kb_or_404(db, kb_id)
    db.delete(kb)  # cascades to documents, conversations, messages, reports
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)