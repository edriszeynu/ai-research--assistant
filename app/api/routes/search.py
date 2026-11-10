from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models import KnowledgeBase
from app.services.vector_store import search

router = APIRouter(tags=["search"])


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    limit: int = Field(default=5, ge=1, le=20)


@router.post("/knowledge-bases/{kb_id}/search")
def search_knowledge_base(kb_id: UUID, payload: SearchRequest, db: Session = Depends(get_db)):
    if db.get(KnowledgeBase, kb_id) is None:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    return {"results": search(kb_id, payload.query, payload.limit)}