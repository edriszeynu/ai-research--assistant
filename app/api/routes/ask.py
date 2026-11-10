from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models import KnowledgeBase
from app.services.rag import answer_question

router = APIRouter(tags=["ask"])


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    limit: int | None = Field(default=None, ge=1, le=20)


@router.post("/knowledge-bases/{kb_id}/ask")
def ask(kb_id: UUID, payload: AskRequest, db: Session = Depends(get_db)):
    if db.get(KnowledgeBase, kb_id) is None:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    try:
        return answer_question(kb_id, payload.question.strip(), payload.limit)
    except Exception:
        raise HTTPException(status_code=502, detail="The language model request failed. Try again.")