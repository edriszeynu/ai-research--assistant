from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.agents.graph import run_agent
from app.db.session import get_db
from app.models import Conversation, Message, Report
from app.schemas.conversation import MessageRead, SendMessage

router = APIRouter(tags=["agent"])


class AgentReply(BaseModel):
    action: str
    message: MessageRead
    report_id: UUID | None = None


@router.post("/conversations/{conv_id}/agent", response_model=AgentReply)
def agent_message(conv_id: UUID, payload: SendMessage, db: Session = Depends(get_db)):
    conv = db.get(Conversation, conv_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    question = payload.question.strip()

    recent = db.scalars(
        select(Message)
        .where(Message.conversation_id == conv_id)
        .order_by(Message.created_at.desc())
        .limit(6)
    ).all()
    history = [{"role": m.role, "content": m.content} for m in reversed(recent)]

    try:
        state = run_agent(conv.knowledge_base_id, question, history)
    except Exception:
        raise HTTPException(status_code=502, detail="The language model request failed. Try again.")

    report_id = None
    if state.get("report"):
        report = Report(
            knowledge_base_id=conv.knowledge_base_id,
            title=state["report"]["title"][:255],
            content=state["report"]["content"],
        )
        db.add(report)
        db.flush()  # assigns report.id
        report_id = report.id

    answer = state["answer"]
    if report_id:
        answer += f" (report id: {report_id})"

    db.add(Message(conversation_id=conv_id, role="user", content=question))
    assistant = Message(
        conversation_id=conv_id,
        role="assistant",
        content=answer,
        sources=state.get("sources") or [],
    )
    db.add(assistant)
    if not conv.title:
        conv.title = question[:60]
    db.commit()
    db.refresh(assistant)
    return AgentReply(action=state["action"], message=assistant, report_id=report_id)