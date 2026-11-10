import re
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models import KnowledgeBase, Report
from app.schemas.report import ReportCreate, ReportListItem, ReportRead
from app.services.report import NoContentError, generate_report

router = APIRouter(tags=["reports"])


def get_report_or_404(db: Session, report_id: UUID) -> Report:
    report = db.get(Report, report_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Report not found")
    return report


@router.post(
    "/knowledge-bases/{kb_id}/reports",
    response_model=ReportRead,
    status_code=status.HTTP_201_CREATED,
)
def create_report(kb_id: UUID, payload: ReportCreate, db: Session = Depends(get_db)):
    if db.get(KnowledgeBase, kb_id) is None:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    try:
        title, content = generate_report(kb_id, payload.topic.strip(), payload.title)
    except NoContentError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception:
        raise HTTPException(status_code=502, detail="The language model request failed. Try again.")

    report = Report(knowledge_base_id=kb_id, title=title[:255], content=content)
    db.add(report)
    db.commit()
    db.refresh(report)
    return report


@router.get("/knowledge-bases/{kb_id}/reports", response_model=list[ReportListItem])
def list_reports(kb_id: UUID, db: Session = Depends(get_db)):
    if db.get(KnowledgeBase, kb_id) is None:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    return db.scalars(
        select(Report)
        .where(Report.knowledge_base_id == kb_id)
        .order_by(Report.created_at.desc())
    ).all()


@router.get("/reports/{report_id}", response_model=ReportRead)
def get_report(report_id: UUID, db: Session = Depends(get_db)):
    return get_report_or_404(db, report_id)


@router.get("/reports/{report_id}/download")
def download_report(report_id: UUID, db: Session = Depends(get_db)):
    report = get_report_or_404(db, report_id)
    filename = re.sub(r"[^A-Za-z0-9_-]+", "_", report.title).strip("_")[:60] or "report"
    return Response(
        content=report.content,
        media_type="text/markdown",
        headers={"Content-Disposition": f'attachment; filename="{filename}.md"'},
    )


@router.delete("/reports/{report_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_report(report_id: UUID, db: Session = Depends(get_db)):
    report = get_report_or_404(db, report_id)
    db.delete(report)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)