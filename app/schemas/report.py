from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ReportCreate(BaseModel):
    topic: str = Field(min_length=1, max_length=500)
    title: str | None = Field(default=None, max_length=255)


class ReportListItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    knowledge_base_id: UUID
    title: str
    created_at: datetime


class ReportRead(ReportListItem):
    content: str