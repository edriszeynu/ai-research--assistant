from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class DocumentRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    knowledge_base_id: UUID
    filename: str
    file_type: str
    size_bytes: int
    status: str
    chunk_count: int
    error_message: str | None = None
    created_at: datetime