from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.services.vector_store import ensure_collection

@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_collection()
    yield

app = FastAPI(title="AI Knowledge & Research Assistant", lifespan=lifespan)