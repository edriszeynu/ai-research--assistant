from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import knowledge_bases

app = FastAPI(title="AI Knowledge & Research Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(knowledge_bases.router, prefix="/api")


@app.get("/health")
def health():
    return {"status": "ok"}