"""FastAPI app: POST /chat for messages, GET / for the web chat page."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .agent import BookingAgent, build_default_agent

STATIC_DIR = Path(__file__).parent / "static"
log = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=100, pattern=r"^[A-Za-z0-9_-]+$")
    message: str = Field(min_length=1, max_length=4000)


class ChatResponse(BaseModel):
    reply: str
    tool_calls: list[str]
    latency_ms: int
    input_tokens: int
    output_tokens: int
    guardrail: str | None = None
    error: str | None = None


def create_app(agent: BookingAgent | None = None) -> FastAPI:
    logging.basicConfig(level=logging.INFO)
    app = FastAPI(title="AI Booking Agent", version="1.0.0")
    app.state.agent = agent

    def get_agent() -> BookingAgent:
        if app.state.agent is None:
            try:
                app.state.agent = build_default_agent()
            except Exception as exc:
                log.error("Could not start the agent: %s", exc)
                raise HTTPException(
                    status_code=503,
                    detail="Agent is not configured. Set LLM_MODEL and your provider's API key in .env.",
                ) from exc
        return app.state.agent

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    # Sync endpoint: FastAPI runs it in a thread pool, so slow model calls don't block other requests.
    @app.post("/chat", response_model=ChatResponse)
    def chat(req: ChatRequest) -> ChatResponse:
        result = get_agent().chat(req.session_id, req.message)
        return ChatResponse(**result.to_dict())

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    return app


app = create_app()
