"""FastAPI app for Deepzzzz. Runs locally for one user (you)."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .agent import Deepzzzz, build_default
from .store import StoreError

STATIC_DIR = Path(__file__).parent / "static"
log = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)


class ChatResponse(BaseModel):
    reply: str
    tool_calls: list[str]
    latency_ms: int
    input_tokens: int
    output_tokens: int
    guardrail: str | None = None
    error: str | None = None


def create_app(bot: Deepzzzz | None = None) -> FastAPI:
    logging.basicConfig(level=logging.INFO)
    app = FastAPI(title="Deepzzzz", version="2.0.0")
    app.state.bot = bot

    def get_bot() -> Deepzzzz:
        if app.state.bot is None:
            try:
                app.state.bot = build_default()
            except Exception as exc:
                log.error("Could not start Deepzzzz: %s", exc)
                reason = str(exc).splitlines()[0][:200] if str(exc) else type(exc).__name__
                raise HTTPException(
                    status_code=503,
                    detail=f"Deepzzzz isn't set up yet. Check LLM_MODEL and your API key in .env. Reason: {reason}",
                ) from exc
        return app.state.bot

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/chat", response_model=ChatResponse)
    def chat(req: ChatRequest) -> ChatResponse:
        return ChatResponse(**get_bot().chat(req.message).to_dict())

    @app.get("/history")
    def history() -> list[dict]:
        return get_bot().transcript()

    @app.post("/reset")
    def reset() -> dict:
        get_bot().reset()
        return {"status": "reset"}

    @app.get("/memories")
    def memories() -> list[dict]:
        return [{"id": m.id, "content": m.content} for m in get_bot().store.list_memories(limit=500)]

    @app.delete("/memories/{memory_id}")
    def delete_memory(memory_id: int) -> dict:
        try:
            get_bot().store.delete_memory(memory_id)
        except StoreError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        return {"status": "deleted"}

    @app.get("/reminders")
    def reminders() -> list[dict]:
        return [{"id": r.id, "text": r.text, "due": r.due.isoformat(timespec="minutes") if r.due else None}
                for r in get_bot().store.list_reminders()]

    @app.post("/reminders/{reminder_id}/done")
    def reminder_done(reminder_id: int) -> dict:
        try:
            get_bot().store.complete_reminder(reminder_id)
        except StoreError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        return {"status": "done"}

    @app.get("/moods")
    def moods(days: int = 14) -> list[dict]:
        days = max(1, min(days, 90))
        return [{"date": d.day.isoformat(), "average": d.average,
                 "entries": [{"score": m.score, "label": m.label, "note": m.note,
                              "time": m.created_at.strftime("%H:%M")} for m in d.entries]}
                for d in get_bot().store.mood_days(days)]

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    return app


app = create_app()
