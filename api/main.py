"""Chat UI v1.0 for the house-value assistant: one page (static/index.html) and one
endpoint that runs a question through assistant.ask.

The page keeps the conversation and sends it with each question, as text turns the way
assistant.chat keeps them, so the server holds no sessions.

    uv run python -m api.main        # then open http://127.0.0.1:8000
"""

import pathlib
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from src.components import assistant

VERSION = "1.0"
PAGE = pathlib.Path(__file__).parent / "static" / "index.html"


@asynccontextmanager
async def lifespan(app):
    assistant.warm_up()  # data, models and the Ollama model load while the page opens
    yield


app = FastAPI(title="House Value Assistant", version=VERSION, lifespan=lifespan)


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    question: str
    history: list[Turn] = []


@app.get("/")
def page():
    return FileResponse(PAGE)


@app.post("/api/chat")
def chat(request: ChatRequest):
    # a plain def, so FastAPI runs it in a worker thread: an answer takes several seconds
    try:
        answer = assistant.ask(request.question, [turn.model_dump() for turn in request.history])
    except SystemExit as e:  # the assistant's message for a missing Ollama server or model
        raise HTTPException(status_code=503, detail=str(e))
    return {"answer": answer}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
