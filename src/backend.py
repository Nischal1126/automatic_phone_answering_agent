"""
backend.py

FastAPI backend for the KU Dept of AI RAG voice agent.

Strict split, by design:
    - Text chat  -> RAG only. No STT, no TTS. Fast, silent, just an answer.
    - Voice chat -> full pipeline: STT -> RAG -> TTS. Always returns both
                    the spoken-back audio and the text, because the whole
                    point of clicking "speak" is to get a voice reply.

This mirrors how the caller-facing product will actually work: someone
typing a question wants a quick text answer; someone using voice wants
the full conversational experience read back to them.

Assumes main.py (same folder) exposes:
    build_rag_chain() -> (chain, retriever)
    answer_query(chain, query: str) -> str
    load_stt_model() -> faster_whisper.WhisperModel

Install (on top of what main.py needs):
    pip install fastapi "uvicorn[standard]" python-multipart edge-tts

Run:
    uvicorn backend:app --host 0.0.0.0 --port 8000 --reload
"""

import base64
import io
import logging
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import edge_tts
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from main import build_rag_chain, answer_query, load_stt_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("voice-agent-backend")

EDGE_TTS_VOICE = "en-US-AriaNeural"

# ---------------------------------------------------------------------------
# App state — heavy resources loaded once at startup, not per-request
# ---------------------------------------------------------------------------

state: dict = {"chain": None, "retriever": None, "stt_model": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading RAG pipeline...")
    chain, retriever = build_rag_chain()
    state["chain"] = chain
    state["retriever"] = retriever

    logger.info("Loading speech-to-text model...")
    state["stt_model"] = load_stt_model()

    logger.info("Startup complete.")
    yield
    state.clear()


app = FastAPI(title="KU Dept of AI Voice Agent Backend", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten this to your real frontend origin(s) in production
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"service": "voice-agent-backend", "docs": "/docs", "health": "/health"}


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class TextQuery(BaseModel):
    query: str


class TextResponse(BaseModel):
    response: str


class VoiceChatResponse(BaseModel):
    transcript: str
    response: str
    audio_base64: str  # always present — voice mode always speaks the reply


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _transcribe(audio_bytes: bytes) -> str:
    segments, _ = state["stt_model"].transcribe(io.BytesIO(audio_bytes), language="en")
    return " ".join(seg.text.strip() for seg in segments).strip()


async def _synthesize_to_base64(text: str) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    tmp.close()
    try:
        communicate = edge_tts.Communicate(text, EDGE_TTS_VOICE)
        await communicate.save(tmp.name)
        data = Path(tmp.name).read_bytes()
        return base64.b64encode(data).decode("utf-8")
    finally:
        Path(tmp.name).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "rag_ready": state["chain"] is not None,
        "stt_ready": state["stt_model"] is not None,
    }


@app.post("/chat/text", response_model=TextResponse)
async def chat_text(payload: TextQuery):
    """
    Text-only path: RAG answer, nothing else. No STT (there's no audio to
    transcribe) and no TTS (typed questions get typed answers).
    """
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")
    try:
        response = answer_query(state["chain"], payload.query)
    except Exception as exc:  # noqa: BLE001
        logger.exception("answer_query failed")
        raise HTTPException(status_code=500, detail=f"RAG pipeline error: {exc}") from exc
    return TextResponse(response=response)


@app.post("/chat/voice", response_model=VoiceChatResponse)
async def chat_voice(audio: UploadFile = File(...)):
    """
    Voice path: the full pipeline, always. STT -> RAG -> TTS. This only
    runs when the user explicitly records/uploads audio (i.e. clicked
    "speak" in the UI) — there's no toggle here because voice mode always
    implies a spoken reply.
    """
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="empty audio upload")

    try:
        transcript = _transcribe(audio_bytes)
    except Exception as exc:  # noqa: BLE001
        logger.exception("transcription failed")
        raise HTTPException(status_code=500, detail=f"STT error: {exc}") from exc

    if not transcript:
        raise HTTPException(status_code=422, detail="couldn't make out any speech in that recording")

    try:
        response_text = answer_query(state["chain"], transcript)
    except Exception as exc:  # noqa: BLE001
        logger.exception("answer_query failed")
        raise HTTPException(status_code=500, detail=f"RAG pipeline error: {exc}") from exc

    try:
        audio_b64 = await _synthesize_to_base64(response_text)
    except Exception as exc:  # noqa: BLE001
        logger.exception("speech synthesis failed")
        raise HTTPException(status_code=500, detail=f"TTS error: {exc}") from exc

    return VoiceChatResponse(
        transcript=transcript,
        response=response_text,
        audio_base64=audio_b64,
    )